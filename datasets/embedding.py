import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
from tqdm import tqdm
from datasets.utils import logger
import faiss
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

# Constants for better maintainability
DEFAULT_GENRE_DIM = 64
DEFAULT_PLOT_WEIGHT = 0.8
DEFAULT_GENRE_WEIGHT = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_E5_PREFIX = "passage: "
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-large"
DEFAULT_DEVICE = "cpu"
DEFAULT_THREADS = 16

class MovieEmbeddings:
    def __init__(self, 
                 data_df: pl.LazyFrame | pl.DataFrame, 
                 batch_size: int = DEFAULT_BATCH_SIZE, 
                 checkpoint_path: str | None = None,
                 model_name: str = DEFAULT_MODEL_NAME,
                 device: str = DEFAULT_DEVICE,
                 e5_prefix: str = DEFAULT_E5_PREFIX,
                 random_state: int = DEFAULT_RANDOM_STATE):
        """
        Initialize the MovieEmbeddings generator.
        
        Args:
            data_df: DataFrame containing movie data with 'tid', 'plot', and 'genre' columns
            batch_size: Batch size for embedding generation
            checkpoint_path: Optional path for saving checkpoints
            model_name: SentenceTransformer model to use for plot embeddings
            device: Device to use for embedding generation ('cpu', 'cuda', 'mps')
            e5_prefix: Prefix to add to text before embedding (for E5 models)
            random_state: Random state for reproducible results
        """
        self._validate_input_data(data_df)
        self.df = data_df
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device
        self.e5_prefix = e5_prefix
        self.random_state = random_state
        self._model: SentenceTransformer | None = None
        
        from datasets.utils import CheckpointManager
        self.checkpoint_manager = CheckpointManager(Path(checkpoint_path)) if checkpoint_path else None
        self.embeddings_df: pl.DataFrame | None = None
    
    def _validate_input_data(self, data_df: pl.LazyFrame | pl.DataFrame) -> None:
        """Validate that input DataFrame has required columns."""
        lazy = data_df.lazy() if isinstance(data_df, pl.DataFrame) else data_df
        columns = lazy.collect_schema().names()
        
        required_columns = {"tid"}
        missing_columns = required_columns - set(columns)
        if missing_columns:
            raise ValueError(f"Input DataFrame missing required columns: {missing_columns}")
        
        # Check if we have at least one of plot or genre
        optional_columns = {"plot", "genre"}
        available_optional = optional_columns.intersection(set(columns))
        if not available_optional:
            raise ValueError(f"Input DataFrame must have at least one of: {optional_columns}")
        
        logger.info(f"Input validation passed. Available columns: {available_optional}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name} on device: {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def gen_plot_emb(self) -> None:
        """
        Stream through the dataset in batches, generate embeddings, checkpoint after each batch,
        and return the full embeddings DataFrame (float32).
        
        Raises:
            ValueError: If no plot data is available
        """
        logger.info("Generating plot embeddings in streaming batches...")
        
        # Prepare lazy DataFrame
        lazy = self.df.lazy() if isinstance(self.df, pl.DataFrame) else self.df
        
        # Validate plot column exists
        if "plot" not in lazy.collect_schema().names():
            raise ValueError("Plot column not found in input data")
            
        lazy = lazy.select(["tid", "plot"]).filter(pl.col("plot").is_not_null())
        
        # Compute total rows
        total = int(lazy.select(pl.len().alias("n")).collect()["n"][0])
        if total == 0:
            raise ValueError("No movies with plot data found")
        
        # Resume from checkpoint if exists
        offset = 0
        cumulative: pl.DataFrame | None = None
        if self.checkpoint_manager and self.checkpoint_manager.exists():
            cumulative = self.checkpoint_manager.load()
            offset = cumulative.height
            logger.info(f"Resuming from checkpoint with {offset} embeddings.")
        
        # Initialize progress bar
        pbar = tqdm(total=total, desc="Plot embedding progress", unit="plots", initial=offset)
        
        try:
            while offset < total:
                # Use slice for offset and limit
                batch = lazy.slice(offset, self.batch_size).collect(engine="streaming")
                if batch.is_empty():
                    break
                
                # Prefix text for embedding (E5 model requirement)
                batch = batch.with_columns(
                    (pl.lit(self.e5_prefix) + pl.col("plot")).alias("prefixed_plot")
                )
                texts = batch["prefixed_plot"].to_list()
                tids = batch["tid"].to_list()
                
                # Generate embeddings
                embs = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    device=self.device,
                    convert_to_numpy=True,
                    precision="float32",
                )
                
                # Ensure float32 and normalize
                embs = embs.astype("float32")
                faiss.normalize_L2(embs)
                
                # Create batch DataFrame
                batch_df = pl.DataFrame({
                    "tid": tids, 
                    "plot_embedding": embs.tolist()
                })
                
                # Accumulate results
                if cumulative is None:
                    cumulative = batch_df
                else:
                    cumulative = pl.concat([cumulative, batch_df])
                
                # Checkpoint and update progress
                if self.checkpoint_manager:
                    self.checkpoint_manager.update(cumulative)
                    logger.debug(f"Checkpoint saved with {cumulative.height} embeddings.")
                
                pbar.update(batch.height)
                offset += self.batch_size
                
        except KeyboardInterrupt:
            # Checkpoint on interrupt
            if self.checkpoint_manager and cumulative is not None:
                self.checkpoint_manager.update(cumulative)
                logger.info(f"Checkpoint saved on interrupt with {cumulative.height} embeddings.")
            raise
        finally:
            pbar.close()
        
        # Update or create embeddings DataFrame
        if self.embeddings_df is not None:
            # Join with existing embeddings
            self.embeddings_df = self.embeddings_df.join(cumulative, on="tid", how="full")
        else:
            self.embeddings_df = cumulative if cumulative is not None else pl.DataFrame(
                [], schema=[("tid", pl.UInt32), ("plot_embedding", pl.List(pl.Float32))]
            )

    def _create_nca_labels_from_plots(self, tids: list, target_dim: int) -> np.ndarray | None:
        """
        Create NCA labels by clustering plot embeddings.
        
        Args:
            tids: List of movie IDs that need labels
            target_dim: Number of clusters to create
            
        Returns:
            Array of cluster labels, or None if insufficient data
        """
        if (self.embeddings_df is None or 
            "plot_embedding" not in self.embeddings_df.columns):
            return None
            
        # Get plot embeddings for movies that have genres
        tid_to_plot = dict(zip(
            self.embeddings_df["tid"].to_list(),
            self.embeddings_df["plot_embedding"].to_list()
        ))
        
        plot_vectors = []
        valid_indices = []
        for i, tid in enumerate(tids):
            if tid in tid_to_plot:
                plot_vectors.append(tid_to_plot[tid])
                valid_indices.append(i)
        
        if len(plot_vectors) <= target_dim:
            logger.warning(f"Insufficient plot vectors ({len(plot_vectors)}) for clustering into {target_dim} groups")
            return None
            
        # Cluster plot embeddings to create meaningful labels
        plot_matrix = np.array(plot_vectors, dtype=np.float32)
        kmeans = KMeans(n_clusters=target_dim, random_state=self.random_state, n_init=10)
        plot_labels = kmeans.fit_predict(plot_matrix)
        
        # Create labels array for all genre vectors
        labels = np.zeros(len(tids), dtype=int)
        for i, valid_idx in enumerate(valid_indices):
            labels[valid_idx] = plot_labels[i]
        
        logger.info(f"Created {target_dim} plot-based clusters for NCA")
        return labels
    
    def _create_fallback_labels(self, genre_matrix: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Create fallback NCA labels based on genre diversity.
        
        Args:
            genre_matrix: One-hot encoded genre matrix
            target_dim: Target number of label groups
            
        Returns:
            Array of labels based on genre diversity
        """
        genre_diversity = np.sum(genre_matrix, axis=1)  # Number of genres per movie
        if genre_diversity.max() > 1:
            labels = np.digitize(genre_diversity, bins=np.linspace(1, genre_diversity.max(), target_dim))
        else:
            # Fallback if all movies have only one genre
            labels = np.zeros(len(genre_matrix), dtype=int)
        return labels

    def gen_genre_emb(self, target_dim: int = DEFAULT_GENRE_DIM, use_plot_correlation: bool = True) -> None:
        """
        Generate enriched genre embeddings by combining one-hot genre vectors with plot-aware sub-genre clusters.
        
        This method creates richer genre representations by:
        1. Starting with traditional one-hot genre vectors (23 dimensions for your dataset)
        2. Using plot embeddings to create sub-genre clusters within each genre
        3. Combining these to create plot-aware genre embeddings that distinguish nuanced differences
           (e.g., Fast & Furious vs Indiana Jones both being Action/Adventure)
        
        Args:
            target_dim: Target dimension for final genre vectors. The method will create
                       enriched representations up to this dimension. (default: 32)
            use_plot_correlation: If True and plot embeddings exist, create plot-aware 
                                sub-genre clusters for richer representations (default: True)
        """
        logger.info(f"Generating enriched genre embeddings up to {target_dim} dimensions...")
        
        # Prepare DataFrame
        lazy = self.df.lazy() if isinstance(self.df, pl.DataFrame) else self.df
        genre_data = lazy.select(["tid", "genre"]).filter(pl.col("genre").is_not_null()).collect()
        
        # Extract all unique genres from the dataset
        all_genres = set()
        for genres in genre_data["genre"].to_list():
            if genres:
                if isinstance(genres, list):
                    all_genres.update(genres)
                elif isinstance(genres, str):
                    # Handle comma-separated string format
                    all_genres.update([g.strip() for g in genres.split(",")])
        
        all_genres = sorted(list(all_genres))
        genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
        logger.info(f"Found {len(all_genres)} unique genres: {all_genres[:10]}...")
        
        # Create one-hot encoding for each movie
        genre_vectors = []
        tids = []
        
        for row in genre_data.iter_rows(named=True):
            tid = row["tid"]
            genres = row["genre"]
            
            # Create one-hot vector
            one_hot = np.zeros(len(all_genres), dtype=np.float32)
            
            if genres:
                movie_genres = genres if isinstance(genres, list) else [g.strip() for g in genres.split(",")]
                for genre in movie_genres:
                    if genre in genre_to_idx:
                        one_hot[genre_to_idx[genre]] = 1.0
            
            genre_vectors.append(one_hot)
            tids.append(tid)
        
        # Convert to numpy array
        genre_matrix = np.array(genre_vectors, dtype=np.float32)
        logger.info(f"Created base genre matrix with shape: {genre_matrix.shape}")
        
        # Generate enriched genre embeddings using plot information
        if (use_plot_correlation and 
            self.embeddings_df is not None and 
            "plot_embedding" in self.embeddings_df.columns):
            
            logger.info("Creating plot-aware enriched genre embeddings...")
            genre_enriched = self._create_plot_enriched_genres(genre_matrix, tids, all_genres, target_dim)
        else:
            logger.info("Plot embeddings not available, using standard genre representation...")
            # Fall back to simple padding or compression
            if len(all_genres) <= target_dim:
                if len(all_genres) < target_dim:
                    padding = np.zeros((genre_matrix.shape[0], target_dim - len(all_genres)), dtype=np.float32)
                    genre_enriched = np.hstack([genre_matrix, padding])
                    logger.info(f"Padded genre vectors to {target_dim} dimensions")
                else:
                    genre_enriched = genre_matrix
                    logger.info(f"Using full genre representation with {len(all_genres)} dimensions")
            else:
                # Simple truncation if too many genres
                genre_enriched = genre_matrix[:, :target_dim]
                logger.info(f"Truncated genre vectors to {target_dim} dimensions")
        
        # Normalize the final vectors
        genre_enriched = normalize(genre_enriched, norm='l2')
        
        # Create DataFrame with genre embeddings
        genre_embeddings_df = pl.DataFrame({
            "tid": tids,
            "genre_embedding": pl.Series(genre_enriched.tolist())
        })
        
        # Update or create embeddings DataFrame
        if self.embeddings_df is not None:
            # Join with existing embeddings
            self.embeddings_df = self.embeddings_df.join(genre_embeddings_df, on="tid", how="full")
        else:
            self.embeddings_df = genre_embeddings_df
        
        logger.info(f"Enriched genre embeddings generated for {len(tids)} movies with {genre_enriched.shape[1]} dimensions")
    
    def _fallback_nca_compression(self, genre_matrix: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Fallback method for NCA compression based on genre diversity.
        
        Args:
            genre_matrix: One-hot encoded genre matrix
            target_dim: Target dimension for compression
            
        Returns:
            Compressed genre matrix
        """
        # Fallback to genre diversity
        genre_diversity = np.sum(genre_matrix, axis=1)
        if genre_diversity.max() > 1:
            labels = np.digitize(genre_diversity, bins=np.linspace(1, genre_diversity.max(), target_dim))
        else:
            labels = np.zeros(len(genre_matrix), dtype=int)
        
        # Standardize the data first
        scaler = StandardScaler()
        genre_matrix_scaled = scaler.fit_transform(genre_matrix)
        
        # Apply NCA
        nca = NeighborhoodComponentsAnalysis(n_components=target_dim, random_state=self.random_state)
        genre_compressed = nca.fit_transform(genre_matrix_scaled, labels)
        
        return genre_compressed
    
    def merge(self, plot_weight: float = DEFAULT_PLOT_WEIGHT, genre_weight: float = DEFAULT_GENRE_WEIGHT):
        """
        Merge plot and genre embeddings with specified weights into a combined embedding.
        
        Args:
            plot_weight: Weight for plot embeddings (default: DEFAULT_PLOT_WEIGHT)
            genre_weight: Weight for genre embeddings (default: DEFAULT_GENRE_WEIGHT)
        """
        if self.embeddings_df is None:
            raise ValueError("Embeddings DataFrame is not generated yet.")
        
        # Check if both embedding types exist
        has_plot = "plot_embedding" in self.embeddings_df.columns
        has_genre = "genre_embedding" in self.embeddings_df.columns
        
        if not has_plot or not has_genre:
            logger.warning("Both plot and genre embeddings must be present for merging.")
            return
        
        # Validate weights
        if plot_weight < 0 or genre_weight < 0:
            raise ValueError("Weights must be non-negative.")
        
        total_weight = plot_weight + genre_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be positive.")
        
        # Normalize weights to sum to 1.0
        plot_weight_norm = plot_weight / total_weight
        genre_weight_norm = genre_weight / total_weight
        
        logger.info(f"Merging embeddings with weights - Plot: {plot_weight_norm:.2f}, Genre: {genre_weight_norm:.2f}")
        
        # Extract embeddings as numpy arrays
        plot_embeddings = np.vstack(self.embeddings_df["plot_embedding"].to_list()).astype("float32")
        genre_embeddings = np.vstack(self.embeddings_df["genre_embedding"].to_list()).astype("float32")
        
        # Combine embeddings with weights
        combined_embeddings = np.hstack([
            plot_embeddings * plot_weight_norm,
            genre_embeddings * genre_weight_norm
        ])
        
        # Normalize the combined embeddings
        faiss.normalize_L2(combined_embeddings)
        
        # Add combined embeddings to DataFrame
        self.embeddings_df = self.embeddings_df.with_columns(
            pl.Series("combined_embedding", combined_embeddings.tolist())
        )
        
        logger.info(f"Combined embeddings created with shape: {combined_embeddings.shape}")
    
    def _create_plot_enriched_genres(self, genre_matrix: np.ndarray, tids: list, all_genres: list, target_dim: int) -> np.ndarray:
        """
        Create enriched genre embeddings by using plot embeddings to create sub-genre clusters.
        
        This method:
        1. Groups movies by their primary genre combinations
        2. Within each genre group, clusters movies by plot similarity
        3. Creates enriched vectors that combine genre information with plot-based sub-clusters
        
        Args:
            genre_matrix: One-hot encoded genre matrix
            tids: List of movie IDs
            all_genres: List of all unique genres
            target_dim: Target dimension for enriched vectors
            
        Returns:
            Enriched genre matrix with plot-aware sub-genre information
        """
        # Get plot embeddings for movies
        tid_to_plot = dict(zip(
            self.embeddings_df["tid"].to_list(),
            self.embeddings_df["plot_embedding"].to_list()
        ))
        
        # Start with base genre matrix
        enriched_vectors = []
        
        # Group movies by their genre combinations
        genre_groups = {}
        for i, (tid, genre_vec) in enumerate(zip(tids, genre_matrix)):
            # Create a key from the genre vector (which genres are active)
            active_genres = tuple(np.where(genre_vec > 0)[0])
            if active_genres not in genre_groups:
                genre_groups[active_genres] = []
            genre_groups[active_genres].append((i, tid, genre_vec))
        
        logger.info(f"Found {len(genre_groups)} unique genre combinations")
        
        # For each genre group, create sub-clusters based on plot similarity
        for genre_combo, movies in genre_groups.items():
            if len(movies) <= 1:
                # Single movie or empty group - use original genre vector
                for movie_idx, tid, genre_vec in movies:
                    enriched_vec = self._create_single_movie_enriched_vector(genre_vec, target_dim)
                    enriched_vectors.append((movie_idx, enriched_vec))
                continue
            
            # Get plot embeddings for movies in this genre group
            plot_vecs = []
            valid_movies = []
            for movie_idx, tid, genre_vec in movies:
                if tid in tid_to_plot:
                    plot_vecs.append(tid_to_plot[tid])
                    valid_movies.append((movie_idx, tid, genre_vec))
            
            if len(plot_vecs) <= 1:
                # Not enough plot embeddings - use original genre vectors
                for movie_idx, tid, genre_vec in movies:
                    enriched_vec = self._create_single_movie_enriched_vector(genre_vec, target_dim)
                    enriched_vectors.append((movie_idx, enriched_vec))
                continue
            
            # Determine number of sub-clusters for this genre group
            # More movies = more sub-clusters, but limited by available dimensions
            remaining_dims = target_dim - len(all_genres)
            max_subclusters = min(max(2, len(plot_vecs) // 5), max(1, remaining_dims))
            n_subclusters = min(len(plot_vecs), max_subclusters)
            
            if n_subclusters > 1:
                # Cluster plot embeddings to create sub-genres
                plot_matrix = np.array(plot_vecs, dtype=np.float32)
                kmeans = KMeans(n_clusters=n_subclusters, random_state=self.random_state, n_init=10)
                subcluster_labels = kmeans.fit_predict(plot_matrix)
                
                # Create enriched vectors for each movie in this genre group
                for i, (movie_idx, tid, genre_vec) in enumerate(valid_movies):
                    subcluster_id = subcluster_labels[i]
                    enriched_vec = self._create_enriched_vector(genre_vec, all_genres, genre_combo, subcluster_id, n_subclusters, target_dim)
                    enriched_vectors.append((movie_idx, enriched_vec))
                
                # Handle movies without plot embeddings
                for movie_idx, tid, genre_vec in movies:
                    if tid not in tid_to_plot:
                        enriched_vec = self._create_single_movie_enriched_vector(genre_vec, target_dim)
                        enriched_vectors.append((movie_idx, enriched_vec))
            else:
                # Not enough movies for clustering - use original genre vectors
                for movie_idx, tid, genre_vec in movies:
                    enriched_vec = self._create_single_movie_enriched_vector(genre_vec, target_dim)
                    enriched_vectors.append((movie_idx, enriched_vec))
        
        # Sort by original movie index and extract vectors
        enriched_vectors.sort(key=lambda x: x[0])
        final_matrix = np.array([vec for _, vec in enriched_vectors], dtype=np.float32)
        
        logger.info(f"Created plot-enriched genre matrix with shape: {final_matrix.shape}")
        return final_matrix
    
    def _create_enriched_vector(self, genre_vec: np.ndarray, all_genres: list, genre_combo: tuple, 
                               subcluster_id: int, n_subclusters: int, target_dim: int) -> np.ndarray:
        """
        Create an enriched genre vector that combines base genre info with sub-cluster information.
        
        Args:
            genre_vec: Original one-hot genre vector
            all_genres: List of all genres
            genre_combo: Tuple of active genre indices
            subcluster_id: ID of the sub-cluster within this genre group
            n_subclusters: Total number of sub-clusters for this genre group
            target_dim: Target dimension for the enriched vector
            
        Returns:
            Enriched genre vector
        """
        # Start with base genre vector
        enriched = np.zeros(target_dim, dtype=np.float32)
        
        # Copy original genre information
        genre_portion = min(len(all_genres), target_dim)
        enriched[:genre_portion] = genre_vec[:genre_portion]
        
        # Add sub-cluster information in remaining dimensions
        if genre_portion < target_dim:
            # Create a unique sub-cluster signature for this genre combination and cluster
            remaining_dims = target_dim - genre_portion
            
            # Use hash of genre combination and subcluster to create consistent sub-cluster features
            import hashlib
            combo_hash = hashlib.md5(str(genre_combo).encode()).hexdigest()
            cluster_hash = hashlib.md5(f"{combo_hash}_{subcluster_id}".encode()).hexdigest()
            
            # Convert hash to numeric features
            for i in range(min(remaining_dims, 8)):  # Use up to 8 dimensions for sub-cluster info
                byte_val = int(cluster_hash[i*2:(i+1)*2], 16)
                enriched[genre_portion + i] = (byte_val / 255.0) * 0.5  # Scale to [0, 0.5]
        
        return enriched
    
    def _create_single_movie_enriched_vector(self, genre_vec: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Create an enriched vector for a single movie (no clustering possible).
        
        Args:
            genre_vec: Original one-hot genre vector
            target_dim: Target dimension for the enriched vector
            
        Returns:
            Enriched genre vector (padded if necessary)
        """
        enriched = np.zeros(target_dim, dtype=np.float32)
        genre_portion = min(len(genre_vec), target_dim)
        enriched[:genre_portion] = genre_vec[:genre_portion]
        return enriched

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from datasets.utils import logger, read_parquet_file, write_parquet_file

    parser = argparse.ArgumentParser(description="Generate embeddings for movie plots and genres.")
    parser.add_argument('--input', type=str, default="datasets/dist/movies_processed_sn.parquet", 
                        help="Input Parquet file with movie data.")
    parser.add_argument('--output', type=str, default="datasets/dist/movies_embeddings.parquet", 
                        help="Output Parquet file to write embeddings.")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, 
                        help=f"Batch size for embedding model (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS, 
                        help=f"Number of CPU threads to use (default: {DEFAULT_THREADS}).")
    parser.add_argument('--checkpoint-path', type=str, default="datasets/dist/movies_embeddings.cp.parquet", 
                        help="Path to save embeddings checkpoint (optional).")
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME,
                        help=f"SentenceTransformer model name for plot embeddings (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help=f"Device to use for embedding generation ('cpu', 'cuda', 'mps') (default: {DEFAULT_DEVICE}).")
    parser.add_argument('--e5-prefix', type=str, default=DEFAULT_E5_PREFIX,
                        help=f"Prefix to add to text before embedding for E5 models (default: '{DEFAULT_E5_PREFIX}').")
    parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE,
                        help=f"Random state for reproducible results (default: {DEFAULT_RANDOM_STATE}).")
    parser.add_argument('--plot-only', action='store_true', 
                        help="Generate only plot embeddings.")
    parser.add_argument('--genre-only', action='store_true', 
                        help="Generate only genre embeddings.")
    parser.add_argument('--genre-dim', type=int, default=DEFAULT_GENRE_DIM, 
                        help=f"Target dimension for enriched genre embeddings. Base genres are preserved and enhanced with plot-aware sub-clusters. (default: {DEFAULT_GENRE_DIM}).")
    parser.add_argument('--no-plot-correlation', action='store_true',
                        help="Disable plot-aware genre enrichment. Use only basic one-hot genre vectors.")
    parser.add_argument('--merge', action='store_true', 
                        help="Merge plot and genre embeddings into combined embedding.")
    parser.add_argument('--plot-weight', type=float, default=DEFAULT_PLOT_WEIGHT, 
                        help=f"Weight for plot embeddings when merging (default: {DEFAULT_PLOT_WEIGHT}).")
    parser.add_argument('--genre-weight', type=float, default=DEFAULT_GENRE_WEIGHT, 
                        help=f"Weight for genre embeddings when merging (default: {DEFAULT_GENRE_WEIGHT}).")
    args = parser.parse_args()

    # Validate arguments
    if args.plot_only and args.genre_only:
        logger.error("Cannot specify both --plot-only and --genre-only.")
        exit(1)
    
    if args.merge and (args.plot_only or args.genre_only):
        logger.error("Cannot use --merge with --plot-only or --genre-only.")
        exit(1)

    # Configure CPU threads according to user settings
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Read data
    logger.info(f"Reading data from {args.input}")
    df = read_parquet_file(Path(args.input), lazy=False)

    # Generate embeddings
    generator = MovieEmbeddings(
        df, 
        batch_size=args.batch_size, 
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        device=args.device,
        e5_prefix=args.e5_prefix,
        random_state=args.random_state
    )
    
    try:
        if args.genre_only:
            # Generate only genre embeddings
            generator.gen_genre_emb(
                target_dim=args.genre_dim, 
                use_plot_correlation=not args.no_plot_correlation
            )
        elif args.plot_only:
            # Generate only plot embeddings
            generator.gen_plot_emb()
        else:
            # Generate both plot and genre embeddings
            generator.gen_plot_emb()
            generator.gen_genre_emb(
                target_dim=args.genre_dim, 
                use_plot_correlation=not args.no_plot_correlation
            )
            
            # Merge embeddings if requested
            if args.merge:
                generator.merge(plot_weight=args.plot_weight, genre_weight=args.genre_weight)
                
    except KeyboardInterrupt:
        logger.warning("Embedding generation interrupted.")
        exit(1)
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        exit(1)

    # Ensure we have embeddings to save
    if generator.embeddings_df is None or generator.embeddings_df.is_empty():
        logger.error("No embeddings were generated.")
        exit(1)

    # Save embeddings
    logger.info(f"Saving embeddings to {args.output}")
    write_parquet_file(generator.embeddings_df, Path(args.output))
    logger.info(f"Embeddings saved successfully. Shape: {generator.embeddings_df.shape}")

