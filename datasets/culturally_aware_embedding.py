import polars as pl
import numpy as np
import faiss
from datasets.utils import logger, read_parquet_file, write_parquet_file
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import hashlib
import json
from typing import List

class CulturallyAwareMovieEmbedding:
    def __init__(self, plot_dim=1024, genre_dim=64, localization_dim=32, 
                 plot_weight=0.50, genre_weight=0.25, localization_weight=0.05, popularity_weight=0.20):
        """
        Culturally-aware movie embedding that balances semantic similarity with cultural relevance
        
        This system addresses the core problem: semantic similarity alone finds niche films
        instead of culturally iconic ones. It combines:
        1. Plot embeddings (semantic meaning)
        2. Enriched genre embeddings (thematic categories)  
        3. Localization embeddings (cultural/regional relevance)
        4. Popularity boost (promotes culturally iconic films)
        
        Args:
            plot_dim: Dimension of plot embeddings (semantic content)
            genre_dim: Target dimension for enriched genre embeddings
            localization_dim: Target dimension for country/language embeddings
            plot_weight: Weight for plot embeddings (primary semantic signal)
            genre_weight: Weight for genre embeddings (thematic relevance)
            localization_weight: Weight for localization (cultural relevance)
            popularity_weight: Weight for popularity boost (promotes iconic films)
        """
        self.plot_dim = plot_dim
        self.genre_dim = genre_dim
        self.localization_dim = localization_dim
        self.plot_weight = plot_weight
        self.genre_weight = genre_weight
        self.localization_weight = localization_weight
        self.popularity_weight = popularity_weight
        
        # Ensure weights sum to 1
        total_weight = plot_weight + genre_weight + localization_weight + popularity_weight
        self.plot_weight /= total_weight
        self.genre_weight /= total_weight
        self.localization_weight /= total_weight
        self.popularity_weight /= total_weight
        
        # Vocabularies for cultural features
        self.country_vocab = {}
        self.language_vocab = {}
        
        # Final embedding dimension
        self.final_dim = plot_dim + genre_dim + localization_dim + 1  # +1 for popularity scalar
        
        logger.info(f"Culturally-aware embedding dimensions: plot={plot_dim}, genre={genre_dim}, localization={localization_dim}, popularity=1")
        logger.info(f"Weights: plot={self.plot_weight:.3f}, genre={self.genre_weight:.3f}, localization={self.localization_weight:.3f}, popularity={self.popularity_weight:.3f}")
    
    def build_localization_vocabularies(self, movies_df):
        """Build vocabularies for countries and languages"""
        logger.info("Building localization vocabularies...")
        
        # Extract all unique countries
        all_countries = set()
        for country_list in movies_df["country"].to_list():
            if country_list and isinstance(country_list, list):
                all_countries.update([c.strip() for c in country_list if c])
            elif country_list and isinstance(country_list, str) and country_list != "":
                countries = country_list.split(",")
                all_countries.update([c.strip() for c in countries])
        
        self.country_vocab = {country: idx for idx, country in enumerate(sorted(all_countries))}
        logger.info(f"Country vocabulary size: {len(self.country_vocab)}")
        
        # Extract all unique languages
        all_languages = set()
        for language_list in movies_df["language"].to_list():
            if language_list and isinstance(language_list, list):
                all_languages.update([l.strip() for l in language_list if l])
            elif language_list and isinstance(language_list, str) and language_list != "":
                languages = language_list.split(",")
                all_languages.update([l.strip() for l in languages])
        
        self.language_vocab = {language: idx for idx, language in enumerate(sorted(all_languages))}
        logger.info(f"Language vocabulary size: {len(self.language_vocab)}")
    
    def _create_localization_enriched_embeddings(self, movies_df, plot_embeddings, target_dim):
        """
        Create enriched localization embeddings using the same approach as enriched genres.
        Groups movies by country/language combinations and creates sub-clusters based on plot similarity.
        """
        logger.info("Creating plot-enriched localization embeddings...")
        
        # Create combined localization matrix (countries + languages)
        total_localization_features = len(self.country_vocab) + len(self.language_vocab)
        localization_matrix = np.zeros((len(movies_df), total_localization_features), dtype=np.float32)
        
        # Fill localization matrix
        for i, (country_data, language_data) in enumerate(zip(movies_df["country"].to_list(), movies_df["language"].to_list())):
            # Process countries
            if country_data and isinstance(country_data, list):
                for country in country_data:
                    if country and country in self.country_vocab:
                        localization_matrix[i, self.country_vocab[country]] = 1.0
            elif country_data and isinstance(country_data, str) and country_data != "":
                countries = [c.strip() for c in country_data.split(",")]
                for country in countries:
                    if country in self.country_vocab:
                        localization_matrix[i, self.country_vocab[country]] = 1.0
            
            # Process languages (offset by country vocab size)
            language_offset = len(self.country_vocab)
            if language_data and isinstance(language_data, list):
                for language in language_data:
                    if language and language in self.language_vocab:
                        localization_matrix[i, language_offset + self.language_vocab[language]] = 1.0
            elif language_data and isinstance(language_data, str) and language_data != "":
                languages = [l.strip() for l in language_data.split(",")]
                for language in languages:
                    if language in self.language_vocab:
                        localization_matrix[i, language_offset + self.language_vocab[language]] = 1.0
        
        logger.info(f"Created base localization matrix with shape: {localization_matrix.shape}")
        
        # Group movies by their localization combinations
        localization_groups = {}
        for i, loc_vec in enumerate(localization_matrix):
            # Create a key from the localization vector (which features are active)
            active_features = tuple(np.where(loc_vec > 0)[0])
            if active_features not in localization_groups:
                localization_groups[active_features] = []
            localization_groups[active_features].append((i, loc_vec))
        
        logger.info(f"Found {len(localization_groups)} unique localization combinations")
        
        # Create enriched vectors for each group
        enriched_vectors = []
        
        for loc_combo, movies in localization_groups.items():
            if len(movies) <= 1:
                # Single movie - use base vector padded to target dimension
                for movie_idx, loc_vec in movies:
                    enriched_vec = self._create_single_localization_vector(loc_vec, target_dim)
                    enriched_vectors.append((movie_idx, enriched_vec))
                continue
            
            # Get plot embeddings for this localization group
            plot_vecs = []
            valid_movies = []
            for movie_idx, loc_vec in movies:
                if movie_idx < len(plot_embeddings):
                    plot_vecs.append(plot_embeddings[movie_idx])
                    valid_movies.append((movie_idx, loc_vec))
            
            if len(plot_vecs) <= 1:
                # Not enough plot embeddings - use base vectors
                for movie_idx, loc_vec in movies:
                    enriched_vec = self._create_single_localization_vector(loc_vec, target_dim)
                    enriched_vectors.append((movie_idx, enriched_vec))
                continue
            
            # Create sub-clusters based on plot similarity
            remaining_dims = target_dim - total_localization_features
            max_subclusters = min(max(2, len(plot_vecs) // 3), max(1, remaining_dims))
            n_subclusters = min(len(plot_vecs), max_subclusters)
            
            if n_subclusters > 1:
                plot_matrix = np.array(plot_vecs, dtype=np.float32)
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
                subcluster_labels = kmeans.fit_predict(plot_matrix)
                
                # Create enriched vectors
                for i, (movie_idx, loc_vec) in enumerate(valid_movies):
                    subcluster_id = subcluster_labels[i]
                    enriched_vec = self._create_enriched_localization_vector(
                        loc_vec, loc_combo, subcluster_id, n_subclusters, target_dim
                    )
                    enriched_vectors.append((movie_idx, enriched_vec))
                
                # Handle movies without plot embeddings
                for movie_idx, loc_vec in movies:
                    if movie_idx >= len(plot_embeddings):
                        enriched_vec = self._create_single_localization_vector(loc_vec, target_dim)
                        enriched_vectors.append((movie_idx, enriched_vec))
            else:
                # Not enough movies for clustering
                for movie_idx, loc_vec in movies:
                    enriched_vec = self._create_single_localization_vector(loc_vec, target_dim)
                    enriched_vectors.append((movie_idx, enriched_vec))
        
        # Sort by original movie index and extract vectors
        enriched_vectors.sort(key=lambda x: x[0])
        final_matrix = np.array([vec for _, vec in enriched_vectors], dtype=np.float32)
        
        logger.info(f"Created plot-enriched localization matrix with shape: {final_matrix.shape}")
        return final_matrix
    
    def _create_enriched_localization_vector(self, loc_vec, loc_combo, subcluster_id, target_dim):
        """Create enriched localization vector with better dimensionality handling"""
        enriched = np.zeros(target_dim, dtype=np.float32)
        
        # Get active features (countries and languages that are set)
        active_features = np.where(loc_vec > 0)[0]
        
        if len(active_features) > 0:
            # With larger target_dim (e.g., 128), we can be more direct
            if target_dim >= 64:
                # Use a smarter mapping that preserves more locality
                # Map each active feature to multiple dimensions to reduce collisions
                for i, feature_idx in enumerate(active_features):
                    # Map each feature to 2-3 dimensions using different hash functions
                    base_pos = (feature_idx * 17) % (target_dim // 2)  # Use first half
                    enriched[base_pos] = 0.8  # Strong signal
                    
                    # Add secondary mapping to reduce collisions
                    secondary_pos = (feature_idx * 31 + 7) % (target_dim // 2) + (target_dim // 2)
                    enriched[secondary_pos] = 0.6  # Secondary signal
                    
                    # Add a third weak signal for robustness
                    if i < 3:  # Only for first few features to avoid overcrowding
                        tertiary_pos = (feature_idx * 53 + 13) % (target_dim // 4)
                        enriched[tertiary_pos] = max(enriched[tertiary_pos], 0.3)
            else:
                # Fallback to hash-based approach for smaller dimensions
                feature_hash = hashlib.md5(str(sorted(active_features)).encode()).hexdigest()
                for i in range(min(target_dim, 16)):
                    byte_val = int(feature_hash[i*2:(i+1)*2], 16)
                    enriched[i] = (byte_val / 255.0) * 0.7
        
        # Add sub-cluster information in remaining dimensions if we have space
        remaining_start = target_dim - 16 if target_dim > 16 else 0
        if remaining_start > 0 and remaining_start < target_dim:
            combo_hash = hashlib.md5(str(loc_combo).encode()).hexdigest()
            cluster_hash = hashlib.md5(f"{combo_hash}_{subcluster_id}".encode()).hexdigest()
            
            for i in range(min(16, target_dim - remaining_start)):
                byte_val = int(cluster_hash[i*2:(i+1)*2], 16)
                enriched[remaining_start + i] = (byte_val / 255.0) * 0.2  # Weaker sub-cluster signal
        
        return enriched
    
    def _create_single_localization_vector(self, loc_vec, target_dim):
        """Create enriched vector for single movie with better dimensionality handling"""
        enriched = np.zeros(target_dim, dtype=np.float32)
        
        # Find active features
        active_features = np.where(loc_vec > 0)[0]
        
        if len(active_features) > 0:
            if target_dim >= 64:
                # Use the same smarter mapping as above
                for i, feature_idx in enumerate(active_features):
                    base_pos = (feature_idx * 17) % (target_dim // 2)
                    enriched[base_pos] = 0.8
                    
                    secondary_pos = (feature_idx * 31 + 7) % (target_dim // 2) + (target_dim // 2)
                    enriched[secondary_pos] = 0.6
                    
                    if i < 3:
                        tertiary_pos = (feature_idx * 53 + 13) % (target_dim // 4)
                        enriched[tertiary_pos] = max(enriched[tertiary_pos], 0.3)
            else:
                # Fallback to hash-based approach
                feature_hash = hashlib.md5(str(sorted(active_features)).encode()).hexdigest()
                for i in range(min(target_dim, 16)):
                    byte_val = int(feature_hash[i*2:(i+1)*2], 16)
                    enriched[i] = (byte_val / 255.0) * 0.7
        
        return enriched
    
    def create_popularity_boost_vector(self, movies_df):
        """
        Create popularity boost vector using sn_votes.
        
        The popularity distribution is exponential (most movies have very low popularity),
        so we need a more aggressive scaling to properly promote culturally iconic films.
        
        Strategy:
        - Use log scaling to compress the exponential range
        - Add exponential boost for high-popularity movies
        - Ensure meaningful separation between popularity tiers
        """
        logger.info("Creating popularity boost vectors...")
        
        # Get normalized votes (sn_votes)
        sn_votes = movies_df["sn_votes"].to_numpy().astype(np.float32)
        
        # Handle None/NaN values - give them low boost (less popular)
        sn_votes = np.nan_to_num(sn_votes, nan=0.0)
        
        # Add small epsilon to avoid log(0)
        sn_votes_safe = sn_votes + 1e-6
        
        # Apply log scaling to compress the exponential range
        log_popularity = np.log10(sn_votes_safe * 1000 + 1)  # Scale up before log for better range
        
        # Add exponential boost for high-popularity movies (> 0.1 sn_votes)
        high_popularity_mask = sn_votes > 0.1
        exponential_boost = np.zeros_like(sn_votes)
        exponential_boost[high_popularity_mask] = np.power(sn_votes[high_popularity_mask], 1.5)
        
        # Combine log scaling with exponential boost
        popularity_boost = log_popularity + exponential_boost
        
        # Normalize to reasonable range (0.01 to 2.0)
        popularity_boost = (popularity_boost - popularity_boost.min()) / (popularity_boost.max() - popularity_boost.min())
        popularity_boost = popularity_boost * 1.99 + 0.01  # Scale to [0.01, 2.0]
        
        logger.info(f"Original sn_votes range: {sn_votes.min():.3f} to {sn_votes.max():.3f}")
        logger.info(f"Popularity boost range: {popularity_boost.min():.3f} to {popularity_boost.max():.3f}")
        logger.info(f"Mean boost: {popularity_boost.mean():.3f}")
        
        # Show some examples of the transformation
        high_pop_indices = np.where(sn_votes > 0.1)[0][:5]
        if len(high_pop_indices) > 0:
            logger.info("High popularity examples:")
            for idx in high_pop_indices:
                logger.info(f"  sn_votes: {sn_votes[idx]:.3f} -> boost: {popularity_boost[idx]:.3f}")
        
        low_pop_indices = np.where(sn_votes < 0.01)[0][:5]
        if len(low_pop_indices) > 0:
            logger.info("Low popularity examples:")
            for idx in low_pop_indices:
                logger.info(f"  sn_votes: {sn_votes[idx]:.3f} -> boost: {popularity_boost[idx]:.3f}")
        
        return popularity_boost.reshape(-1, 1)  # Column vector
    
    def fit_transform(self, movies_df, plot_embeddings, genre_embeddings):
        """
        Fit the model and transform all data into culturally-aware embeddings.
        
        Args:
            movies_df: DataFrame with movie metadata
            plot_embeddings: Pre-computed plot embeddings from embedding.py
            genre_embeddings: Pre-computed enriched genre embeddings from embedding.py
        """
        logger.info("Fitting culturally-aware embedding model...")
        
        # Build localization vocabularies
        self.build_localization_vocabularies(movies_df)
        
        # Store plot and genre embeddings for query processing
        # This enables semantic genre inference for queries
        self._stored_plot_embeddings = plot_embeddings[:, :self.plot_dim].astype(np.float32)
        self._stored_genre_embeddings = genre_embeddings[:, :self.genre_dim].astype(np.float32)
        
        # Normalize stored embeddings for cosine similarity
        faiss.normalize_L2(self._stored_plot_embeddings)
        faiss.normalize_L2(self._stored_genre_embeddings)
        
        logger.info(f"Stored {len(self._stored_plot_embeddings)} plot embeddings for query processing")
        logger.info(f"Stored {len(self._stored_genre_embeddings)} genre embeddings for query processing")
        
        # Create enriched localization embeddings
        localization_embeddings = self._create_localization_enriched_embeddings(
            movies_df, plot_embeddings, self.localization_dim
        )
        
        # Create popularity boost vector
        popularity_boost = self.create_popularity_boost_vector(movies_df)
        
        # Ensure all embeddings are the same length
        n_movies = len(movies_df)
        assert plot_embeddings.shape[0] == n_movies, f"Plot embeddings mismatch: {plot_embeddings.shape[0]} vs {n_movies}"
        assert genre_embeddings.shape[0] == n_movies, f"Genre embeddings mismatch: {genre_embeddings.shape[0]} vs {n_movies}"
        assert localization_embeddings.shape[0] == n_movies, f"Localization embeddings mismatch: {localization_embeddings.shape[0]} vs {n_movies}"
        assert popularity_boost.shape[0] == n_movies, f"Popularity boost mismatch: {popularity_boost.shape[0]} vs {n_movies}"
        
        # Ensure proper dimensions
        plot_embeddings = plot_embeddings[:, :self.plot_dim]  # Truncate if needed
        genre_embeddings = genre_embeddings[:, :self.genre_dim]  # Truncate if needed
        
        # Combine all embeddings with weights
        logger.info("Combining embeddings with cultural awareness...")
        combined_embeddings = np.hstack([
            plot_embeddings * self.plot_weight,
            genre_embeddings * self.genre_weight,
            localization_embeddings * self.localization_weight,
            popularity_boost * self.popularity_weight
        ])
        
        # Normalize final embeddings
        combined_embeddings = combined_embeddings.astype(np.float32)
        faiss.normalize_L2(combined_embeddings)
        
        logger.info(f"Final culturally-aware embedding shape: {combined_embeddings.shape}")
        
        return combined_embeddings
    
    def transform_query(self, query_text, user_countries=None, user_languages=None, prefer_popular=True):
        """
        Transform a query into the same embedding space with cultural preferences.
        
        Args:
            query_text: The search query
            user_countries: List of preferred countries (e.g., ['USA', 'UK'])  
            user_languages: List of preferred languages (e.g., ['English', 'Spanish'])
            prefer_popular: If True, bias towards popular movies (default: True)
        """
        # Get plot embedding
        model = SentenceTransformer("intfloat/multilingual-e5-large")
        plot_embedding = model.encode([f"passage: {query_text}"], convert_to_numpy=True)
        plot_embedding = plot_embedding[:, :self.plot_dim]  # Ensure correct dimension
        
        # Create semantic genre embedding for the query
        genre_embedding = self._create_query_genre_embedding(query_text, plot_embedding)
        
        # Create localization embedding based on user preferences
        localization_embedding = self._create_query_localization_vector(user_countries, user_languages)
        
        # Create popularity preference (if prefer_popular, set high boost)
        # Use the same scaling as the training data for consistency
        if prefer_popular:
            popularity_value = 1.5  # High boost = prefer popular movies (upper range)
        else:
            popularity_value = 0.5  # Low boost = neutral towards popularity (lower range)
        
        popularity_vector = np.array([[popularity_value]], dtype=np.float32)
        
        # Combine with weights
        query_embedding = np.hstack([
            plot_embedding * self.plot_weight,
            genre_embedding * self.genre_weight,
            localization_embedding * self.localization_weight,
            popularity_vector * self.popularity_weight
        ]).astype(np.float32)
        
        faiss.normalize_L2(query_embedding)
        return query_embedding
    
    def _create_query_localization_vector(self, user_countries, user_languages):
        """Create localization vector for query based on user preferences"""
        total_features = len(self.country_vocab) + len(self.language_vocab)
        localization_vec = np.zeros(total_features, dtype=np.float32)
        
        # Set preferred countries
        if user_countries:
            for country in user_countries:
                if country in self.country_vocab:
                    localization_vec[self.country_vocab[country]] = 1.0
        
        # Set preferred languages (offset by country vocab size)
        if user_languages:
            language_offset = len(self.country_vocab)
            for language in user_languages:
                if language in self.language_vocab:
                    localization_vec[language_offset + self.language_vocab[language]] = 1.0
        
        # If no preferences specified, create neutral vector
        if not user_countries and not user_languages:
            # Slightly favor common countries/languages
            common_countries = ['USA', 'United States', 'UK', 'United Kingdom']
            common_languages = ['English']
            
            for country in common_countries:
                if country in self.country_vocab:
                    localization_vec[self.country_vocab[country]] = 0.5
            
            language_offset = len(self.country_vocab)
            for language in common_languages:
                if language in self.language_vocab:
                    localization_vec[language_offset + self.language_vocab[language]] = 0.5
        
        # Reduce to target dimension using the same method as movie vectors
        query_localization = self._create_single_localization_vector(localization_vec, self.localization_dim)
        
        return query_localization.reshape(1, -1)
    
    def _infer_genres_from_query(self, query_text):
        """Infer likely genres from query text"""
        query_lower = query_text.lower()
        inferred_genres = []
        
        # Genre keyword mapping
        genre_keywords = {
            'Crime': ['crime', 'criminal', 'mafia', 'gangster', 'mob', 'heist', 'robbery', 'murder', 'detective', 'police'],
            'Drama': ['family', 'loyalty', 'relationship', 'emotion', 'character', 'drama', 'tragic', 'serious'],
            'Action': ['action', 'fight', 'chase', 'explosion', 'combat', 'battle', 'war', 'hero'],
            'Thriller': ['thriller', 'suspense', 'tension', 'mystery', 'dangerous', 'edge'],
            'Comedy': ['funny', 'comedy', 'humor', 'laugh', 'hilarious', 'comic'],
            'Horror': ['horror', 'scary', 'fear', 'monster', 'ghost', 'terror', 'nightmare'],
            'Romance': ['love', 'romance', 'romantic', 'relationship', 'couple', 'wedding'],
            'Sci-Fi': ['science', 'future', 'space', 'alien', 'robot', 'technology', 'sci-fi'],
            'Fantasy': ['magic', 'fantasy', 'wizard', 'dragon', 'supernatural', 'mythical']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                inferred_genres.append(genre)
        
        logger.info(f"Inferred genres from query '{query_text}': {inferred_genres}")
        return inferred_genres
    
    def _create_query_genre_embedding(self, query_text: str, query_plot_embedding: np.ndarray) -> np.ndarray:
        """
        Create genre embedding for query using semantic similarity to movie genre embeddings
        
        This method finds movies with similar plot embeddings and uses their genre embeddings
        to create a genre representation for the query. This is much more sophisticated than
        keyword matching and leverages the enriched genre embeddings we've built.
        
        Args:
            query_text: The search query
            query_plot_embedding: Pre-computed plot embedding for the query
            
        Returns:
            Genre embedding vector for the query
        """
        # Quick keyword inference as a fallback
        inferred_genres = self._infer_genres_from_query(query_text)
        logger.info(f"Inferred genres from query '{query_text}': {inferred_genres}")
        
        # If we have stored genre embeddings from the training data, use semantic matching
        if hasattr(self, '_stored_genre_embeddings') and self._stored_genre_embeddings is not None:
            return self._create_semantic_genre_embedding(query_plot_embedding, inferred_genres)
        
        # Fallback: create zero embedding (will be improved when we store training embeddings)
        logger.warning("No stored genre embeddings available - using zero vector")
        return np.zeros((1, self.genre_dim), dtype=np.float32)
    
    def _create_semantic_genre_embedding(self, query_plot_embedding: np.ndarray, 
                                       fallback_genres: List[str]) -> np.ndarray:
        """
        Create genre embedding using semantic similarity to existing movie embeddings
        
        This finds the most similar movies by plot and uses their genre embeddings
        to create a weighted genre representation for the query.
        """
        # Flatten the query embedding for comparison
        query_flat = query_plot_embedding.flatten()
        
        # Find top similar movies by plot similarity
        similarities = np.dot(self._stored_plot_embeddings, query_flat) / (
            np.linalg.norm(self._stored_plot_embeddings, axis=1) * np.linalg.norm(query_flat)
        )
        
        # Get top 20 most similar movies
        top_indices = np.argsort(similarities)[-20:]
        top_similarities = similarities[top_indices]
        
        # Weight the similarities (softmax-like)
        weights = np.exp(top_similarities * 3)  # Amplify differences
        weights = weights / weights.sum()
        
        # Create weighted average of genre embeddings
        weighted_genre_embedding = np.zeros(self.genre_dim, dtype=np.float32)
        for idx, weight in zip(top_indices, weights):
            weighted_genre_embedding += weight * self._stored_genre_embeddings[idx]
        
        logger.info(f"Created semantic genre embedding using {len(top_indices)} similar movies")
        logger.info(f"Genre embedding norm: {np.linalg.norm(weighted_genre_embedding):.3f}")
        
        return weighted_genre_embedding.reshape(1, -1)
    
    def save_model(self, path):
        """Save the fitted model components"""
        model_data = {
            'country_vocab': self.country_vocab,
            'language_vocab': self.language_vocab,
            'plot_weight': self.plot_weight,
            'genre_weight': self.genre_weight,
            'localization_weight': self.localization_weight,
            'popularity_weight': self.popularity_weight,
            'plot_dim': self.plot_dim,
            'genre_dim': self.genre_dim,
            'localization_dim': self.localization_dim,
            'final_dim': self.final_dim
        }
        
        # Save metadata
        metadata_path = path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Save stored embeddings for query processing
        embeddings_path = path.replace('.json', '.npz')
        if hasattr(self, '_stored_plot_embeddings') and hasattr(self, '_stored_genre_embeddings'):
            np.savez_compressed(embeddings_path,
                              plot_embeddings=self._stored_plot_embeddings,
                              genre_embeddings=self._stored_genre_embeddings)
            logger.info(f"Stored embeddings saved to {embeddings_path}")
        
        logger.info(f"Culturally-aware model saved to {metadata_path}")

    def load_model(self, path):
        """Load the fitted model components"""
        # Load metadata
        metadata_path = path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'r') as f:
            model_data = json.load(f)
        
        # Restore attributes
        self.country_vocab = model_data['country_vocab']
        self.language_vocab = model_data['language_vocab']
        self.plot_weight = model_data['plot_weight']
        self.genre_weight = model_data['genre_weight']
        self.localization_weight = model_data['localization_weight']
        self.popularity_weight = model_data['popularity_weight']
        self.plot_dim = model_data['plot_dim']
        self.genre_dim = model_data['genre_dim']
        self.localization_dim = model_data['localization_dim']
        self.final_dim = model_data['final_dim']
        
        # Load stored embeddings for query processing
        embeddings_path = path.replace('.json', '.npz')
        try:
            embeddings_data = np.load(embeddings_path)
            self._stored_plot_embeddings = embeddings_data['plot_embeddings']
            self._stored_genre_embeddings = embeddings_data['genre_embeddings']
            logger.info(f"Loaded stored embeddings from {embeddings_path}")
            logger.info(f"Plot embeddings: {self._stored_plot_embeddings.shape}")
            logger.info(f"Genre embeddings: {self._stored_genre_embeddings.shape}")
        except FileNotFoundError:
            logger.warning(f"No stored embeddings found at {embeddings_path}")
            self._stored_plot_embeddings = None
            self._stored_genre_embeddings = None
        
        logger.info(f"Culturally-aware model loaded from {metadata_path}")


if __name__ == "__main__":
    # Load data
    logger.info("Loading movie data and embeddings...")
    movies_df = read_parquet_file(Path("datasets/dist/movies_processed_sn.parquet"), lazy=False)
    embeddings_df = read_parquet_file(Path("datasets/dist/movies_embeddings.parquet"), lazy=False)
    
    # Filter movies with empty plots and align dataframes
    filtered_movies_df = movies_df.filter(pl.col("plot") != "")
    filtered_tids = set(filtered_movies_df["tid"].to_list())
    filtered_embeddings_df = embeddings_df.filter(pl.col("tid").is_in(filtered_tids))
    
    # Ensure proper alignment by sorting both dataframes by tid
    movies_df = filtered_movies_df.sort("tid")
    embeddings_df = filtered_embeddings_df.sort("tid")
    
    # Extract embeddings
    plot_embeddings = np.vstack(embeddings_df["plot_embedding"].to_list()).astype("float32")
    genre_embeddings = np.vstack(embeddings_df["genre_embedding"].to_list()).astype("float32")
    
    logger.info(f"Data loaded: {len(movies_df)} movies")
    logger.info(f"Plot embeddings shape: {plot_embeddings.shape}")
    logger.info(f"Genre embeddings shape: {genre_embeddings.shape}")
    
    # Create culturally-aware embedding system
    ca_embedding = CulturallyAwareMovieEmbedding(
        plot_dim=1024,              # Full plot embedding dimension
        genre_dim=64,               # Enriched genre embedding dimension
        localization_dim=32,        # Country/language embedding dimension
        plot_weight=0.50,           # Primary semantic signal
        genre_weight=0.25,          # Thematic relevance
        localization_weight=0.05,   # Cultural relevance
        popularity_weight=0.20      # Popularity boost (high weight, high impact)
    )
    
    # Fit and transform
    logger.info("Creating culturally-aware embeddings...")
    combined_embeddings = ca_embedding.fit_transform(movies_df, plot_embeddings, genre_embeddings)
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(combined_embeddings.shape[1])
    index.add(combined_embeddings)
    
    # Save everything
    faiss.write_index(index, "datasets/dist/movies_culturally_aware.index")
    ca_embedding.save_model("datasets/dist/culturally_aware_model.json")
    
    logger.info(f"Culturally-aware index built with {index.ntotal} vectors of dimension {combined_embeddings.shape[1]}")
    
    # Test queries
    test_queries = [
        ("mafia crime family loyalty", None, None, True),
        ("space adventure sci-fi", ['USA'], ['English'], True),
        ("romantic comedy", ['France', 'Italy'], ['French', 'Italian'], False),
        ("horror scary movie", None, None, True)
    ]
    
    logger.info("\n" + "="*50)
    logger.info("TESTING CULTURALLY-AWARE SEARCH")
    logger.info("="*50)
    
    for query_text, countries, languages, prefer_popular in test_queries:
        logger.info(f"\nQuery: '{query_text}'")
        logger.info(f"Preferred countries: {countries}")
        logger.info(f"Preferred languages: {languages}")  
        logger.info(f"Prefer popular: {prefer_popular}")
        
        query_embedding = ca_embedding.transform_query(query_text, countries, languages, prefer_popular)
        D, I = index.search(query_embedding, 10)
        
        logger.info(f"Top 10 results:")
        for i, idx in enumerate(I[0]):
            idx_int = int(idx)
            if idx_int < len(movies_df):
                title = movies_df["title"][idx_int]
                rating = movies_df["final_rating"][idx_int]
                votes = movies_df["votes"][idx_int]
                sn_votes = movies_df["sn_votes"][idx_int]
                countries_movie = movies_df["country"][idx_int]
                languages_movie = movies_df["language"][idx_int]
                
                logger.info(f"{i+1}. {title}")
                logger.info(f"    Rating: {rating:.2f}, Votes: {int(votes) if votes else 0} (norm: {sn_votes:.3f})")
                logger.info(f"    Countries: {countries_movie}")
                logger.info(f"    Languages: {languages_movie}")
                logger.info(f"    Similarity: {D[0][i]:.4f}")
        logger.info("-" * 30)
