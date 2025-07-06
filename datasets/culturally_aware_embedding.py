"""
Culturally-Aware Movie Embedding Component Creator

This module creates separate embedding components for runtime weighted search, enabling
dynamic cultural tuning without rebuilding indices. It addresses the core problem that
semantic similarity alone finds niche films instead of culturally iconic ones.

The system generates individual components that can be combined at runtime:
- Plot embeddings (semantic meaning)
- Enriched genre embeddings (thematic categories)
- Localization embeddings (cultural/regional relevance)
- Popularity boost (promotes culturally iconic films)

Key Output:
The main output is create_separate_components() which generates an .npz file containing
all embedding components, used by RuntimeWeightedSearch for dynamic weight adjustment.

Configuration:
All magic constants are centralized in CulturalEmbeddingConfig to enable easy tuning
and experimentation. The global CONFIG instance can be updated at runtime.
"""

import polars as pl
import numpy as np
from datasets.utils import logger, read_parquet_file, write_parquet_file
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import hashlib
import json
from typing import List
from dataclasses import dataclass

@dataclass
class CulturalEmbeddingConfig:
    """Configuration class for culturally-aware embedding magic constants"""
    
    # Hash function constants for feature mapping
    hash_multiplier_primary: int = 17
    hash_multiplier_secondary: int = 31
    hash_multiplier_tertiary: int = 53
    hash_offset_secondary: int = 7
    hash_offset_tertiary: int = 13
    
    # Signal strength constants
    signal_strength_primary: float = 0.8
    signal_strength_secondary: float = 0.6
    signal_strength_tertiary: float = 0.3
    signal_strength_hash_fallback: float = 0.7
    max_tertiary_features: int = 3  # Maximum features to apply tertiary mapping
    
    # Clustering constants
    min_subclusters: int = 2
    subcluster_ratio: int = 3  # movies per subcluster
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10
    
    # Popularity boost constants
    popularity_epsilon: float = 1e-6
    popularity_log_scale: float = 1000.0
    popularity_log_offset: float = 1.0
    popularity_high_threshold: float = 0.1
    popularity_exponential_power: float = 1.5
    popularity_boost_min: float = 0.01
    popularity_boost_max: float = 2.0
    popularity_boost_range: float = 1.99  # max - min
    popularity_zero_threshold: float = 1e-10
    
    # Query processing constants
    query_popularity_high: float = 1.5
    query_popularity_low: float = 0.5
    query_localization_fallback: float = 0.5
    
    # Semantic similarity constants
    semantic_top_k: int = 20
    semantic_weight_amplifier: float = 3.0
    
    # Localization enrichment constants
    cluster_strength_base: float = 0.2
    cluster_signal_strength: float = 0.3
    remaining_dims_reserved: int = 16
    max_cluster_hash_dims: int = 15
    
    # Hash fallback constants
    hash_fallback_max_dims: int = 16
    hash_byte_normalizer: float = 255.0
    
    # Dimension thresholds
    large_dim_threshold: int = 64

# Global configuration instance
CONFIG = CulturalEmbeddingConfig()

class CulturallyAwareMovieEmbedding:
    def __init__(self, plot_dim=1024, genre_dim=64, localization_dim=32):
        """
        Culturally-aware movie embedding component creator for runtime weighted search.
        
        This system creates separate embedding components that can be combined with different
        weights at runtime, addressing the core problem that semantic similarity alone finds
        niche films instead of culturally iconic ones.
        
        Components created:
        1. Plot embeddings (semantic meaning)
        2. Enriched genre embeddings (thematic categories)  
        3. Localization embeddings (cultural/regional relevance)
        4. Popularity boost (promotes culturally iconic films)
        
        The main output is through create_separate_components() which generates an .npz file
        used by RuntimeWeightedSearch for dynamic weight adjustment.
        
        Args:
            plot_dim: Dimension of plot embeddings (semantic content)
            genre_dim: Target dimension for enriched genre embeddings
            localization_dim: Target dimension for country/language embeddings
        """
        self.plot_dim = plot_dim
        self.genre_dim = genre_dim
        self.localization_dim = localization_dim
        
        # Vocabularies for cultural features
        self.country_vocab = {}
        self.language_vocab = {}
        
        logger.info(f"Culturally-aware embedding dimensions: plot={plot_dim}, genre={genre_dim}, localization={localization_dim}, popularity=1")
        logger.info("Weights will be handled dynamically at runtime by RuntimeWeightedSearch")
    
    def update_config(self, new_config: CulturalEmbeddingConfig):
        """Update the configuration instance used by this embedding model"""
        global CONFIG
        CONFIG = new_config
        logger.info("Updated culturally-aware embedding configuration")
    
    def get_config(self) -> CulturalEmbeddingConfig:
        """Get the current configuration instance"""
        return CONFIG
    
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
            max_subclusters = min(max(CONFIG.min_subclusters, len(plot_vecs) // CONFIG.subcluster_ratio), max(1, remaining_dims))
            n_subclusters = min(len(plot_vecs), max_subclusters)
            
            if n_subclusters > 1:
                plot_matrix = np.array(plot_vecs, dtype=np.float32)
                kmeans = KMeans(n_clusters=n_subclusters, random_state=CONFIG.kmeans_random_state, n_init=CONFIG.kmeans_n_init)
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
    
    def _create_enriched_localization_vector(self, loc_vec, loc_combo, subcluster_id, n_subclusters, target_dim):
        """
        Create enriched localization vector with better dimensionality handling
        
        Args:
            loc_vec: Base localization vector with active features
            loc_combo: Tuple of active feature indices for this localization group
            subcluster_id: ID of the subcluster this movie belongs to (0 to n_subclusters-1)
            n_subclusters: Total number of subclusters in this localization group
            target_dim: Target dimension for the output vector
            
        Returns:
            Enriched localization vector incorporating both base features and cluster information
        """
        enriched = np.zeros(target_dim, dtype=np.float32)
        
        # Get active features (countries and languages that are set)
        active_features = np.where(loc_vec > 0)[0]
        
        if len(active_features) > 0:
            # With larger target_dim (e.g., 128), we can be more direct
            if target_dim >= CONFIG.large_dim_threshold:
                # Use a smarter mapping that preserves more locality
                # Map each active feature to multiple dimensions to reduce collisions
                target_half = max(1, target_dim // 2)  # Ensure at least 1
                target_quarter = max(1, target_dim // 4)  # Ensure at least 1
                
                for i, feature_idx in enumerate(active_features):
                    # Map each feature to 2-3 dimensions using different hash functions
                    base_pos = (feature_idx * CONFIG.hash_multiplier_primary) % target_half  # Use first half
                    enriched[base_pos] = CONFIG.signal_strength_primary  # Strong signal
                    
                    # Add secondary mapping to reduce collisions
                    secondary_pos = (feature_idx * CONFIG.hash_multiplier_secondary + CONFIG.hash_offset_secondary) % target_half + target_half
                    if secondary_pos < target_dim:  # Bounds check
                        enriched[secondary_pos] = CONFIG.signal_strength_secondary  # Secondary signal
                    
                    # Add a third weak signal for robustness
                    if i < CONFIG.max_tertiary_features:  # Only for first few features to avoid overcrowding
                        tertiary_pos = (feature_idx * CONFIG.hash_multiplier_tertiary + CONFIG.hash_offset_tertiary) % target_quarter
                        if tertiary_pos < target_dim:  # Bounds check
                            enriched[tertiary_pos] = max(enriched[tertiary_pos], CONFIG.signal_strength_tertiary)
            else:
                # Fallback to hash-based approach for smaller dimensions
                feature_hash = hashlib.md5(str(sorted(active_features)).encode()).hexdigest()
                for i in range(min(target_dim, CONFIG.hash_fallback_max_dims)):
                    byte_val = int(feature_hash[i*2:(i+1)*2], 16)
                    enriched[i] = (byte_val / CONFIG.hash_byte_normalizer) * CONFIG.signal_strength_hash_fallback
        
        # Add sub-cluster information in remaining dimensions if we have space
        remaining_start = target_dim - CONFIG.remaining_dims_reserved if target_dim > CONFIG.remaining_dims_reserved else 0
        if remaining_start > 0 and remaining_start < target_dim:
            combo_hash = hashlib.md5(str(loc_combo).encode()).hexdigest()
            cluster_hash = hashlib.md5(f"{combo_hash}_{subcluster_id}".encode()).hexdigest()
            
            # Use n_subclusters to normalize the cluster influence
            cluster_strength = CONFIG.cluster_strength_base / max(1, n_subclusters)  # Weaker signal for more clusters
            
            # Add normalized cluster ID as a direct feature
            if target_dim > remaining_start:
                normalized_cluster_id = subcluster_id / max(1, n_subclusters - 1) if n_subclusters > 1 else 0
                enriched[remaining_start] = normalized_cluster_id * CONFIG.cluster_signal_strength  # Direct cluster signal
            
            for i in range(min(CONFIG.max_cluster_hash_dims, target_dim - remaining_start - 1)):  # -1 to account for direct cluster feature
                byte_val = int(cluster_hash[i*2:(i+1)*2], 16)
                enriched[remaining_start + 1 + i] = (byte_val / CONFIG.hash_byte_normalizer) * cluster_strength
        
        return enriched
    
    def _create_single_localization_vector(self, loc_vec, target_dim):
        """Create enriched vector for single movie with better dimensionality handling"""
        enriched = np.zeros(target_dim, dtype=np.float32)
        
        # Find active features
        active_features = np.where(loc_vec > 0)[0]
        
        if len(active_features) > 0:
            if target_dim >= CONFIG.large_dim_threshold:
                # Use the same smarter mapping as above
                target_half = max(1, target_dim // 2)  # Ensure at least 1
                target_quarter = max(1, target_dim // 4)  # Ensure at least 1
                
                for i, feature_idx in enumerate(active_features):
                    base_pos = (feature_idx * CONFIG.hash_multiplier_primary) % target_half
                    enriched[base_pos] = CONFIG.signal_strength_primary
                    
                    secondary_pos = (feature_idx * CONFIG.hash_multiplier_secondary + CONFIG.hash_offset_secondary) % target_half + target_half
                    if secondary_pos < target_dim:  # Bounds check
                        enriched[secondary_pos] = CONFIG.signal_strength_secondary
                    
                    if i < CONFIG.max_tertiary_features:
                        tertiary_pos = (feature_idx * CONFIG.hash_multiplier_tertiary + CONFIG.hash_offset_tertiary) % target_quarter
                        if tertiary_pos < target_dim:  # Bounds check
                            enriched[tertiary_pos] = max(enriched[tertiary_pos], CONFIG.signal_strength_tertiary)
            else:
                # Fallback to hash-based approach
                feature_hash = hashlib.md5(str(sorted(active_features)).encode()).hexdigest()
                for i in range(min(target_dim, CONFIG.hash_fallback_max_dims)):
                    byte_val = int(feature_hash[i*2:(i+1)*2], 16)
                    enriched[i] = (byte_val / CONFIG.hash_byte_normalizer) * CONFIG.signal_strength_hash_fallback
        
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
        if "sn_votes" not in movies_df.columns:
            logger.warning("sn_votes column not found - using uniform popularity boost")
            return np.ones((len(movies_df), 1), dtype=np.float32)
            
        sn_votes = movies_df["sn_votes"].to_numpy().astype(np.float32)
        
        # Handle None/NaN values - give them low boost (less popular)
        sn_votes = np.nan_to_num(sn_votes, nan=0.0)
        
        # Add small epsilon to avoid log(0)
        sn_votes_safe = sn_votes + CONFIG.popularity_epsilon
        
        # Apply log scaling to compress the exponential range
        log_popularity = np.log10(sn_votes_safe * CONFIG.popularity_log_scale + CONFIG.popularity_log_offset)  # Scale up before log for better range
        
        # Add exponential boost for high-popularity movies (> 0.1 sn_votes)
        high_popularity_mask = sn_votes > CONFIG.popularity_high_threshold
        exponential_boost = np.zeros_like(sn_votes)
        exponential_boost[high_popularity_mask] = np.power(sn_votes[high_popularity_mask], CONFIG.popularity_exponential_power)
        
        # Combine log scaling with exponential boost
        popularity_boost = log_popularity + exponential_boost
        
        # Normalize to reasonable range (0.01 to 2.0)
        boost_min = popularity_boost.min()
        boost_max = popularity_boost.max()
        boost_range = boost_max - boost_min
        
        if boost_range > CONFIG.popularity_zero_threshold:  # Avoid division by zero
            popularity_boost = (popularity_boost - boost_min) / boost_range
            popularity_boost = popularity_boost * CONFIG.popularity_boost_range + CONFIG.popularity_boost_min  # Scale to [0.01, 2.0]
        else:
            # All values are the same - use neutral boost
            logger.warning("All popularity values are identical - using neutral boost of 1.0")
            popularity_boost = np.full_like(popularity_boost, 1.0)
        
        logger.info(f"Original sn_votes range: {sn_votes.min():.3f} to {sn_votes.max():.3f}")
        logger.info(f"Popularity boost range: {popularity_boost.min():.3f} to {popularity_boost.max():.3f}")
        logger.info(f"Mean boost: {popularity_boost.mean():.3f}")
        
        # Show some examples of the transformation
        high_pop_indices = np.where(sn_votes > CONFIG.popularity_high_threshold)[0][:5]
        if len(high_pop_indices) > 0:
            logger.info("High popularity examples:")
            for idx in high_pop_indices:
                logger.info(f"  sn_votes: {sn_votes[idx]:.3f} -> boost: {popularity_boost[idx]:.3f}")
        
        low_pop_indices = np.where(sn_votes < CONFIG.popularity_boost_min)[0][:5]
        if len(low_pop_indices) > 0:
            logger.info("Low popularity examples:")
            for idx in low_pop_indices:
                logger.info(f"  sn_votes: {sn_votes[idx]:.3f} -> boost: {popularity_boost[idx]:.3f}")
        
        return popularity_boost.reshape(-1, 1)  # Column vector
    

    

    
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
                    localization_vec[self.country_vocab[country]] = CONFIG.query_localization_fallback
            
            language_offset = len(self.country_vocab)
            for language in common_languages:
                if language in self.language_vocab:
                    localization_vec[language_offset + self.language_vocab[language]] = CONFIG.query_localization_fallback
        
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
        
        # Ensure we have valid embeddings
        if self._stored_plot_embeddings is None or len(self._stored_plot_embeddings) == 0:
            logger.warning("No stored plot embeddings available for semantic genre inference")
            return np.zeros((1, self.genre_dim), dtype=np.float32)
        
        # Find top similar movies by plot similarity
        try:
            similarities = np.dot(self._stored_plot_embeddings, query_flat) / (
                np.linalg.norm(self._stored_plot_embeddings, axis=1) * np.linalg.norm(query_flat)
            )
            
            # Handle potential NaN values
            similarities = np.nan_to_num(similarities, nan=0.0)
        except Exception as e:
            logger.error(f"Error computing similarities: {e}")
            return np.zeros((1, self.genre_dim), dtype=np.float32)
        
        # Get top 20 most similar movies
        top_indices = np.argsort(similarities)[-CONFIG.semantic_top_k:]
        top_similarities = similarities[top_indices]
        
        # Weight the similarities (softmax-like)
        weights = np.exp(top_similarities * CONFIG.semantic_weight_amplifier)  # Amplify differences
        weights = weights / weights.sum()
        
        # Create weighted average of genre embeddings
        weighted_genre_embedding = np.zeros(self.genre_dim, dtype=np.float32)
        for idx, weight in zip(top_indices, weights):
            weighted_genre_embedding += weight * self._stored_genre_embeddings[idx]
        
        logger.info(f"Created semantic genre embedding using {len(top_indices)} similar movies")
        logger.info(f"Genre embedding norm: {np.linalg.norm(weighted_genre_embedding):.3f}")
        
        return weighted_genre_embedding.reshape(1, -1)
    
    def save_model(self, path):
        """Save the fitted model metadata (vocabularies and dimensions)"""
        model_data = {
            'country_vocab': self.country_vocab,
            'language_vocab': self.language_vocab,
            'plot_dim': self.plot_dim,
            'genre_dim': self.genre_dim,
            'localization_dim': self.localization_dim
        }
        
        # Save metadata to the specified path (should be .json)
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Culturally-aware model metadata saved to {path}")

    def load_model(self, path):
        """Load the fitted model metadata (vocabularies and dimensions)"""
        # Load metadata from the specified path (should be .json)
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        # Restore attributes
        self.country_vocab = model_data['country_vocab']
        self.language_vocab = model_data['language_vocab']
        self.plot_dim = model_data['plot_dim']
        self.genre_dim = model_data['genre_dim']
        self.localization_dim = model_data['localization_dim']
        
        # Backwards compatibility: ignore old weight fields if present
        if any(key in model_data for key in ['plot_weight', 'genre_weight', 'localization_weight', 'popularity_weight']):
            logger.info("Loaded model file with legacy weight fields (ignored - weights now handled at runtime)")
        
        logger.info(f"Culturally-aware model metadata loaded from {path}")
        logger.info("Ready for component creation and query processing with runtime weighted search.")
    
    def create_separate_components(self, movies_df, plot_embeddings, genre_embeddings, output_path: str = "datasets/dist/embedding_components.npz"):
        """
        Create and save separate embedding components for runtime weighted search
        
        This is the main method for creating the .npz file with individual embedding components
        (plot, genre, localization, popularity) that can be combined with different weights at runtime
        by the RuntimeWeightedSearch system.
        
        Args:
            movies_df: DataFrame with movie metadata
            plot_embeddings: Pre-computed plot embeddings from embedding.py
            genre_embeddings: Pre-computed enriched genre embeddings from embedding.py
            output_path: Path to save the components file (.npz)
            
        Returns:
            Path: The output path where components were saved
        """
        logger.info("Creating separate embedding components for runtime weighted search...")
        
        # Validate inputs
        if len(movies_df) == 0:
            raise ValueError("Empty movies dataframe")
        if plot_embeddings.shape[0] == 0:
            raise ValueError("Empty plot embeddings")
        if genre_embeddings.shape[0] == 0:
            raise ValueError("Empty genre embeddings")
            
        # Build localization vocabularies
        self.build_localization_vocabularies(movies_df)
        
        # Ensure proper dimensions by truncating if needed
        plot_embeddings_comp = plot_embeddings[:, :self.plot_dim].astype(np.float32)
        genre_embeddings_comp = genre_embeddings[:, :self.genre_dim].astype(np.float32)
        
        # Create localization embeddings
        localization_embeddings = self._create_localization_enriched_embeddings(
            movies_df, plot_embeddings, self.localization_dim
        )
        
        # Create popularity vectors
        popularity_vectors = self.create_popularity_boost_vector(movies_df)
        
        # Normalize each component independently for runtime combination
        # L2 normalize using numpy (equivalent to faiss.normalize_L2)
        norms = np.linalg.norm(plot_embeddings_comp, axis=1, keepdims=True)
        plot_embeddings_comp = plot_embeddings_comp / np.maximum(norms, 1e-12)  # Avoid division by zero
        
        norms = np.linalg.norm(genre_embeddings_comp, axis=1, keepdims=True)
        genre_embeddings_comp = genre_embeddings_comp / np.maximum(norms, 1e-12)
        
        norms = np.linalg.norm(localization_embeddings, axis=1, keepdims=True)
        localization_embeddings = localization_embeddings / np.maximum(norms, 1e-12)
        
        # Note: popularity vectors are intentionally not normalized to preserve magnitude differences
        
        # Ensure all components have the same number of movies
        n_movies = len(movies_df)
        assert plot_embeddings_comp.shape[0] == n_movies, f"Plot embeddings mismatch: {plot_embeddings_comp.shape[0]} vs {n_movies}"
        assert genre_embeddings_comp.shape[0] == n_movies, f"Genre embeddings mismatch: {genre_embeddings_comp.shape[0]} vs {n_movies}"
        assert localization_embeddings.shape[0] == n_movies, f"Localization embeddings mismatch: {localization_embeddings.shape[0]} vs {n_movies}"
        assert popularity_vectors.shape[0] == n_movies, f"Popularity vectors mismatch: {popularity_vectors.shape[0]} vs {n_movies}"
        
        # Save components
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_path,
                           plot_embeddings=plot_embeddings_comp,
                           genre_embeddings=genre_embeddings_comp,
                           localization_embeddings=localization_embeddings,
                           popularity_vectors=popularity_vectors)
        
        logger.info(f"Saved embedding components to {output_path}")
        logger.info(f"Component shapes:")
        logger.info(f"  Plot: {plot_embeddings_comp.shape}")
        logger.info(f"  Genre: {genre_embeddings_comp.shape}")
        logger.info(f"  Localization: {localization_embeddings.shape}")
        logger.info(f"  Popularity: {popularity_vectors.shape}")
        
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate culturally-aware movie embedding components')
    parser.add_argument('--output', '-o', type=str, default='datasets/dist/culturally_aware_model',
                      help='Base output path (without extension). Will create .json and .npz files (default: datasets/dist/culturally_aware_model)')
    parser.add_argument('--genre-dim', '-g', type=int, default=128,
                      help='Dimension for enriched genre embeddings (default: 128)')
    parser.add_argument('--localization-dim', type=int, default=128,
                      help='Dimension for localization embeddings (default: 128)')
    args = parser.parse_args()
    
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
        plot_dim=1024,                  # Full plot embedding dimension
        genre_dim=args.genre_dim,       # Enriched genre embedding dimension
        localization_dim=args.localization_dim  # Country/language embedding dimension (configurable)
    )
    
    # Generate output paths from base path
    base_path = args.output
    metadata_path = f"{base_path}.json"
    components_path = f"{base_path}.npz"
    
    # Ensure output directory exists
    output_dir = Path(base_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create separate components for runtime weighted search
    # This will build the vocabularies that we need for the metadata
    logger.info("Creating separate components for runtime weighted search...")
    ca_embedding.create_separate_components(
        movies_df, plot_embeddings, genre_embeddings,
        components_path
    )
    
    # Save model metadata (after vocabularies are built)
    logger.info(f"Saving model metadata to: {metadata_path}")
    ca_embedding.save_model(metadata_path)
    
    logger.info(f"Created files:")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Components: {components_path}")
    logger.info(f"Culturally-aware embedding components ready for runtime weighted search with {len(movies_df)} movies")
