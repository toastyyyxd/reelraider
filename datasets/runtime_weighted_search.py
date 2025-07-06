#!/usr/bin/env python3

"""
Runtime-weighted culturally-aware movie search.

This approach stores the embedding components separately and combines them
at query time with adjustable weights, allowing for dynamic cultural tuning
without rebuilding the index.
"""

import numpy as np
import faiss
import json
from pathlib import Path
import polars as pl
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
from datasets.culturally_aware_embedding import CulturallyAwareMovieEmbedding
from datasets.utils import read_parquet_file, logger

class RuntimeWeightedSearch:
    """
    Movie search system with runtime-adjustable cultural weights.
    
    Stores separate embedding components (plot, genre, localization, popularity)
    and combines them dynamically during search with user-specified weights.
    """
    
    def __init__(self):
        self.plot_embeddings: Optional[np.ndarray] = None
        self.genre_embeddings: Optional[np.ndarray] = None
        self.localization_embeddings: Optional[np.ndarray] = None
        self.popularity_vectors: Optional[np.ndarray] = None
        self.movies_df: Optional[pl.DataFrame] = None
        self.embedding_model: Optional[CulturallyAwareMovieEmbedding] = None
        self.sentence_model: Optional[SentenceTransformer] = None
        
    def load_components(self, components_path: str):
        """Load pre-computed embedding components"""
        logger.info(f"Loading embedding components from {components_path}")
        
        # Load component embeddings
        data = np.load(components_path)
        self.plot_embeddings = data['plot_embeddings']
        self.genre_embeddings = data['genre_embeddings'] 
        self.localization_embeddings = data['localization_embeddings']
        self.popularity_vectors = data['popularity_vectors']
        
        logger.info(f"Loaded components:")
        logger.info(f"  Plot: {self.plot_embeddings.shape}")
        logger.info(f"  Genre: {self.genre_embeddings.shape}")
        logger.info(f"  Localization: {self.localization_embeddings.shape}")
        logger.info(f"  Popularity: {self.popularity_vectors.shape}")
        
        # Load movie data
        self.movies_df = read_parquet_file(Path("datasets/dist/movies_processed_sn.parquet"), lazy=False)
        self.movies_df = self.movies_df.filter(pl.col("plot") != "").sort("tid")
        
        # Load embedding model for query processing
        # Get dimensions from the loaded components
        localization_dim = self.localization_embeddings.shape[1]
        genre_dim = self.genre_embeddings.shape[1] 
        self.embedding_model = CulturallyAwareMovieEmbedding(
            plot_dim=1024, 
            genre_dim=genre_dim, 
            localization_dim=localization_dim
        )
        self.embedding_model.load_model("datasets/dist/culturally_aware_model.json")
        
        # Set the stored embeddings needed for semantic genre embedding creation
        self.embedding_model._stored_genre_embeddings = self.genre_embeddings
        self.embedding_model._stored_plot_embeddings = self.plot_embeddings
        
        # Load sentence transformer for query embeddings
        self.sentence_model = SentenceTransformer("intfloat/multilingual-e5-large")
        
        logger.info(f"Runtime weighted search system ready with {len(self.movies_df)} movies")
    
    def create_combined_embeddings(self, plot_weight: float = 0.5, genre_weight: float = 0.25,
                                 localization_weight: float = 0.05, popularity_weight: float = 0.2) -> np.ndarray:
        """
        Combine embedding components with specified weights at runtime
        
        Args:
            plot_weight: Weight for plot similarity (semantic relevance)
            genre_weight: Weight for genre similarity (thematic relevance)
            localization_weight: Weight for country/language preference (cultural relevance)
            popularity_weight: Weight for popularity boost (cultural iconicity)
        """
        logger.info(f"Combining embeddings with weights: plot={plot_weight:.3f}, genre={genre_weight:.3f}, "
                   f"loc={localization_weight:.3f}, pop={popularity_weight:.3f}")
        
        # Apply weights to each component
        weighted_plot = self.plot_embeddings * plot_weight
        weighted_genre = self.genre_embeddings * genre_weight  
        weighted_localization = self.localization_embeddings * localization_weight
        weighted_popularity = self.popularity_vectors * popularity_weight
        
        # Combine all components
        combined = np.hstack([
            weighted_plot,
            weighted_genre, 
            weighted_localization,
            weighted_popularity
        ]).astype(np.float32)
        
        # Normalize final embeddings
        faiss.normalize_L2(combined)
        
        logger.info(f"Combined embedding shape: {combined.shape}")
        return combined
    
    def create_query_embedding(self, query_text: str, user_countries: Optional[List[str]] = None,
                             user_languages: Optional[List[str]] = None,
                             plot_weight: float = 0.5, genre_weight: float = 0.25,
                             localization_weight: float = 0.05, popularity_weight: float = 0.2) -> np.ndarray:
        """Create query embedding with the same weights as the movie embeddings"""
        
        # Get plot embedding
        plot_embedding = self.sentence_model.encode([f"passage: {query_text}"], convert_to_numpy=True)
        plot_embedding = plot_embedding[:, :1024]  # Ensure correct dimension
        
        # Create semantic genre embedding for the query
        genre_embedding = self.embedding_model._create_query_genre_embedding(query_text, plot_embedding)
        genre_embedding = genre_embedding[:, :self.embedding_model.genre_dim]  # Ensure correct dimension
        
        # Create localization embedding based on user preferences
        localization_embedding = self.embedding_model._create_query_localization_vector(user_countries, user_languages)
        # Use the correct localization dimension from the embedding model
        localization_embedding = localization_embedding[:, :self.embedding_model.localization_dim]
        
        # Fixed popularity preference to bias toward popular movies (avoids embedding space distortion)
        popularity_vector = np.array([[1.5]], dtype=np.float32)
        
        # Combine with the same weights as the movie embeddings
        query_embedding = np.hstack([
            plot_embedding * plot_weight,
            genre_embedding * genre_weight,
            localization_embedding * localization_weight,
            popularity_vector * popularity_weight
        ]).astype(np.float32)
        
        faiss.normalize_L2(query_embedding)
        return query_embedding
    
    def search(self, query_text: str, user_countries: Optional[List[str]] = None,
              user_languages: Optional[List[str]] = None,
              plot_weight: float = 0.5, genre_weight: float = 0.25,
              localization_weight: float = 0.05, popularity_weight: float = 0.2,
              top_k: int = 10) -> pl.DataFrame:
        """
        Search for movies with runtime-adjustable cultural weights
        
        Args:
            query_text: Search query
            user_countries: Preferred countries for cultural relevance
            user_languages: Preferred languages for cultural relevance  
            plot_weight: Weight for semantic plot similarity
            genre_weight: Weight for genre/thematic similarity
            localization_weight: Weight for cultural/regional relevance
            popularity_weight: Weight for popularity/iconicity boost (set to 0 to ignore popularity)
            top_k: Number of results to return
        """
        
        # Create movie embeddings with specified weights
        movie_embeddings = self.create_combined_embeddings(
            plot_weight, genre_weight, localization_weight, popularity_weight
        )
        
        # Build FAISS index
        index = faiss.IndexFlatIP(movie_embeddings.shape[1])
        index.add(movie_embeddings)
        
        # Create query embedding with same weights
        query_embedding = self.create_query_embedding(
            query_text, user_countries, user_languages,
            plot_weight, genre_weight, localization_weight, popularity_weight
        )
        
        # Search
        D, I = index.search(query_embedding, top_k)
        
        # Format results using Polars operations for efficiency
        result_indices = I[0]
        similarities = D[0]
        
        # Filter valid indices
        valid_mask = result_indices < len(self.movies_df)
        valid_indices = result_indices[valid_mask]
        valid_similarities = similarities[valid_mask]
        
        if len(valid_indices) == 0:
            return pl.DataFrame()
        
        # Get the result movies as a DataFrame slice
        result_movies = self.movies_df[valid_indices]
        
        # Add computed columns
        result_movies = result_movies.with_columns([
            pl.Series("rank", range(1, len(valid_indices) + 1)),
            pl.Series("similarity", valid_similarities.astype(np.float32))
        ])
        
        # Add cultural match column
        if user_countries:
            # Use list operations since country is a List(Categorical) column
            cultural_match_expr = pl.col("country").list.eval(
                pl.element().is_in(user_countries)
            ).list.any().fill_null(False)
        else:
            cultural_match_expr = pl.lit(False)
        result_movies = result_movies.with_columns([
            cultural_match_expr.alias("cultural_match")
        ])
        
        return result_movies

def test_runtime_weighting():
    """Test the runtime weighting system with Hong Kong queries"""
    
    # First create components if they don't exist
    components_path = "datasets/dist/culturally_aware_model.npz"
    if not Path(components_path).exists():
        logger.error(f"Embedding components not found at {components_path}")
        logger.error("Please run: python -m datasets.culturally_aware_embedding")
        return
    
    # Initialize search system
    search_system = RuntimeWeightedSearch()
    search_system.load_components(components_path)
    
    # Test query
    query = "Hong Kong martial arts action crime"
    countries = ["Hong Kong"]
    languages = ["Cantonese", "Chinese"]
    
    # Test different weight configurations
    weight_configs = [
        {
            "name": "Default (Popular Western bias)",
            "plot_weight": 0.5, "genre_weight": 0.25, 
            "localization_weight": 0.05, "popularity_weight": 0.2,
            "prefer_popular": True
        },
        {
            "name": "Balanced Cultural",
            "plot_weight": 0.4, "genre_weight": 0.2,
            "localization_weight": 0.2, "popularity_weight": 0.2,
            "prefer_popular": True
        },
        {
            "name": "High Cultural Focus",
            "plot_weight": 0.3, "genre_weight": 0.2,
            "localization_weight": 0.3, "popularity_weight": 0.2,
            "prefer_popular": True
        },
        {
            "name": "Cultural + Low Popularity",
            "plot_weight": 0.4, "genre_weight": 0.25,
            "localization_weight": 0.25, "popularity_weight": 0.1,
            "prefer_popular": False
        }
    ]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING RUNTIME WEIGHT ADJUSTMENT")
    logger.info(f"Query: '{query}'")
    logger.info(f"Cultural preferences: {countries}, {languages}")
    logger.info(f"{'='*80}")
    
    for config in weight_configs:
        logger.info(f"\n{'-'*60}")
        logger.info(f"CONFIGURATION: {config['name']}")
        logger.info(f"Weights: plot={config['plot_weight']:.2f}, genre={config['genre_weight']:.2f}, "
                   f"loc={config['localization_weight']:.2f}, pop={config['popularity_weight']:.2f}")
        logger.info(f"Prefer popular: {config['prefer_popular']}")
        logger.info(f"{'-'*60}")
        
        results = search_system.search(
            query, countries, languages,
            plot_weight=config['plot_weight'],
            genre_weight=config['genre_weight'], 
            localization_weight=config['localization_weight'],
            popularity_weight=config['popularity_weight'],
            top_k=10
        )
        
        hk_count = 0
        for row in results.iter_rows(named=True):
            is_hk = row['cultural_match']
            if is_hk:
                hk_count += 1
                mark = "🇭🇰"
            else:
                mark = "  "
            
            logger.info(f"{row['rank']:2d}. {mark} {row['title']} ({row['year']})")
            logger.info(f"     Countries: {row['country']}")
            logger.info(f"     Pop: {row['sn_votes']:.3f}, Sim: {row['similarity']:.4f}")
        
        logger.info(f"\nHong Kong movies in top 10: {hk_count}/10")

if __name__ == "__main__":
    test_runtime_weighting()
