#!/usr/bin/env python3
"""
Query Engine for ReelRaider's Culturally-Aware Movie Search

This engine provides a high-level interface to search movies using the
culturally-aware embedding system. It loads the pre-built FAISS index
and provides methods for semantic movie search with cultural preferences.

Features:
- Fast similarity search using FAISS
- Cultural preferences (country/language)
- Popularity vs niche movie balancing
- Runtime adjustable embedding weights for cultural tuning
- Genre inference from natural language queries
- Result filtering and ranking
"""

import argparse
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from datasets.utils import logger, read_parquet_file, tid_to_tconst
from datasets.runtime_weighted_search import RuntimeWeightedSearch


@dataclass
class SearchResult:
    """A single movie search result"""
    title: str
    imdb_id: str
    year: int
    rating: float
    votes: int
    popularity_score: float
    countries: List[str]
    languages: List[str]
    genres: List[str]
    plot: str
    similarity: float
    rank: int


@dataclass
class SearchRequest:
    """A movie search request with preferences"""
    query: str
    countries: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    max_results: int = 20
    min_rating: Optional[float] = None
    min_votes: Optional[int] = None
    year_range: Optional[Tuple[int, int]] = None
    # Runtime weight adjustment parameters (optimized for 128-dim localization)
    plot_weight: float = 0.5
    genre_weight: float = 0.25
    localization_weight: float = 0.1  # Increased from 0.05 to leverage 128-dim vectors
    popularity_weight: float = 0.15   # Reduced from 0.2 to make room for localization


class MovieQueryEngine:
    """
    High-level query engine for culturally-aware movie search with runtime weight adjustment
    """

    def __init__(self, model_path: str = "datasets/dist/culturally_aware_model",
                 data_file: str = "datasets/dist/movies_processed_sn.parquet"):
        """
        Initialize the query engine by loading runtime weighted search system
        
        Args:
            model_path: Base path for the culturally-aware embedding model (without extension)
            data_file: Path to the processed movie data file
        """
        self.model_path = Path(model_path)
        self.data_file = Path(data_file)
        self.movies_df = None
        self.runtime_search = None
        
        logger.info("Initializing Movie Query Engine...")
        self._load_data()
        self._load_runtime_weighted_search()
        logger.info("Query engine ready!")
    
    def _load_data(self):
        """Load the processed movie data"""
        movies_path = self.data_file
        if not movies_path.exists():
            raise FileNotFoundError(f"Movie data not found at {movies_path}")
        
        self.movies_df = read_parquet_file(movies_path, lazy=False)
        # Filter out movies with empty plots (same as embedding system)
        self.movies_df = self.movies_df.filter(pl.col("plot") != "").sort("tid")
        
        logger.info(f"Loaded {len(self.movies_df)} movies")
    
    def _load_runtime_weighted_search(self):
        """Load the runtime weighted search system"""
        components_path = self.model_path.with_suffix(".npz")

        if not components_path.exists():
            raise FileNotFoundError(f"Embedding components not found at {components_path}")
        
        self.runtime_search = RuntimeWeightedSearch()
        self.runtime_search.load_components(str(components_path))
        
        logger.info("Loaded runtime weighted search system")
    
    def search(self, request: SearchRequest) -> pl.DataFrame:
        """
        Search for movies based on the request
        
        Args:
            request: SearchRequest with query and preferences
            
        Returns:
            Polars DataFrame with search results and computed columns
        """
        logger.info(f"Searching for: '{request.query}'")
        
        # Use runtime weighted search for dynamic cultural tuning
        raw_results: pl.DataFrame = self.runtime_search.search(
            query_text=request.query,
            user_countries=request.countries,
            user_languages=request.languages,
            plot_weight=request.plot_weight,
            genre_weight=request.genre_weight,
            localization_weight=request.localization_weight,
            popularity_weight=request.popularity_weight,
            top_k=min(request.max_results * 2, 100)  # Get extra results for filtering
        )
        
        # Apply filters using Polars operations
        filtered_results = self._apply_filters(raw_results, request)
        
        # Limit to requested number of results
        final_results = filtered_results.head(request.max_results)
        
        logger.info(f"Found {len(final_results)} results")
        return final_results
    
    def _apply_filters(self, df: pl.DataFrame, request: SearchRequest) -> pl.DataFrame:
        """Apply search filters using Polars operations"""
        result = df
        
        # Rating filter
        if request.min_rating is not None:
            result = result.filter(pl.col("final_rating") >= request.min_rating)
        
        # Votes filter
        if request.min_votes is not None:
            result = result.filter(pl.col("votes") >= request.min_votes)
        
        # Year range filter
        if request.year_range is not None:
            min_year, max_year = request.year_range
            result = result.filter(
                (pl.col("year") >= min_year) & (pl.col("year") <= max_year)
            )
        
        return result
    
    def _parse_list_field(self, field_value) -> List[str]:
        """Parse a list field from the dataframe"""
        if field_value is None:
            return []
        
        if isinstance(field_value, list):
            return [str(item) for item in field_value if item]
        
        if isinstance(field_value, str):
            if field_value.strip() == "":
                return []
            # Handle comma-separated strings
            return [item.strip() for item in field_value.split(',') if item.strip()]
        
        return [str(field_value)]
    
    def search_simple(self, query: str, max_results: int = 10) -> pl.DataFrame:
        """
        Simple search interface for quick queries
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Polars DataFrame with search results
        """
        request = SearchRequest(
            query=query,
            max_results=max_results
        )
        return self.search(request)
    
    def get_recommendations_for_movie(self, movie_tid: int, 
                                    max_results: int = 10,
                                    use_cultural_weights: bool = True,
                                    plot_weight: float = 0.44,
                                    genre_weight: float = 0.13,
                                    localization_weight: float = 0.1,
                                    popularity_weight: float = 0.33) -> pl.DataFrame:
        """
        Get movie recommendations based on a specific movie using culturally-aware search
        
        Args:
            movie_tid: Movie TID (converted from IMDb tconst) to base recommendations on
            max_results: Maximum number of recommendations
            use_cultural_weights: Whether to use cultural preset weights (recommended)
            plot_weight: Weight for plot similarity (used if use_cultural_weights=True)
            genre_weight: Weight for genre similarity
            localization_weight: Weight for cultural localization
            popularity_weight: Weight for popularity
            
        Returns:
            Polars DataFrame with similar movies using cultural awareness
        """
        # Find the movie by tid
        movie_matches = self.movies_df.filter(pl.col("tid") == movie_tid)
        
        if len(movie_matches) == 0:
            logger.warning(f"Movie with TID '{movie_tid}' not found")
            return pl.DataFrame()
        
        # Use the first (and should be only) match
        movie_row = movie_matches.row(0, named=True)
        source_tid = movie_row['tid']  # Get the tid for reliable filtering
        
        # Extract cultural context from the source movie
        source_countries = self._parse_list_field(movie_row['country'])
        source_languages = self._parse_list_field(movie_row['language'])
        
        # Create a query based on the movie's genres and plot
        genres = self._parse_list_field(movie_row['genre'])
        plot_snippet = str(movie_row['plot'])[:200] if movie_row['plot'] else ""
        
        # Combine genres and plot for the query
        genre_text = " ".join(genres).lower()
        query = f"{genre_text} {plot_snippet}"
        
        logger.info(f"Finding movies similar to '{movie_row['title']}' (tid: {source_tid}) with cultural awareness")
        logger.info(f"Source movie context: countries={source_countries}, languages={source_languages}")
        
        if use_cultural_weights:
            # Use culturally-aware search with the movie's cultural context
            request = SearchRequest(
                query=query,
                countries=source_countries if source_countries else None,
                languages=source_languages if source_languages else None,
                max_results=max_results + 1,  # Get extra to filter out the original
                plot_weight=plot_weight,
                genre_weight=genre_weight,
                localization_weight=localization_weight,
                popularity_weight=popularity_weight
            )
            
            results = self.search(request)
            
            # Filter out the original movie from recommendations using tid (more reliable)
            filtered_results = results.filter(
                pl.col("tid") != source_tid
            ).head(max_results)
            
            return filtered_results
        else:
            # Fall back to simple search and filter out original using tid
            all_results = self.search_simple(query, max_results + 1)
            return all_results.filter(
                pl.col("tid") != source_tid
            ).head(max_results)
    
    def get_recommendations_for_movie_by_title(self, movie_title: str, 
                                              max_results: int = 10,
                                              use_cultural_weights: bool = True,
                                              plot_weight: float = 0.44,
                                              genre_weight: float = 0.13,
                                              localization_weight: float = 0.1,
                                              popularity_weight: float = 0.33) -> pl.DataFrame:
        """
        Get movie recommendations based on a movie title (fallback method)
        
        Args:
            movie_title: Title of the movie to base recommendations on
            max_results: Maximum number of recommendations
            use_cultural_weights: Whether to use cultural preset weights (recommended)
            plot_weight: Weight for plot similarity
            genre_weight: Weight for genre similarity
            localization_weight: Weight for cultural localization
            popularity_weight: Weight for popularity
            
        Returns:
            Polars DataFrame with similar movies using cultural awareness
        """
        # Find the movie in our database by title
        movie_matches = self.movies_df.filter(
            pl.col("title").str.to_lowercase().str.contains(movie_title.lower())
        )
        
        if len(movie_matches) == 0:
            logger.warning(f"Movie '{movie_title}' not found")
            return pl.DataFrame()
        
        # Use the first match and get its tid
        movie_row = movie_matches.row(0, named=True)
        movie_tid = movie_row['tid']
        
        logger.info(f"Found movie '{movie_row['title']}' with TID {movie_tid}, using TID-based recommendations")
        
        # Delegate to the TID-based method
        return self.get_recommendations_for_movie(
            movie_tid=movie_tid,
            max_results=max_results,
            use_cultural_weights=use_cultural_weights,
            plot_weight=plot_weight,
            genre_weight=genre_weight,
            localization_weight=localization_weight,
            popularity_weight=popularity_weight
        )
    
    def get_movie_stats(self) -> Dict[str, Any]:
        """Get statistics about the movie database"""
        stats = {
            'total_movies': len(self.movies_df),
            'year_range': (
                int(self.movies_df['year'].min()),
                int(self.movies_df['year'].max())
            ),
            'avg_rating': float(self.movies_df['rating'].mean()),
            'total_countries': len(self.runtime_search.embedding_model.country_vocab),
            'total_languages': len(self.runtime_search.embedding_model.language_vocab),
            'embedding_dimension': (
                self.runtime_search.plot_embeddings.shape[1] + 
                self.runtime_search.genre_embeddings.shape[1] + 
                self.runtime_search.localization_embeddings.shape[1] + 
                self.runtime_search.popularity_vectors.shape[1]
            ),
            'index_size': len(self.movies_df)
        }
        return stats


def main():
    """Command-line interface for the query engine"""
    parser = argparse.ArgumentParser(description='ReelRaider Movie Query Engine')
    parser.add_argument('query', nargs='?', help='Search query for movies')
    parser.add_argument('--model-path', '-m', type=str, default='datasets/dist/culturally_aware_model',
                      help='Base path for the culturally-aware embedding model (without extension)')
    parser.add_argument('--data-file', '-d', type=str, default='datasets/dist/movies_processed_sn.parquet',
                      help='Path to the processed movie data file')
    parser.add_argument('--max-results', '-n', type=int, default=10,
                      help='Maximum number of results (default: 10)')
    parser.add_argument('--countries', nargs='+', 
                      help='Preferred countries (e.g. "USA" "UK")')
    parser.add_argument('--languages', nargs='+',
                      help='Preferred languages (e.g. "English" "Spanish")')
    parser.add_argument('--min-rating', type=float,
                      help='Minimum movie rating (0-10)')
    parser.add_argument('--min-votes', type=int,
                      help='Minimum number of votes')
    parser.add_argument('--year-from', type=int,
                      help='Earliest year to include')
    parser.add_argument('--year-to', type=int,
                      help='Latest year to include')
    parser.add_argument('--stats', action='store_true',
                      help='Show database statistics')
    parser.add_argument('--similar-to', 
                      help='Find movies similar to this title')
    parser.add_argument('--plot-weight', type=float, default=0.5,
                      help='Weight for plot similarity (default: 0.5)')
    parser.add_argument('--genre-weight', type=float, default=0.25,
                      help='Weight for genre similarity (default: 0.25)')
    parser.add_argument('--localization-weight', type=float, default=0.1,
                      help='Weight for localization (country/language) similarity (default: 0.1)')
    parser.add_argument('--popularity-weight', type=float, default=0.15,
                      help='Weight for popularity similarity (default: 0.15)')
    parser.add_argument('--preset', choices=['balanced', 'popular', 'cultural', 'niche', 'ultra-cultural'],
                      help='Use preset weight configuration: '
                           'balanced (default, general purpose with cultural awareness), '
                           'popular (mainstream focus, balanced cultural), '
                           'cultural (prioritizes cultural relevance), '
                           'niche (art house/indie films, semantic focus), '
                           'ultra-cultural (DEPRECATED: extreme localization)')
    
    args = parser.parse_args()
    
    # Apply preset configurations if specified (to be retuned for fixed embedding method)
    if args.preset:
        if args.preset == 'balanced':
            # Default balanced preset - good for general queries with cultural awareness
            args.plot_weight, args.genre_weight = 0.57, 0.15
            args.localization_weight, args.popularity_weight = 0.07, 0.16
        elif args.preset == 'cultural':
            # Cultural preset - emphasizes cultural relevance and plot semantics
            args.plot_weight, args.genre_weight = 0.532, 0.1
            args.localization_weight, args.popularity_weight = 0.16, 0.128
        elif args.preset == 'niche':
            # Niche preset - emphasizes semantic/genre matching, de-emphasizes popularity
            args.plot_weight, args.genre_weight = 0.7, 0.2
            args.localization_weight, args.popularity_weight = 0.05, 0.05
    
    # Initialize query engine
    try:
        engine = MovieQueryEngine(model_path=args.model_path, data_file=args.data_file)
    except FileNotFoundError as e:
        logger.error(f"Failed to initialize query engine: {e}")
        logger.error("Make sure you have run the embedding pipeline first:")
        logger.error("  python -m datasets.culturally_aware_embedding")
        return 1
    
    # Show stats if requested
    if args.stats:
        stats = engine.get_movie_stats()
        print("\n=== ReelRaider Database Statistics ===")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print()
        if not args.query and not args.similar_to:
            return 0  # Exit if only showing stats
    
    # Handle similar movie search
    if args.similar_to:
        print(f"\n=== Movies Similar to '{args.similar_to}' ===")
        if args.preset:
            print(f"Using preset: {args.preset}")
        print(f"Weights: plot={args.plot_weight:.2f}, genre={args.genre_weight:.2f}, "
              f"localization={args.localization_weight:.2f}, popularity={args.popularity_weight:.2f}")
        print()
        
        results = engine.get_recommendations_for_movie_by_title(
            args.similar_to, 
            args.max_results,
            use_cultural_weights=True,
            plot_weight=args.plot_weight,
            genre_weight=args.genre_weight,
            localization_weight=args.localization_weight,
            popularity_weight=args.popularity_weight
        )
        _print_results(results)
        return 0
        return 0
    
    # Check if we have a query
    if not args.query:
        parser.error("A search query is required unless using --similar-to or --stats")
    
    # Build search request
    year_range = None
    if args.year_from or args.year_to:
        year_range = (
            args.year_from or 1900,
            args.year_to or 2030
        )
    
    request = SearchRequest(
        query=args.query,
        countries=args.countries,
        languages=args.languages,
        max_results=args.max_results,
        min_rating=args.min_rating,
        min_votes=args.min_votes,
        year_range=year_range,
        plot_weight=args.plot_weight,
        genre_weight=args.genre_weight,
        localization_weight=args.localization_weight,
        popularity_weight=args.popularity_weight
    )
    
    # Perform search
    results = engine.search(request)
    
    # Display results
    print(f"\n=== Search Results for '{args.query}' ===")
    if args.countries:
        print(f"Preferred countries: {', '.join(args.countries)}")
    if args.languages:
        print(f"Preferred languages: {', '.join(args.languages)}")
    print(f"Weights: plot={args.plot_weight:.2f}, genre={args.genre_weight:.2f}, "
          f"localization={args.localization_weight:.2f}, popularity={args.popularity_weight:.2f}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print()
    
    _print_results(results)
    
    return 0


def _print_results(results: pl.DataFrame):
    """Print search results in a nice format"""
    if len(results) == 0:
        print("No results found.")
        return
    
    for row in results.iter_rows(named=True):
        # Parse list fields for display
        countries = _parse_list_field_for_display(row['country'])
        languages = _parse_list_field_for_display(row['language'])
        genres = _parse_list_field_for_display(row['genre'])
        
        print(f"{row['rank']}. {row['title']} ({row['year']})")
        print(f"   Rating: {row['final_rating']:.1f}/10, Votes: {row['votes']:,}, Popularity: {row['sn_votes']:.3f}")
        print(f"   Genres: {', '.join(genres)}")
        print(f"   Countries: {', '.join(countries)}")
        print(f"   Languages: {', '.join(languages)}")
        print(f"   Similarity: {row['similarity']:.4f}")
        
        # Show plot summary (first 150 chars)
        if row['plot']:
            plot_summary = row['plot'][:150] + "..." if len(row['plot']) > 150 else row['plot']
            print(f"   Plot: {plot_summary}")
        
        print()


def _parse_list_field_for_display(field_value) -> List[str]:
    """Parse a list field for display purposes"""
    if field_value is None:
        return []
    
    if isinstance(field_value, list):
        return [str(item) for item in field_value if item]
    
    if isinstance(field_value, str):
        if field_value.strip() == "":
            return []
        # Handle comma-separated strings
        return [item.strip() for item in field_value.split(',') if item.strip()]
    
    return [str(field_value)]


if __name__ == "__main__":
    exit(main())
