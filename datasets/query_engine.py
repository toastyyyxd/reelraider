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
import faiss
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from datasets.utils import logger, read_parquet_file
from datasets.culturally_aware_embedding import CulturallyAwareMovieEmbedding
from datasets.runtime_weighted_search import RuntimeWeightedSearch


@dataclass
class SearchResult:
    """A single movie search result"""
    title: str
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
    prefer_popular: bool = True
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

    def __init__(self, model_path: str = "datasets/dist/culturally_aware_model.json",
                 data_file: str = "datasets/dist/movies_processed_sn.parquet",
                 use_runtime_weights: bool = True):
        """
        Initialize the query engine by loading pre-built indices and data
        
        Args:
            model_path: Path to the culturally-aware embedding model
            data_file: Path to the processed movie data file
            use_runtime_weights: Whether to use runtime weight adjustment (recommended)
        """
        self.model_path = Path(model_path)
        self.data_file = Path(data_file)
        self.use_runtime_weights = use_runtime_weights
        self.movies_df = None
        self.embedding_system = None
        self.faiss_index = None
        self.sentence_model = None
        self.runtime_search = None
        
        logger.info("Initializing Movie Query Engine...")
        self._load_data()
        
        if use_runtime_weights:
            self._load_runtime_weighted_search()
        else:
            self._load_embedding_system()
            self._load_faiss_index()
            
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
        """Load the runtime weighted search system (recommended)"""
        components_path = self.model_path.with_suffix(".npz")

        if not components_path.exists():
            raise FileNotFoundError(f"Embedding components not found at {components_path}")
        
        self.runtime_search = RuntimeWeightedSearch()
        self.runtime_search.load_components(str(components_path))
        
        logger.info("Loaded runtime weighted search system")
    
    def _load_embedding_system(self):
        """Load the culturally-aware embedding system"""
        model_path = self.model_path.with_suffix(".json")
        if not model_path.exists():
            raise FileNotFoundError(f"Embedding model not found at {model_path}")
        
        # Create embedding system with default parameters (will be overwritten)
        self.embedding_system = CulturallyAwareMovieEmbedding()
        
        # Load the model (this will restore all parameters and stored embeddings)
        self.embedding_system.load_model(str(model_path))
        
        # Load sentence transformer model for query encoding
        self.sentence_model = SentenceTransformer("intfloat/multilingual-e5-large")
        
        logger.info("Loaded culturally-aware embedding system")
    
    def _load_faiss_index(self):
        """Load the pre-built FAISS index"""
        index_path = self.model_path.with_suffix(".index")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
    
    def search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Search for movies based on the request
        
        Args:
            request: SearchRequest with query and preferences
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        logger.info(f"Searching for: '{request.query}'")
        
        if self.use_runtime_weights and self.runtime_search:
            # Use runtime weighted search for better cultural tuning
            raw_results = self.runtime_search.search(
                query_text=request.query,
                user_countries=request.countries,
                user_languages=request.languages,
                prefer_popular=request.prefer_popular,
                plot_weight=request.plot_weight,
                genre_weight=request.genre_weight,
                localization_weight=request.localization_weight,
                popularity_weight=request.popularity_weight,
                top_k=min(request.max_results * 2, 100)  # Get extra results for filtering
            )
            
            # Convert to SearchResult format and apply filters
            results = []
            for raw_result in raw_results:
                # Apply filters based on movie metadata
                if not self._passes_filters_raw(raw_result, request):
                    continue
                
                result = SearchResult(
                    title=raw_result['title'],
                    year=raw_result['year'],
                    rating=raw_result['rating'],
                    votes=raw_result['votes'],
                    popularity_score=raw_result['popularity'],
                    countries=self._parse_list_field(raw_result['countries']),
                    languages=self._parse_list_field(raw_result['languages']),
                    genres=self._parse_list_field(raw_result['genres']),
                    plot=raw_result['plot'],
                    similarity=raw_result['similarity'],
                    rank=raw_result['rank']
                )
                
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= request.max_results:
                    break
        else:
            # Fall back to original fixed-weight search
            query_embedding = self.embedding_system.transform_query(
                request.query,
                user_countries=request.countries,
                user_languages=request.languages,
                prefer_popular=request.prefer_popular
            )
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(
                query_embedding, 
                min(request.max_results * 2, 100)  # Get extra results for filtering
            )
            
            # Convert to search results
            results = []
            for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                idx_int = int(idx)
                if idx_int >= len(self.movies_df):
                    continue
                    
                movie_row = self.movies_df.row(idx_int, named=True)
                
                # Apply filters
                if not self._passes_filters(movie_row, request):
                    continue
                
                # Create search result
                result = SearchResult(
                    title=movie_row['title'],
                    year=int(movie_row['year']) if movie_row['year'] else 0,
                    rating=float(movie_row['rating']) if movie_row['rating'] else 0.0,
                    votes=int(movie_row['votes']) if movie_row['votes'] else 0,
                    popularity_score=float(movie_row['sn_votes']) if movie_row['sn_votes'] else 0.0,
                    countries=self._parse_list_field(movie_row['country']),
                    languages=self._parse_list_field(movie_row['language']),
                    genres=self._parse_list_field(movie_row['genre']),
                    plot=str(movie_row['plot']) if movie_row['plot'] else "",
                    similarity=float(similarity),
                    rank=rank + 1
                )
                
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= request.max_results:
                    break
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def _passes_filters(self, movie_row: Dict[str, Any], request: SearchRequest) -> bool:
        """Check if a movie passes the search filters"""
        
        # Rating filter
        if request.min_rating is not None:
            rating = movie_row.get('rating')
            if rating is None or rating < request.min_rating:
                return False
        
        # Votes filter
        if request.min_votes is not None:
            votes = movie_row.get('votes')
            if votes is None or votes < request.min_votes:
                return False
        
        # Year range filter
        if request.year_range is not None:
            year = movie_row.get('year')
            if year is None:
                return False
            min_year, max_year = request.year_range
            if year < min_year or year > max_year:
                return False
        
        return True
    
    def _passes_filters_raw(self, result: dict, request: SearchRequest) -> bool:
        """Check if a raw search result passes the search filters"""
        
        # Rating filter
        if request.min_rating is not None:
            rating = result.get('rating')
            if rating is None or rating < request.min_rating:
                return False
        
        # Votes filter
        if request.min_votes is not None:
            votes = result.get('votes')
            if votes is None or votes < request.min_votes:
                return False
        
        # Year range filter
        if request.year_range is not None:
            year = result.get('year')
            if year is None:
                return False
            min_year, max_year = request.year_range
            if year < min_year or year > max_year:
                return False
        
        return True
    
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
    
    def search_simple(self, query: str, max_results: int = 10, 
                     prefer_popular: bool = True) -> List[SearchResult]:
        """
        Simple search interface for quick queries
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            prefer_popular: Whether to prefer popular movies
            
        Returns:
            List of search results
        """
        request = SearchRequest(
            query=query,
            max_results=max_results,
            prefer_popular=prefer_popular
        )
        return self.search(request)
    
    def get_recommendations_for_movie(self, movie_title: str, 
                                    max_results: int = 10,
                                    use_cultural_weights: bool = True,
                                    plot_weight: float = 0.44,
                                    genre_weight: float = 0.13,
                                    localization_weight: float = 0.1,
                                    popularity_weight: float = 0.33) -> List[SearchResult]:
        """
        Get movie recommendations based on a specific movie using culturally-aware search
        
        Args:
            movie_title: Title of the movie to base recommendations on
            max_results: Maximum number of recommendations
            use_cultural_weights: Whether to use cultural preset weights (recommended)
            plot_weight: Weight for plot similarity (used if use_cultural_weights=True)
            genre_weight: Weight for genre similarity
            localization_weight: Weight for cultural localization
            popularity_weight: Weight for popularity
            
        Returns:
            List of similar movies using cultural awareness
        """
        # Find the movie in our database
        movie_matches = self.movies_df.filter(
            pl.col("title").str.to_lowercase().str.contains(movie_title.lower())
        )
        
        if len(movie_matches) == 0:
            logger.warning(f"Movie '{movie_title}' not found")
            return []
        
        # Use the first match
        movie_row = movie_matches.row(0, named=True)
        
        # Extract cultural context from the source movie
        source_countries = self._parse_list_field(movie_row['country'])
        source_languages = self._parse_list_field(movie_row['language'])
        
        # Create a query based on the movie's genres and plot
        genres = self._parse_list_field(movie_row['genre'])
        plot_snippet = str(movie_row['plot'])[:200] if movie_row['plot'] else ""
        
        # Combine genres and plot for the query
        genre_text = " ".join(genres).lower()
        query = f"{genre_text} {plot_snippet}"
        
        logger.info(f"Finding movies similar to '{movie_row['title']}' with cultural awareness")
        logger.info(f"Source movie context: countries={source_countries}, languages={source_languages}")
        
        if use_cultural_weights and self.use_runtime_weights:
            # Use culturally-aware search with the movie's cultural context
            request = SearchRequest(
                query=query,
                countries=source_countries if source_countries else None,
                languages=source_languages if source_languages else None,
                prefer_popular=False,  # Focus on quality over popularity for recommendations
                max_results=max_results + 1,  # Get extra to filter out the original
                plot_weight=plot_weight,
                genre_weight=genre_weight,
                localization_weight=localization_weight,
                popularity_weight=popularity_weight
            )
            
            results = self.search(request)
            
            # Filter out the original movie from recommendations
            filtered_results = []
            original_title_lower = movie_row['title'].lower()
            
            for result in results:
                # Skip if it's the same movie (case-insensitive title match)
                if result.title.lower() == original_title_lower:
                    continue
                
                filtered_results.append(result)
                
                # Stop when we have enough results
                if len(filtered_results) >= max_results:
                    break
            
            return filtered_results
        else:
            # Fall back to simple search (original behavior)
            return self.search_simple(query, max_results + 1, prefer_popular=True)[1:]  # Skip the original movie
    
    def get_movie_stats(self) -> Dict[str, Any]:
        """Get statistics about the movie database"""
        stats = {
            'total_movies': len(self.movies_df),
            'year_range': (
                int(self.movies_df['year'].min()),
                int(self.movies_df['year'].max())
            ),
            'avg_rating': float(self.movies_df['rating'].mean()),
            'total_countries': len(self.embedding_system.country_vocab),
            'total_languages': len(self.embedding_system.language_vocab),
            'embedding_dimension': self.faiss_index.d,
            'index_size': self.faiss_index.ntotal
        }
        return stats


def main():
    """Command-line interface for the query engine"""
    parser = argparse.ArgumentParser(description='ReelRaider Movie Query Engine')
    parser.add_argument('query', nargs='?', help='Search query for movies')
    parser.add_argument('--model-path', '-m', type=str, default='datasets/dist/culturally_aware_model',
                      help='Name of the culturally-aware embedding model for <model_name>.json/npz/index')
    parser.add_argument('--data-file', '-d', type=str, default='datasets/dist/movies_processed_sn.parquet',
                      help='Path to the processed movie data file')
    parser.add_argument('--max-results', '-n', type=int, default=10,
                      help='Maximum number of results (default: 10)')
    parser.add_argument('--countries', nargs='+', 
                      help='Preferred countries (e.g. "USA" "UK")')
    parser.add_argument('--languages', nargs='+',
                      help='Preferred languages (e.g. "English" "Spanish")')
    parser.add_argument('--no-popular', action='store_true',
                      help='Do not prefer popular movies')
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
    parser.add_argument('--no-runtime-weights', action='store_true',
                      help='Use fixed embedding weights instead of runtime adjustment')
    parser.add_argument('--preset', choices=['balanced', 'popular', 'cultural', 'niche', 'ultra-cultural'],
                      help='Use preset weight configuration: '
                           'balanced (default, general purpose with cultural awareness), '
                           'popular (mainstream focus, balanced cultural), '
                           'cultural (prioritizes cultural relevance), '
                           'niche (art house/indie films, semantic focus), '
                           'ultra-cultural (DEPRECATED: extreme localization)')
    
    args = parser.parse_args()
    
    # Apply preset configurations if specified (optimized for 128-dim localization)
    if args.preset:
        if args.preset == 'balanced':
            # Default balanced preset - good for general queries with light cultural awareness
            args.plot_weight, args.genre_weight = 0.5, 0.25
            args.localization_weight, args.popularity_weight = 0.1, 0.15
            args.no_popular = False  # prefer_popular = True
        elif args.preset == 'popular':
            # Mainstream preset - balances cultural awareness with popularity
            args.plot_weight, args.genre_weight = 0.4, 0.2
            args.localization_weight, args.popularity_weight = 0.2, 0.2
            args.no_popular = False  # prefer_popular = True
        elif args.preset == 'cultural':
            # Cultural preset - emphasizes plot semantics while maintaining cultural relevance
            args.plot_weight, args.genre_weight = 0.44, 0.13
            args.localization_weight, args.popularity_weight = 0.1, 0.33
            args.no_popular = True  # prefer_popular = False
        elif args.preset == 'niche':
            # Niche preset - emphasizes semantic/genre matching, finds hidden gems
            args.plot_weight, args.genre_weight = 0.6, 0.3
            args.localization_weight, args.popularity_weight = 0.05, 0.05
            args.no_popular = True  # prefer_popular = False
        elif args.preset == 'ultra-cultural':
            # DEPRECATED: Extreme cultural localization - essentially a country filter
            # This preset bypasses semantic ML and is not recommended for general use
            args.plot_weight, args.genre_weight = 0.15, 0.05
            args.localization_weight, args.popularity_weight = 0.75, 0.05
            args.no_popular = True  # prefer_popular = False
    
    # Initialize query engine
    try:
        engine = MovieQueryEngine(model_path=args.model_path, data_file=args.data_file, use_runtime_weights=not args.no_runtime_weights)
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
        if engine.use_runtime_weights:
            print(f"Weights: plot={args.plot_weight:.2f}, genre={args.genre_weight:.2f}, "
                  f"localization={args.localization_weight:.2f}, popularity={args.popularity_weight:.2f}")
        print()
        
        results = engine.get_recommendations_for_movie(
            args.similar_to, 
            args.max_results,
            use_cultural_weights=not args.no_runtime_weights,
            plot_weight=args.plot_weight,
            genre_weight=args.genre_weight,
            localization_weight=args.localization_weight,
            popularity_weight=args.popularity_weight
        )
        _print_results(results)
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
        prefer_popular=not args.no_popular,
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
    print(f"Prefer popular: {request.prefer_popular}")
    if engine.use_runtime_weights:
        print(f"Weights: plot={args.plot_weight:.2f}, genre={args.genre_weight:.2f}, "
              f"localization={args.localization_weight:.2f}, popularity={args.popularity_weight:.2f}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print()
    
    _print_results(results)
    
    return 0


def _print_results(results: List[SearchResult]):
    """Print search results in a nice format"""
    if not results:
        print("No results found.")
        return
    
    for result in results:
        print(f"{result.rank}. {result.title} ({result.year})")
        print(f"   Rating: {result.rating:.1f}/10, Votes: {result.votes:,}, Popularity: {result.popularity_score:.3f}")
        print(f"   Genres: {', '.join(result.genres)}")
        print(f"   Countries: {', '.join(result.countries)}")
        print(f"   Languages: {', '.join(result.languages)}")
        print(f"   Similarity: {result.similarity:.4f}")
        
        # Show plot summary (first 150 chars)
        if result.plot:
            plot_summary = result.plot[:150] + "..." if len(result.plot) > 150 else result.plot
            print(f"   Plot: {plot_summary}")
        
        print()


if __name__ == "__main__":
    exit(main())
