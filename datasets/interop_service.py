#!/usr/bin/env python3
"""
gRPC Interop Service for ReelRaider Movie Query Engine

This service provides a gRPC interface to the MovieQueryEngine, enabling
Node.js and other clients to perform movie searches via protocol buffers.
It supports both custom search requests and preset configurations.
"""

import argparse
import asyncio
import grpc
from concurrent import futures
from typing import List, Optional, Tuple, Dict, Any
import polars as pl

from datasets.utils import logger, tconst_to_tid, tid_to_tconst
from datasets.query_engine import MovieQueryEngine, SearchRequest as QuerySearchRequest

# Generated protobuf imports (will be available after running protoc)
try:
    from . import interop_service_pb2
    from . import interop_service_pb2_grpc
except ImportError:
    try:
        import interop_service_pb2
        import interop_service_pb2_grpc
    except ImportError:
        logger.error("gRPC protobuf files not found. Please run: python -m grpc_tools.protoc -I. --python_out=datasets --grpc_python_out=datasets interop_service.proto")
        raise

class InteropServicer(interop_service_pb2_grpc.InteropServiceServicer):
    """gRPC servicer implementation for movie search"""
    
    def __init__(self, model_path: str, data_file: str):
        """Initialize the servicer with MovieQueryEngine"""
        self.engine = MovieQueryEngine(model_path=model_path, data_file=data_file)
        logger.info("InteropServicer initialized")

    def Search(self, request: interop_service_pb2.SearchRequest, context):
        """Handle search requests (both custom and preset-based)"""
        try:
            # Validate search query
            if not request.query or not request.query.strip():
                return self._create_error_response(
                    interop_service_pb2.ERROR_INVALID_QUERY,
                    "Search query cannot be empty",
                    "Please provide a valid search query"
                )
            
            # Validate parameters
            if request.max_results < 0:
                return self._create_error_response(
                    interop_service_pb2.ERROR_INVALID_PARAMETERS,
                    "max_results cannot be negative",
                    "Please provide a non-negative value for max_results"
                )
            
            if request.min_rating < 0 or request.min_rating > 10:
                return self._create_error_response(
                    interop_service_pb2.ERROR_INVALID_PARAMETERS,
                    "min_rating must be between 0 and 10",
                    "Please provide a rating between 0.0 and 10.0"
                )
            
            # Convert protobuf request to internal SearchRequest
            search_request = self._convert_search_request(request)
            
            # Perform the search
            results_df = self.engine.search(search_request)
            
            # Check if we got any results
            if len(results_df) == 0:
                logger.info(f"No results found for query: '{request.query}'")
                # Return empty results with no error (valid case)
                return interop_service_pb2.SearchResponse()
            
            # Convert results to protobuf response
            response = self._convert_to_response(results_df)
            
            logger.info(f"Search completed: {len(response.results)} results")
            return response
            
        except ValueError as e:
            logger.error(f"Invalid search parameters: {e}")
            return self._create_error_response(
                interop_service_pb2.ERROR_INVALID_PARAMETERS,
                f"Invalid parameters: {str(e)}",
                "Please check your search parameters and try again"
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._create_error_response(
                interop_service_pb2.ERROR_SEARCH_FAILED,
                f"Search operation failed: {str(e)}",
                "Please try again later or contact support if the problem persists"
            )
    
    def GetRecommendations(self, request: interop_service_pb2.RecommendationRequest, context):
        """Handle movie recommendation requests"""
        try:
            # Validate IMDb ID
            if not request.imdb_id or not request.imdb_id.strip():
                return self._create_error_response(
                    interop_service_pb2.ERROR_INVALID_IMDB_ID,
                    "IMDb ID cannot be empty",
                    "Please provide a valid IMDb ID (e.g., tt0816692)"
                )
            
            # Convert IMDb ID to TID
            try:
                movie_tid = tconst_to_tid(request.imdb_id)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid IMDb ID '{request.imdb_id}': {e}")
                return self._create_error_response(
                    interop_service_pb2.ERROR_INVALID_IMDB_ID,
                    f"Invalid IMDb ID format: {request.imdb_id}",
                    "Please provide a valid IMDb ID (e.g., tt0816692)"
                )
            
            # Validate parameters
            if request.max_results < 0:
                return self._create_error_response(
                    interop_service_pb2.ERROR_INVALID_PARAMETERS,
                    "max_results cannot be negative",
                    "Please provide a non-negative value for max_results"
                )
            
            # Convert protobuf request to method parameters
            weights = self._get_recommendation_weights(request)
            
            # Perform the recommendation search using TID
            results_df = self.engine.get_recommendations_for_movie(
                movie_tid=movie_tid,
                max_results=request.max_results if request.max_results > 0 else 10,
                use_cultural_weights=request.use_cultural_weights,
                plot_weight=weights['plot_weight'],
                genre_weight=weights['genre_weight'],
                localization_weight=weights['localization_weight'],
                popularity_weight=weights['popularity_weight']
            )
            
            # Check if we got any results
            if len(results_df) == 0:
                logger.info(f"No recommendations found for IMDb ID '{request.imdb_id}' (TID: {movie_tid})")
                return self._create_error_response(
                    interop_service_pb2.ERROR_MOVIE_NOT_FOUND,
                    f"No recommendations found for movie: {request.imdb_id}",
                    "Movie may not exist in our database or may not have sufficient data for recommendations"
                )
            
            # Convert results to protobuf response
            response = self._convert_to_response(results_df)
            
            logger.info(f"Recommendations completed: {len(response.results)} results for IMDb ID '{request.imdb_id}' (TID: {movie_tid})")
            return response
            
        except ValueError as e:
            logger.error(f"Invalid recommendation parameters: {e}")
            return self._create_error_response(
                interop_service_pb2.ERROR_INVALID_PARAMETERS,
                f"Invalid parameters: {str(e)}",
                "Please check your recommendation parameters and try again"
            )
        except Exception as e:
            logger.error(f"Recommendations failed: {e}")
            return self._create_error_response(
                interop_service_pb2.ERROR_RECOMMENDATIONS_FAILED,
                f"Recommendation operation failed: {str(e)}",
                "Please try again later or contact support if the problem persists"
            )
    
    def _convert_search_request(self, request: interop_service_pb2.SearchRequest) -> QuerySearchRequest:
        """Convert protobuf SearchRequest to internal SearchRequest"""
        # Handle preset if specified
        if request.preset != interop_service_pb2.PRESET_UNSPECIFIED:
            weights = self._get_preset_weights(request.preset)
            # Use preset weights as defaults, but allow override if custom weights are provided
            plot_weight = request.plot_weight if request.plot_weight > 0 else weights['plot_weight']
            genre_weight = request.genre_weight if request.genre_weight > 0 else weights['genre_weight']
            localization_weight = request.localization_weight if request.localization_weight > 0 else weights['localization_weight']
            popularity_weight = request.popularity_weight if request.popularity_weight > 0 else weights['popularity_weight']
        else:
            # Use provided weights or defaults
            plot_weight = request.plot_weight if request.plot_weight > 0 else 0.5
            genre_weight = request.genre_weight if request.genre_weight > 0 else 0.25
            localization_weight = request.localization_weight if request.localization_weight > 0 else 0.1
            popularity_weight = request.popularity_weight if request.popularity_weight > 0 else 0.15
        
        return QuerySearchRequest(
            query=request.query,
            countries=list(request.countries) if request.countries else None,
            languages=list(request.languages) if request.languages else None,
            max_results=request.max_results if request.max_results > 0 else 20,
            min_rating=request.min_rating if request.min_rating > 0 else None,
            min_votes=request.min_votes if request.min_votes > 0 else None,
            year_range=self._get_year_range(request.year_from, request.year_to),
            plot_weight=plot_weight,
            genre_weight=genre_weight,
            localization_weight=localization_weight,
            popularity_weight=popularity_weight
        )

    
    def _get_preset_weights(self, preset: int) -> Dict[str, float]:
        """Get weight configuration for a preset"""
        presets = {
            interop_service_pb2.PRESET_POPULAR: {
                'plot_weight': 0.515,
                'genre_weight': 0.215,
                'localization_weight': 0.05,
                'popularity_weight': 0.2
            },
            interop_service_pb2.PRESET_CULTURAL: {
                'plot_weight': 0.532,
                'genre_weight': 0.1,
                'localization_weight': 0.16,
                'popularity_weight': 0.128
            },
            interop_service_pb2.PRESET_NICHE: {
                'plot_weight': 0.7,
                'genre_weight': 0.2,
                'localization_weight': 0.05,
                'popularity_weight': 0.05
            }
        }
        
        return presets.get(preset, {
            'plot_weight': 0.5,
            'genre_weight': 0.25,
            'localization_weight': 0.1,
            'popularity_weight': 0.15
        })
    
    def _get_recommendation_weights(self, request: interop_service_pb2.RecommendationRequest) -> Dict[str, float]:
        """Get weight configuration for recommendation request"""
        # Handle preset if specified
        if request.preset != interop_service_pb2.PRESET_UNSPECIFIED:
            preset_weights = self._get_preset_weights(request.preset)
            # Use preset weights as defaults, but allow override if custom weights are provided
            return {
                'plot_weight': request.plot_weight if request.plot_weight > 0 else preset_weights['plot_weight'],
                'genre_weight': request.genre_weight if request.genre_weight > 0 else preset_weights['genre_weight'],
                'localization_weight': request.localization_weight if request.localization_weight > 0 else preset_weights['localization_weight'],
                'popularity_weight': request.popularity_weight if request.popularity_weight > 0 else preset_weights['popularity_weight']
            }
        else:
            # Use provided weights or defaults (matching the query engine defaults for recommendations)
            return {
                'plot_weight': request.plot_weight if request.plot_weight > 0 else 0.44,
                'genre_weight': request.genre_weight if request.genre_weight > 0 else 0.13,
                'localization_weight': request.localization_weight if request.localization_weight > 0 else 0.1,
                'popularity_weight': request.popularity_weight if request.popularity_weight > 0 else 0.33
            }
    
    def _get_year_range(self, year_from: int, year_to: int) -> Optional[Tuple[int, int]]:
        """Convert year range parameters to tuple"""
        if year_from > 0 or year_to > 0:
            return (year_from if year_from > 0 else 1900, 
                   year_to if year_to > 0 else 2030)
        return None
    
    def _convert_to_response(self, results_df: pl.DataFrame) -> interop_service_pb2.SearchResponse:
        """Convert Polars DataFrame to protobuf SearchResponse"""
        response = interop_service_pb2.SearchResponse()
        
        for row in results_df.iter_rows(named=True):
            movie_result = interop_service_pb2.MovieResult()
            movie_result.title = row.get("title", "")
            
            # Convert tid to IMDb tconst format (tt0123456)
            tid = row.get("tid", 0)
            if tid:
                movie_result.imdb_id = tid_to_tconst(tid)
            else:
                movie_result.imdb_id = row.get("imdb_id", "")  # fallback
            
            movie_result.year = int(row.get("year", 0))
            movie_result.rating = float(row.get("final_rating", 0.0))
            movie_result.votes = int(row.get("votes", 0))
            movie_result.popularity_score = float(row.get("sn_votes", 0.0))
            
            # Handle list fields
            movie_result.countries.extend(self._parse_list_field(row.get("country", [])))
            movie_result.languages.extend(self._parse_list_field(row.get("language", [])))
            movie_result.genres.extend(self._parse_list_field(row.get("genre", [])))
            
            movie_result.plot = row.get("plot", "")
            movie_result.poster = row.get("poster", "")
            movie_result.similarity = float(row.get("similarity", 0.0))
            movie_result.rank = int(row.get("rank", 0))
            
            response.results.append(movie_result)
        
        return response
    
    def _create_error_response(self, error_code: int, message: str, details: str = "") -> interop_service_pb2.SearchResponse:
        """Create a SearchResponse with error information"""
        response = interop_service_pb2.SearchResponse()
        response.error.code = error_code
        response.error.message = message
        response.error.details = details
        return response
    
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


def serve(model_path: str, data_file: str, port: int = 50051):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    interop_service_pb2_grpc.add_InteropServiceServicer_to_server(
        InteropServicer(model_path, data_file), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"ReelRaider Interop Service started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ReelRaider gRPC Interop Service')
    parser.add_argument('--model-path', '-m', type=str, 
                       default='datasets/dist/culturally_aware_model',
                       help='Base path for the culturally-aware embedding model')
    parser.add_argument('--data-file', '-d', type=str, 
                       default='datasets/dist/movies_processed_sn.parquet',
                       help='Path to the processed movie data file')
    parser.add_argument('--port', '-p', type=int, default=50051,
                       help='Port to listen on (default: 50051)')
    
    args = parser.parse_args()
    
    try:
        serve(args.model_path, args.data_file, args.port)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
