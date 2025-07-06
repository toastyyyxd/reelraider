// gRPC client service for ReelRaider interop
import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load the protobuf definition
const PROTO_PATH = path.join(__dirname, '../../interop_service.proto');
const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true
});

const reelraider = grpc.loadPackageDefinition(packageDefinition).reelraider as any;

// Error code constants (matching the protobuf enum)
export const ErrorCode = {
    ERROR_NONE: 0,
    ERROR_INVALID_QUERY: 1,
    ERROR_INVALID_IMDB_ID: 2,
    ERROR_MOVIE_NOT_FOUND: 3,
    ERROR_INVALID_PARAMETERS: 4,
    ERROR_SEARCH_FAILED: 5,
    ERROR_RECOMMENDATIONS_FAILED: 6,
    ERROR_SERVICE_UNAVAILABLE: 7
} as const;

// Search preset constants
export const SearchPreset = {
    PRESET_UNSPECIFIED: 0,
    PRESET_POPULAR: 1,
    PRESET_NICHE: 2,
    PRESET_CULTURAL: 3
} as const;

// TypeScript interfaces for the gRPC messages
export interface SearchRequest {
    query: string;
    countries?: string[];
    languages?: string[];
    max_results?: number;
    min_rating?: number;
    min_votes?: number;
    year_from?: number;
    year_to?: number;
    preset?: number;
    plot_weight?: number;
    genre_weight?: number;
    localization_weight?: number;
    popularity_weight?: number;
}

export interface RecommendationRequest {
    imdb_id: string;
    countries?: string[];
    languages?: string[];
    max_results?: number;
    use_cultural_weights?: boolean;
    preset?: number;
    plot_weight?: number;
    genre_weight?: number;
    localization_weight?: number;
    popularity_weight?: number;
}

export interface MovieResult {
    title: string;
    imdb_id: string;
    year: number;
    rating: number;
    votes: number;
    popularity_score: number;
    countries: string[];
    languages: string[];
    genres: string[];
    plot: string;
    poster: string;
    similarity: number;
    rank: number;
}

export interface ErrorInfo {
    code: number;
    message: string;
    details: string;
}

export interface SearchResponse {
    results: MovieResult[];
    error?: ErrorInfo;
}

export class GrpcClient {
    private client: any;

    constructor(serverAddress = 'localhost:50051') {
        this.client = new reelraider.InteropService(
            serverAddress,
            grpc.credentials.createInsecure()
        );
    }

    /**
     * Handle a search response with proper error checking
     */
    private handleResponse(response: SearchResponse, operation = 'Operation'): MovieResult[] {
        if (response.error && response.error.code !== ErrorCode.ERROR_NONE) {
            const errorName = this.getErrorName(response.error.code);
            const errorMsg = `${operation} Error [${errorName}]: ${response.error.message}`;
            if (response.error.details) {
                throw new Error(`${errorMsg} - ${response.error.details}`);
            }
            throw new Error(errorMsg);
        }
        
        return response.results || [];
    }

    /**
     * Get human-readable error name
     */
    private getErrorName(errorCode: number): string {
        const errorNames: Record<number, string> = {
            [ErrorCode.ERROR_NONE]: 'ERROR_NONE',
            [ErrorCode.ERROR_INVALID_QUERY]: 'ERROR_INVALID_QUERY',
            [ErrorCode.ERROR_INVALID_IMDB_ID]: 'ERROR_INVALID_IMDB_ID',
            [ErrorCode.ERROR_MOVIE_NOT_FOUND]: 'ERROR_MOVIE_NOT_FOUND',
            [ErrorCode.ERROR_INVALID_PARAMETERS]: 'ERROR_INVALID_PARAMETERS',
            [ErrorCode.ERROR_SEARCH_FAILED]: 'ERROR_SEARCH_FAILED',
            [ErrorCode.ERROR_RECOMMENDATIONS_FAILED]: 'ERROR_RECOMMENDATIONS_FAILED',
            [ErrorCode.ERROR_SERVICE_UNAVAILABLE]: 'ERROR_SERVICE_UNAVAILABLE'
        };
        return errorNames[errorCode] || `UNKNOWN_ERROR_${errorCode}`;
    }

    /**
     * Perform a search with error handling
     */
    async searchMovies(query: string, options: Partial<SearchRequest> = {}): Promise<MovieResult[]> {
        return new Promise((resolve, reject) => {
            const request: SearchRequest = {
                query,
                max_results: options.max_results || 10,
                preset: options.preset || SearchPreset.PRESET_UNSPECIFIED,
                countries: options.countries || [],
                languages: options.languages || [],
                min_rating: options.min_rating || 0,
                min_votes: options.min_votes || 0,
                year_from: options.year_from || 0,
                year_to: options.year_to || 0,
                ...options
            };
            
            this.client.Search(request, (error: grpc.ServiceError | null, response: SearchResponse) => {
                if (error) {
                    // gRPC-level error
                    reject(new Error(`gRPC Error: ${error.code}: ${error.details}`));
                    return;
                }
                
                try {
                    const results = this.handleResponse(response, 'Search');
                    resolve(results);
                } catch (err) {
                    reject(err);
                }
            });
        });
    }

    /**
     * Get movie recommendations with error handling
     */
    async getRecommendations(imdbId: string, options: Partial<RecommendationRequest> = {}): Promise<MovieResult[]> {
        return new Promise((resolve, reject) => {
            const request: RecommendationRequest = {
                imdb_id: imdbId,
                max_results: options.max_results || 10,
                use_cultural_weights: options.use_cultural_weights || false,
                preset: options.preset || SearchPreset.PRESET_UNSPECIFIED,
                countries: options.countries || [],
                languages: options.languages || [],
                ...options
            };
            
            this.client.GetRecommendations(request, (error: grpc.ServiceError | null, response: SearchResponse) => {
                if (error) {
                    // gRPC-level error
                    reject(new Error(`gRPC Error: ${error.code}: ${error.details}`));
                    return;
                }
                
                try {
                    const results = this.handleResponse(response, 'Recommendations');
                    resolve(results);
                } catch (err) {
                    reject(err);
                }
            });
        });
    }

    /**
     * Close the gRPC client connection
     */
    close(): void {
        this.client.close();
    }
}

// Create a singleton instance
export const grpcClient = new GrpcClient();
