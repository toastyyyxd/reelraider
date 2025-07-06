// Frontend API service to communicate with the backend
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

export interface SearchOptions {
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

export interface RecommendationOptions {
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

export interface ApiResponse<T> {
    results?: T[];
    success: boolean;
    error?: string;
}

export const SearchPresets = {
    PRESET_UNSPECIFIED: 0,
    PRESET_POPULAR: 1,
    PRESET_NICHE: 2,
    PRESET_CULTURAL: 3
} as const;

export class MovieApiService {
    private baseUrl: string;

    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    /**
     * Search for movies based on a query
     */
    async searchMovies(query: string, options: SearchOptions = {}): Promise<MovieResult[]> {
        try {
            const response = await fetch(`${this.baseUrl}/api/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query,
                    options
                })
            });

            const data: ApiResponse<MovieResult> = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            if (!data.success) {
                throw new Error(data.error || 'Search failed');
            }

            return data.results || [];
        } catch (error) {
            console.error('Search error:', error);
            throw error instanceof Error ? error : new Error('Search failed');
        }
    }

    /**
     * Get movie recommendations based on an IMDb ID
     */
    async getRecommendations(imdbId: string, options: RecommendationOptions = {}): Promise<MovieResult[]> {
        try {
            const response = await fetch(`${this.baseUrl}/api/recommendations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    imdbId,
                    options
                })
            });

            const data: ApiResponse<MovieResult> = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            if (!data.success) {
                throw new Error(data.error || 'Recommendations failed');
            }

            return data.results || [];
        } catch (error) {
            console.error('Recommendations error:', error);
            throw error instanceof Error ? error : new Error('Recommendations failed');
        }
    }

    /**
     * Get available search presets
     */
    async getPresets(): Promise<Record<string, number>> {
        try {
            const response = await fetch(`${this.baseUrl}/api/presets`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            return data.presets || {};
        } catch (error) {
            console.error('Presets error:', error);
            return SearchPresets;
        }
    }

    /**
     * Health check
     */
    async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`);
            const data = await response.json();
            return response.ok && data.status === 'ok';
        } catch (error) {
            console.error('Health check error:', error);
            return false;
        }
    }
}

// Create a singleton instance
export const movieApi = new MovieApiService();
