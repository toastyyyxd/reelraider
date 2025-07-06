// TypeScript for ReelRaider frontend
import { movieApi, SearchPresets, type MovieResult } from './movie-api.js';
import { locationService, type UserLocation } from './location-service.js';

let userLocation: UserLocation | null = null;
let isUsingCulturalSearch = true; // Default to cultural search
let currentSearchQuery: string | null = null;
let currentRecommendationId: string | null = null;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize the app
    initializeApp();
});

async function initializeApp() {
    // Check backend health
    const isHealthy = await movieApi.healthCheck();
    console.log('Backend health:', isHealthy);

    // Detect user location
    await initializeUserLocation();

    // View switching functionality
    setupViewSwitching();
    
    // Search functionality
    setupSearchFunctionality();
    
    // Movie card interactions
    setupMovieCardInteractions();
    
    // Simulate video placeholder animation
    setupReelAnimation();
}

async function initializeUserLocation() {
    try {
        console.log('Detecting user location...');
        userLocation = await locationService.getUserLocation({
            useIPGeolocation: true,
            useBrowserGeolocation: false, // Don't ask for permission automatically
            fallbackCountry: 'United States'
        });
        
        console.log('User location detected:', userLocation);
        
        // Show location info in the UI (optional)
        updateLocationDisplay();
    } catch (error) {
        console.warn('Location detection failed:', error);
        userLocation = { country: 'United States', countryCode: 'US' };
    }
}

function updateLocationDisplay() {
    const locationIndicator = document.getElementById('locationIndicator');
    const locationText = locationIndicator?.querySelector('.location-text');
    
    if (!locationIndicator || !locationText) return;
    
    // Remove previous state classes
    locationIndicator.classList.remove('cultural', 'popular', 'toggling');
    
    if (userLocation?.country) {
        const searchType = isUsingCulturalSearch ? 'Cultural' : 'Popular';
        const languages = locationService.getLanguagesForAPI(userLocation);
        const languageText = languages.length > 0 ? ` (${languages.slice(0, 2).join(', ')})` : '';
        
        locationText.textContent = `${searchType} search for ${userLocation.country}${languageText}`;
        locationIndicator.classList.add('loaded');
        locationIndicator.classList.remove('error');
        
        // Add visual distinction for search type
        locationIndicator.classList.add(isUsingCulturalSearch ? 'cultural' : 'popular');
        
        // Update the icon based on search type
        const icon = locationIndicator.querySelector('i');
        if (icon) {
            icon.className = isUsingCulturalSearch ? 'fas fa-globe' : 'fas fa-fire';
        }
        
        console.log(`Using ${searchType} search for ${userLocation.country} with languages:`, languages);
    } else {
        locationText.textContent = 'Location detection failed - using global recommendations';
        locationIndicator.classList.add('error');
        locationIndicator.classList.remove('loaded');
    }
    
    // Add click handler for toggle
    locationIndicator.onclick = async (e) => {
        e.preventDefault();
        await toggleSearchMode();
    };
    
    // Add title for tooltip
    const searchType = isUsingCulturalSearch ? 'Cultural' : 'Popular';
    locationIndicator.title = `Click to switch to ${isUsingCulturalSearch ? 'Popular' : 'Cultural'} search (Currently: ${searchType})`;
}

async function toggleSearchMode() {
    const locationIndicator = document.getElementById('locationIndicator');
    const locationText = locationIndicator?.querySelector('.location-text');
    
    if (!locationIndicator || !locationText) return;
    
    // Add toggle animation
    locationIndicator.classList.add('toggling');
    
    // Toggle the search mode
    isUsingCulturalSearch = !isUsingCulturalSearch;
    
    // Show immediate feedback
    const newSearchType = isUsingCulturalSearch ? 'Cultural' : 'Popular';
    locationText.textContent = `Switched to ${newSearchType} search!`;
    
    // Update the display after animation
    setTimeout(() => {
        locationIndicator.classList.remove('toggling');
        updateLocationDisplay();
    }, 600);
    
    console.log(`Search mode toggled to: ${newSearchType}`);
}

async function refreshUserLocation() {
    const locationIndicator = document.getElementById('locationIndicator');
    const locationText = locationIndicator?.querySelector('.location-text');
    
    if (locationText) {
        locationText.textContent = 'Refreshing location...';
    }
    
    try {
        // Clear cache and re-detect
        locationService.clearCache();
        userLocation = await locationService.getUserLocation({
            useIPGeolocation: true,
            useBrowserGeolocation: true, // Ask for permission on manual refresh
            fallbackCountry: 'United States'
        });
        
        console.log('Location refreshed:', userLocation);
        updateLocationDisplay();
    } catch (error) {
        console.warn('Location refresh failed:', error);
        if (locationText) {
            locationText.textContent = 'Location refresh failed';
        }
        if (locationIndicator) {
            locationIndicator.classList.add('error');
        }
    }
}

function setupViewSwitching() {
    const islandButtons = document.querySelectorAll<HTMLDivElement>('.island-btn');
    const views = document.querySelectorAll<HTMLDivElement>('.view');

    islandButtons.forEach(button => {
        button.addEventListener('click', function () {
            const targetView = (this as HTMLElement).getAttribute('data-view');
            if (!targetView) return;

            // Update button states
            islandButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Update view visibility
            views.forEach(view => {
                view.classList.remove('active');
            });

            const target = document.getElementById(targetView);
            if (target) target.classList.add('active');
        });
    });
}

function setupSearchFunctionality() {
    const searchInput = document.querySelector<HTMLInputElement>('.search-bar input');
    const searchContainer = document.querySelector<HTMLDivElement>('.search-container');
    
    if (!searchInput) return;

    // Clear results when input is empty
    searchInput.addEventListener('input', (e) => {
        const query = (e.target as HTMLInputElement).value.trim();
        
        if (query.length === 0) {
            clearSearchResults();
        }
    });

    // Handle enter key - only trigger search on Enter
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const query = searchInput.value.trim();
            if (query.length >= 2) {
                performSearch(query);
            } else if (query.length === 0) {
                clearSearchResults();
            }
        }
    });
}

async function performSearch(query: string) {
    console.log('Searching for:', query);
    
    try {
        showLoadingState();
        
        // Prepare search options with user location and language
        const options: any = {
            max_results: 50,
            preset: isUsingCulturalSearch ? SearchPresets.PRESET_CULTURAL : SearchPresets.PRESET_POPULAR,
            min_rating: 0.5
        };
        
        // Add user location data if available
        if (userLocation) {
            const countries = locationService.getCountriesForAPI(userLocation);
            const languages = locationService.getLanguagesForAPI(userLocation);
            
            if (countries.length > 0) {
                options.countries = countries;
                console.log('Including user countries in search:', countries);
            }
            
            if (languages.length > 0) {
                options.languages = languages;
                console.log('Including user languages in search:', languages);
            }
        }
        
        const results = await movieApi.searchMovies(query, options);
        
        console.log('Search results received:', results.length, 'movies');
        console.log('First result:', results[0]);
        
        displaySearchResults(results);
    } catch (error) {
        console.error('Search failed:', error);
        showErrorState(error instanceof Error ? error.message : 'Search failed');
    }
}

function showLoadingState() {
    const galleryGrid = document.querySelector('.gallery-grid');
    if (galleryGrid) {
        galleryGrid.innerHTML = `
            <div class="loading-state">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Searching movies...</p>
            </div>
        `;
    }
}

function showErrorState(message: string) {
    const galleryGrid = document.querySelector('.gallery-grid');
    if (galleryGrid) {
        galleryGrid.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error: ${message}</p>
            </div>
        `;
    }
}

function clearSearchResults() {
    const galleryGrid = document.querySelector('.gallery-grid');
    if (galleryGrid) {
        galleryGrid.innerHTML = '<div class="no-results">Start typing to search for movies...</div>';
    }
}

function displaySearchResults(results: MovieResult[]) {
    const galleryGrid = document.querySelector('.gallery-grid');
    if (!galleryGrid) return;

    if (results.length === 0) {
        galleryGrid.innerHTML = '<div class="no-results">No movies found. Try a different search term.</div>';
        return;
    }

    galleryGrid.innerHTML = results.map(movie => createMovieCard(movie)).join('');
    
    // Re-setup movie card interactions for new cards
    setupMovieCardInteractions();
    triggerCardAnimations(); // Trigger animations for new cards
}

function createMovieCard(movie: MovieResult): string {
    const stars = createStarRating(movie.rating);
    const posterUrl = movie.poster || '/placeholder-poster.jpg';
    const genres = movie.genres.slice(0, 3).join(', ');
    const countries = movie.countries.slice(0, 2).join(', ');
    
    return `
        <div class="movie-card" data-imdb-id="${movie.imdb_id}">
            <div class="movie-poster" style="background-image: url('${posterUrl}')">
                <div class="movie-title">${movie.title}</div>
                <div class="movie-year">${movie.year}</div>
                <div class="movie-overlay">
                    <button class="btn-recommend" title="Get recommendations">
                        <i class="fas fa-magic"></i>
                    </button>
                </div>
            </div>
            <div class="movie-info">
                <div class="rating">
                    <div class="stars">${stars}</div>
                    <div class="rating-value">${(movie.rating * 10).toFixed(1)}/10</div>
                    <div class="vote-count">(${movie.votes.toLocaleString()} votes)</div>
                </div>
                <div class="movie-details">
                    <div class="genres">${genres}</div>
                    <div class="countries">${countries}</div>
                </div>
                <div class="summary">${movie.plot.slice(0, 150)}${movie.plot.length > 150 ? '...' : ''}</div>
            </div>
        </div>
    `;
}

function createStarRating(rating: number): string {
    const stars = [];
    // Rating comes as 0-1 normalized, convert to 0-5 stars
    const normalizedRating = rating * 5;
    
    for (let i = 1; i <= 5; i++) {
        if (i <= normalizedRating) {
            stars.push('<i class="fas fa-star"></i>');
        } else if (i - 0.5 <= normalizedRating) {
            stars.push('<i class="fas fa-star-half-alt"></i>');
        } else {
            stars.push('<i class="far fa-star"></i>');
        }
    }
    
    return stars.join('');
}

function setupMovieCardInteractions() {
    // Add hover effects to movie cards
    const movieCards = document.querySelectorAll<HTMLDivElement>('.movie-card');
    console.log('Setting up interactions for', movieCards.length, 'movie cards');
    
    movieCards.forEach(card => {
        card.addEventListener('mouseenter', function () {
            (this as HTMLElement).style.transform = 'translateY(-5px)';
        });

        card.addEventListener('mouseleave', function () {
            (this as HTMLElement).style.transform = 'translateY(0)';
        });
    });

    // Handle recommendation buttons
    const recommendButtons = document.querySelectorAll<HTMLButtonElement>('.btn-recommend');
    console.log('Setting up', recommendButtons.length, 'recommendation buttons');
    
    recommendButtons.forEach((button, index) => {
        console.log('Setting up button', index, button);
        button.addEventListener('click', async (e) => {
            console.log('Recommendation button clicked!', e);
            e.stopPropagation();
            e.preventDefault();
            
            const movieCard = button.closest('.movie-card') as HTMLElement;
            const imdbId = movieCard?.getAttribute('data-imdb-id');
            
            console.log('Movie card:', movieCard, 'IMDb ID:', imdbId);
            
            if (imdbId) {
                await getRecommendations(imdbId);
            } else {
                console.error('No IMDb ID found for movie card');
            }
        });
    });
}

async function getRecommendations(imdbId: string) {
    console.log('Getting recommendations for IMDb ID:', imdbId);
    
    try {
        showLoadingState();
        
        // Prepare recommendation options with user location and language
        const options: any = {
            max_results: 50,
            preset: isUsingCulturalSearch ? SearchPresets.PRESET_CULTURAL : SearchPresets.PRESET_POPULAR
        };
        
        // Add user location data if available
        if (userLocation) {
            const countries = locationService.getCountriesForAPI(userLocation);
            const languages = locationService.getLanguagesForAPI(userLocation);
            
            if (countries.length > 0) {
                options.countries = countries;
                console.log('Including user countries in recommendations:', countries);
            }
            
            if (languages.length > 0) {
                options.languages = languages;
                console.log('Including user languages in recommendations:', languages);
            }
        }
        
        console.log('Calling movieApi.getRecommendations with options:', options);
        const results = await movieApi.getRecommendations(imdbId, options);
        
        console.log('Received recommendations:', results.length, 'results');
        displaySearchResults(results);
        
        // Update search input to show we're in recommendation mode
        const searchInput = document.querySelector<HTMLInputElement>('.search-bar input');
        if (searchInput) {
            const searchType = isUsingCulturalSearch ? 'Cultural' : 'Popular';
            const locationText = userLocation?.country ? ` (${searchType} - ${userLocation.country})` : '';
            searchInput.placeholder = `Recommendations for movie ID: ${imdbId}${locationText}`;
        }
    } catch (error) {
        console.error('Recommendations failed:', error);
        showErrorState(error instanceof Error ? error.message : 'Recommendations failed');
    }
}

function setupReelAnimation() {
    const reelVideo = document.querySelector<HTMLDivElement>('.reel-video');
    const colors = ['#2d2b55', '#3e3b73', '#1a1838', '#4a2c6d'];
    let currentColor = 0;

    if (reelVideo) {
        setInterval(() => {
            reelVideo.style.background = `linear-gradient(45deg, ${colors[currentColor]}, ${colors[(currentColor + 1) % colors.length]})`;
            currentColor = (currentColor + 1) % colors.length;
        }, 3000);
    }
}

function triggerCardAnimations() {
    const cards = document.querySelectorAll<HTMLElement>('.movie-card');
    cards.forEach((card, index) => {
        // Reset animation
        card.style.animation = 'none';
        card.style.opacity = '0';
        
        // Force reflow
        card.offsetHeight;
        
        // Restart animation with stagger - faster for first 20, immediate for rest
        const delay = index < 20 ? index * 50 : 0; // 50ms intervals for first 20
        setTimeout(() => {
            card.style.animation = `slideInFade 0.6s ease-out forwards`;
        }, delay);
    });
}
