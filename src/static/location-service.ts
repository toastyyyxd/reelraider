// Location detection service for ReelRaider
export interface UserLocation {
    country?: string;
    countryCode?: string;
    region?: string;
    city?: string;
    timezone?: string;
    latitude?: number;
    longitude?: number;
}

export interface LocationServiceOptions {
    useIPGeolocation?: boolean;
    useBrowserGeolocation?: boolean;
    fallbackCountry?: string;
}

export class LocationService {
    private cachedLocation: UserLocation | null = null;
    private detectionPromise: Promise<UserLocation> | null = null;

    /**
     * Get user's location using multiple detection methods
     */
    async getUserLocation(options: LocationServiceOptions = {}): Promise<UserLocation> {
        // Return cached location if available
        if (this.cachedLocation) {
            return this.cachedLocation;
        }

        // Return existing promise if detection is in progress
        if (this.detectionPromise) {
            return this.detectionPromise;
        }

        // Start new detection
        this.detectionPromise = this.detectLocation(options);
        const location = await this.detectionPromise;
        this.cachedLocation = location;
        return location;
    }

    /**
     * Detect location using multiple methods
     */
    private async detectLocation(options: LocationServiceOptions): Promise<UserLocation> {
        const location: UserLocation = {};

        // Method 1: Try IP-based geolocation first (fast and reliable)
        if (options.useIPGeolocation !== false) {
            try {
                const ipLocation = await this.detectLocationFromIP();
                Object.assign(location, ipLocation);
            } catch (error) {
                console.warn('IP geolocation failed:', error);
            }
        }

        // Method 2: Try browser geolocation API (more accurate but requires permission)
        if (options.useBrowserGeolocation === true && navigator.geolocation) {
            try {
                const browserLocation = await this.detectLocationFromBrowser();
                // Merge with IP location, preferring browser location for coordinates
                Object.assign(location, browserLocation);
            } catch (error) {
                console.warn('Browser geolocation failed:', error);
            }
        }

        // Method 3: Fallback using timezone
        if (!location.country) {
            try {
                const timezoneLocation = this.detectLocationFromTimezone();
                Object.assign(location, timezoneLocation);
            } catch (error) {
                console.warn('Timezone detection failed:', error);
            }
        }

        // Method 4: Ultimate fallback
        if (!location.country && options.fallbackCountry) {
            location.country = options.fallbackCountry;
        }

        return location;
    }

    /**
     * Detect location using IP geolocation service
     */
    private async detectLocationFromIP(): Promise<UserLocation> {
        try {
            // Use a free IP geolocation service
            const response = await fetch('https://ipapi.co/json/', {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();

            return {
                country: data.country_name,
                countryCode: data.country_code,
                region: data.region,
                city: data.city,
                timezone: data.timezone,
                latitude: parseFloat(data.latitude),
                longitude: parseFloat(data.longitude)
            };
        } catch (error) {
            // Fallback to another service
            try {
                const response = await fetch('https://ipinfo.io/json');
                const data = await response.json();
                
                return {
                    country: data.country,
                    countryCode: data.country,
                    region: data.region,
                    city: data.city,
                    timezone: data.timezone,
                    latitude: data.loc ? parseFloat(data.loc.split(',')[0]) : undefined,
                    longitude: data.loc ? parseFloat(data.loc.split(',')[1]) : undefined
                };
            } catch (fallbackError) {
                throw new Error('All IP geolocation services failed');
            }
        }
    }

    /**
     * Detect location using browser geolocation API
     */
    private async detectLocationFromBrowser(): Promise<UserLocation> {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error('Geolocation not supported'));
                return;
            }

            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude
                    });
                },
                (error) => {
                    reject(new Error(`Geolocation error: ${error.message}`));
                },
                {
                    timeout: 10000,
                    maximumAge: 600000, // 10 minutes
                    enableHighAccuracy: false
                }
            );
        });
    }

    /**
     * Detect location from browser timezone
     */
    private detectLocationFromTimezone(): UserLocation {
        const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        
        // Basic timezone to country mapping
        const timezoneCountryMap: Record<string, { country: string; countryCode: string }> = {
            'America/New_York': { country: 'United States', countryCode: 'US' },
            'America/Los_Angeles': { country: 'United States', countryCode: 'US' },
            'America/Chicago': { country: 'United States', countryCode: 'US' },
            'America/Denver': { country: 'United States', countryCode: 'US' },
            'Europe/London': { country: 'United Kingdom', countryCode: 'GB' },
            'Europe/Paris': { country: 'France', countryCode: 'FR' },
            'Europe/Berlin': { country: 'Germany', countryCode: 'DE' },
            'Europe/Rome': { country: 'Italy', countryCode: 'IT' },
            'Europe/Madrid': { country: 'Spain', countryCode: 'ES' },
            'Asia/Tokyo': { country: 'Japan', countryCode: 'JP' },
            'Asia/Shanghai': { country: 'China', countryCode: 'CN' },
            'Asia/Kolkata': { country: 'India', countryCode: 'IN' },
            'Australia/Sydney': { country: 'Australia', countryCode: 'AU' },
            'America/Toronto': { country: 'Canada', countryCode: 'CA' },
            'America/Sao_Paulo': { country: 'Brazil', countryCode: 'BR' }
        };

        const locationInfo = timezoneCountryMap[timezone];
        
        return {
            timezone,
            country: locationInfo?.country,
            countryCode: locationInfo?.countryCode
        };
    }

    /**
     * Convert location to countries array for API requests
     */
    getCountriesForAPI(location: UserLocation): string[] {
        const countries: string[] = [];
        
        if (location.country) {
            countries.push(location.country);
        }
        
        if (location.countryCode && location.countryCode !== location.country) {
            countries.push(location.countryCode);
        }
        
        return countries;
    }

    /**
     * Infer languages based on location
     */
    getLanguagesForAPI(location: UserLocation): string[] {
        const languages: string[] = [];
        
        // Country/region to language mapping
        const countryLanguageMap: Record<string, string[]> = {
            'Hong Kong': ['Cantonese', 'Chinese', 'English'],
            'HK': ['Cantonese', 'Chinese', 'English'],
            'China': ['Chinese', 'Mandarin'],
            'CN': ['Chinese', 'Mandarin'],
            'Taiwan': ['Chinese', 'Mandarin'],
            'TW': ['Chinese', 'Mandarin'],
            'Japan': ['Japanese'],
            'JP': ['Japanese'],
            'Korea': ['Korean'],
            'KR': ['Korean'],
            'South Korea': ['Korean'],
            'France': ['French'],
            'FR': ['French'],
            'Germany': ['German'],
            'DE': ['German'],
            'Italy': ['Italian'],
            'IT': ['Italian'],
            'Spain': ['Spanish'],
            'ES': ['Spanish'],
            'Mexico': ['Spanish'],
            'MX': ['Spanish'],
            'Brazil': ['Portuguese'],
            'BR': ['Portuguese'],
            'Portugal': ['Portuguese'],
            'PT': ['Portuguese'],
            'Russia': ['Russian'],
            'RU': ['Russian'],
            'India': ['Hindi', 'English'],
            'IN': ['Hindi', 'English'],
            'United Kingdom': ['English'],
            'GB': ['English'],
            'United States': ['English'],
            'US': ['English'],
            'Canada': ['English', 'French'],
            'CA': ['English', 'French'],
            'Australia': ['English'],
            'AU': ['English'],
            'New Zealand': ['English'],
            'NZ': ['English'],
            'Singapore': ['English', 'Chinese', 'Mandarin'],
            'SG': ['English', 'Chinese', 'Mandarin'],
            'Malaysia': ['English', 'Chinese', 'Mandarin'],
            'MY': ['English', 'Chinese', 'Mandarin'],
            'Thailand': ['Thai'],
            'TH': ['Thai'],
            'Vietnam': ['Vietnamese'],
            'VN': ['Vietnamese'],
            'Philippines': ['English', 'Filipino'],
            'PH': ['English', 'Filipino'],
            'Indonesia': ['Indonesian'],
            'ID': ['Indonesian'],
            'Netherlands': ['Dutch'],
            'NL': ['Dutch'],
            'Belgium': ['Dutch', 'French'],
            'BE': ['Dutch', 'French'],
            'Switzerland': ['German', 'French', 'Italian'],
            'CH': ['German', 'French', 'Italian'],
            'Austria': ['German'],
            'AT': ['German'],
            'Sweden': ['Swedish'],
            'SE': ['Swedish'],
            'Norway': ['Norwegian'],
            'NO': ['Norwegian'],
            'Denmark': ['Danish'],
            'DK': ['Danish'],
            'Finland': ['Finnish'],
            'FI': ['Finnish'],
            'Poland': ['Polish'],
            'PL': ['Polish'],
            'Czech Republic': ['Czech'],
            'CZ': ['Czech'],
            'Hungary': ['Hungarian'],
            'HU': ['Hungarian'],
            'Romania': ['Romanian'],
            'RO': ['Romanian'],
            'Greece': ['Greek'],
            'GR': ['Greek'],
            'Turkey': ['Turkish'],
            'TR': ['Turkish'],
            'Israel': ['Hebrew', 'English'],
            'IL': ['Hebrew', 'English'],
            'Egypt': ['Arabic'],
            'EG': ['Arabic'],
            'Saudi Arabia': ['Arabic'],
            'SA': ['Arabic'],
            'United Arab Emirates': ['Arabic', 'English'],
            'AE': ['Arabic', 'English'],
            'South Africa': ['English', 'Afrikaans'],
            'ZA': ['English', 'Afrikaans'],
            'Nigeria': ['English'],
            'NG': ['English'],
            'Kenya': ['English', 'Swahili'],
            'KE': ['English', 'Swahili'],
            'Argentina': ['Spanish'],
            'AR': ['Spanish'],
            'Chile': ['Spanish'],
            'CL': ['Spanish'],
            'Colombia': ['Spanish'],
            'CO': ['Spanish'],
            'Peru': ['Spanish'],
            'PE': ['Spanish'],
            'Venezuela': ['Spanish'],
            'VE': ['Spanish']
        };

        // Check country first
        if (location.country && countryLanguageMap[location.country]) {
            languages.push(...countryLanguageMap[location.country]);
        }
        
        // Check country code if different results
        if (location.countryCode && 
            location.countryCode !== location.country && 
            countryLanguageMap[location.countryCode]) {
            const codeLanguages = countryLanguageMap[location.countryCode];
            codeLanguages.forEach(lang => {
                if (!languages.includes(lang)) {
                    languages.push(lang);
                }
            });
        }

        // Special handling for regions within countries
        if (location.region) {
            const regionLanguageMap: Record<string, string[]> = {
                'Hong Kong': ['Cantonese', 'Chinese', 'English'],
                'Macau': ['Cantonese', 'Chinese', 'Portuguese'],
                'Quebec': ['French', 'English'],
                'Catalonia': ['Catalan', 'Spanish'],
                'Bavaria': ['German'],
                'Lombardy': ['Italian'],
                'Andalusia': ['Spanish'],
                'Scotland': ['English', 'Scottish Gaelic'],
                'Wales': ['English', 'Welsh'],
                'Brittany': ['French', 'Breton'],
                'Flanders': ['Dutch'],
                'Wallonia': ['French']
            };

            if (regionLanguageMap[location.region]) {
                const regionLanguages = regionLanguageMap[location.region];
                regionLanguages.forEach(lang => {
                    if (!languages.includes(lang)) {
                        languages.unshift(lang); // Prioritize regional languages
                    }
                });
            }
        }

        return languages;
    }

    /**
     * Clear cached location (useful for testing or when user changes location)
     */
    clearCache(): void {
        this.cachedLocation = null;
        this.detectionPromise = null;
    }
}

// Create a singleton instance
export const locationService = new LocationService();
