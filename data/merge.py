import pandas as pd
import numpy as np

def load_basics(path: str) -> pd.DataFrame:
    basics = pd.read_csv(
        path,
        sep='\t',
        na_values=['\\N'],
        keep_default_na=False,
        engine='c',
        low_memory=False
    )
    print(f"Raw basics shape: {basics.shape}")
    return basics

def filter_movies(basics: pd.DataFrame) -> pd.DataFrame:
    movies = basics[basics['titleType'].isin(['movie', 'tvMovie'])].copy()
    print(f"Filtered movies shape: {movies.shape}")
    return movies

def load_ratings(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def merge_ratings(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    rated_movies = movies.merge(ratings, on='tconst', how='left')
    print(f"After merging ratings: {rated_movies.shape}")
    return rated_movies

def print_stats(rated_movies: pd.DataFrame):
    percent_with_ratings = rated_movies['averageRating'].notna().mean()
    print(f"% of movies with ratings: {percent_with_ratings:.2%}")
    # Expected: ~45% should have ratings
    print(f"Rated movies: {len(rated_movies):,}")
    print(f"Avg rating: {rated_movies['averageRating'].mean():.1f}")
    print(f"Median votes: {rated_movies['numVotes'].median():,}")
    print(f"Year range: {rated_movies['startYear'].min()} - {rated_movies['startYear'].max()}")

    rated_movies = rated_movies.dropna(subset=['averageRating'])
    return rated_movies

def filter_quality_stats(rated_movies: pd.DataFrame):
    quality_movies = rated_movies[
        (rated_movies['numVotes'] >= 100) &      # Atleast 100 votes
        (rated_movies['startYear'] >= 1970) &    # Modern enough
        (rated_movies['averageRating'] >= 1.0) & # Reasonable rating
        (~rated_movies['genres'].isna())         # Genres should not be N/A
    ].copy()
    quality_movies = boost_niches(quality_movies)
    return quality_movies

def boost_niches(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ratingWeight column to boost niche (less-voted) movies with a tapered boost."""
    p10 = np.percentile(df['numVotes'], 10)
    p20 = np.percentile(df['numVotes'], 20)
    def tapered_weight(votes):
        if votes <= p10:
            return 1.0
        elif votes <= p20:
            return (p20 - votes) / (p20 - p10)
        else:
            return 0.0
    df = df.copy()
    df['ratingWeight'] = df['numVotes'].apply(tapered_weight)
    return df

def print_quality_stats(quality_movies: pd.DataFrame):
    print(f"Quality movies: {len(quality_movies):,}")
    print(f"New median votes: {quality_movies['numVotes'].median():,}")
    print(f"New avg rating: {quality_movies['averageRating'].mean():.1f}")
    print(f"New year range: {quality_movies['startYear'].min()} - {quality_movies['startYear'].max()}")
    boosted = quality_movies[quality_movies['ratingWeight'] > 0]
    percent_boosted = len(boosted) / len(quality_movies) if len(quality_movies) > 0 else 0
    print(f"Boosted movies (weight > 0): {len(boosted):,} ({percent_boosted:.2%})")
    return quality_movies



basics = load_basics('title.basics.tsv.gz')
movies = filter_movies(basics)
ratings = load_ratings('title.ratings.tsv.gz')
rated_movies = merge_ratings(movies, ratings)
rated_movies = print_stats(rated_movies)
print("-------------------------")
quality_movies = filter_quality_stats(rated_movies)
quality_movies = boost_niches(quality_movies)
quality_movies = print_quality_stats(quality_movies)
