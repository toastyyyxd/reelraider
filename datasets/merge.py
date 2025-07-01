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
    print(f"Filtered shape: {movies.shape}")
    return movies

def load_ratings(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def merge_ratings(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    rated_movies = movies.merge(ratings, on='tconst', how='left')
    print(f"After merging ratings: {rated_movies.shape}")
    return rated_movies

def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_before = len(df)
    # Ensure string columns
    for col in ['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'genres']:
        df[col] = df[col].astype(str)
    # isAdult as boolean (0/1)
    df['isAdult'] = df['isAdult'].fillna(0).astype(int).astype(bool)
    # startYear and endYear as nullable Int64
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce').astype('Int64')
    df['endYear'] = pd.to_numeric(df['endYear'], errors='coerce').astype('Int64')
    # runtimeMinutes as nullable Int64
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce').astype('Int64')
    # numVotes as int
    df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce').astype('Int64')
    # Drop rows with missing required fields
    required = ['tconst', 'primaryTitle', 'startYear', 'runtimeMinutes', 'genres', 'numVotes', 'averageRating']
    df = df.dropna(subset=required)
    n_after = len(df)
    percent_dropped = (n_before - n_after) / n_before if n_before > 0 else 0
    print(f"Dropped {n_before - n_after:,} rows ({percent_dropped:.2%}) due to invalid or missing ratings or data.")
    return df

def print_stats(rated_movies: pd.DataFrame):
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

basics = load_basics('raw/title.basics.tsv.gz')
movies = filter_movies(basics)
ratings = load_ratings('raw/title.ratings.tsv.gz')
rated_movies = merge_ratings(movies, ratings)
rated_movies = clean_types(rated_movies)
rated_movies = print_stats(rated_movies)
print("-------------------------")
quality_movies = filter_quality_stats(rated_movies)
quality_movies = boost_niches(quality_movies)
quality_movies = print_quality_stats(quality_movies)
print("-------------------------")
print("Saving final shape:", quality_movies.shape)
quality_movies.to_parquet('dist/merged_raw.parquet',
    index=False,
    compression='snappy',
    engine='pyarrow'
)
print("Saved to 'dist/merged_raw.parquet'")