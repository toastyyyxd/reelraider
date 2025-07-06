import polars as pl
from polars import LazyFrame
from pathlib import Path
from datasets.utils import logger, read_csv_file, write_parquet_file

class TitleIdFilter:
    def __init__(
        self,
        basics_df: LazyFrame,
        ratings_df: LazyFrame,
        min_year: int = 1970,
        min_votes: int = 2500,
        min_rating: float = 1.0
    ):
        self.basics_df = basics_df
        self.ratings_df = ratings_df
        self.min_year = min_year
        self.min_votes = min_votes
        self.min_rating = min_rating
        self.filtered_df = None
    
    def apply(self):
        logger.info("Filtering dataset into IDs only to prevent derivation of copyrighted data.")
        self.filtered_df = self.basics_df\
            .select(["tconst", "startYear", "titleType"])\
            .filter([
                pl.col("titleType").is_in(["movie", "tvMovie"]),
                pl.col("startYear") >= self.min_year
            ])\
            .join(
                self.ratings_df.select(["tconst", "averageRating", "numVotes"]),
                on="tconst",
                how="left"
            )\
            .filter([
                pl.col("numVotes").is_not_null(),
                pl.col("averageRating").is_not_null(),
                pl.col("numVotes") >= self.min_votes,
                pl.col("averageRating") >= self.min_rating
            ])\
            .with_columns(
                pl.col("tconst").str.slice(2).cast(pl.UInt32).alias("tid") # Remove 'tt' prefix and cast to UInt32
            )\
            .select(["tid"])
        self.filtered_df = self.filtered_df.collect()
        logger.info("Filtering complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter IMDB dataset and save as Parquet.")
    parser.add_argument('--basics', type=str, default="datasets/raw/title.basics.tsv.gz", help="Path to title.basics.tsv.gz")
    parser.add_argument('--ratings', type=str, default="datasets/raw/title.ratings.tsv.gz", help="Path to title.ratings.tsv.gz")
    parser.add_argument('--output', type=str, default="datasets/dist/ids.parquet", help="Output Parquet file path")
    parser.add_argument('--min-year', type=int, default=1970, help="Minimum start year (inclusive)")
    parser.add_argument('--min-votes', type=int, default=2500, help="Minimum number of votes (inclusive)")
    parser.add_argument('--min-rating', type=float, default=1.0, help="Minimum average rating (inclusive)")

    args = parser.parse_args()

    basics_path = Path(args.basics)
    ratings_path = Path(args.ratings)
    output_path = Path(args.output)

    basics_df = read_csv_file(basics_path, lazy=True)
    ratings_df = read_csv_file(ratings_path, lazy=True)

    id_filter = TitleIdFilter(
        basics_df,
        ratings_df,
        min_year=args.min_year,
        min_votes=args.min_votes,
        min_rating=args.min_rating
    )
    id_filter.apply()

    write_parquet_file(id_filter.filtered_df, output_path)
    logger.info(f"Filtered IDs saved to {output_path}")