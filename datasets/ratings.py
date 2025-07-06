import polars as pl
from polars import LazyFrame, DataFrame
from pathlib import Path
from datasets.utils import logger, read_parquet_file, write_parquet_file

class MergeRatings:
    def __init__(self, ratings_df: LazyFrame):
        self.source_df = ratings_df
        self.mean_votes = None
        self.mean_rating = None
        self.mean_metascore = None
        self.weighted_df = None

    def normalize_columns(self):
        logger.debug("Normalizing columns for ratings DataFrame.")
        self.source_df = self.source_df.with_columns(
            pl.when(pl.col("votes") == 0).then(None).otherwise(pl.col("votes")).cast(pl.Float64).alias("votes"), # Replace 0 votes with None, to prevent DIV BY 0
            (pl.col("rating").cast(pl.Float64) / 10).alias("n_rating"),
            (pl.col("metascore").cast(pl.Float64) / 100).alias("n_metascore"),
        )

    def calc_mean_votes(self):
        logger.debug("Calculating mean votes for ratings DataFrame.")
        self.mean_votes = self.source_df.select(pl.col("votes").mean().alias("mean_votes")).collect()["mean_votes"][0]
        logger.info(f"Mean votes calculated: {self.mean_votes}")

    def calc_mean_rating(self):
        logger.debug("Calculating mean rating for ratings DataFrame.")
        self.mean_rating = self.source_df.select(pl.col("n_rating").mean().alias("mean_rating")).collect()["mean_rating"][0]
        logger.info(f"Mean rating calculated: {self.mean_rating}")

    def calc_mean_metascore(self):
        logger.debug("Calculating mean Metascore for ratings DataFrame.")
        self.mean_metascore = self.source_df.select(pl.col("n_metascore").mean().alias("mean_metascore")).collect()["mean_metascore"][0]
        logger.info(f"Mean Metascore calculated: {self.mean_metascore}")

    def weigh_ratings(self):
        """
        votes, rating, and metascore columns may be null in some rows.
        > votes may be present but rating and metascore may not.
        > rating may be present but metascore may not.
        > metascore may be present but rating may not. etc.
        so we need to handle these cases.
        Here we will fallback accordingly for each row:
        - If metascore is present, we can incorporate it.
        - If one of rating or votes is not present, we fallback to metascore,
        - If neither rating/votes nor metascore is present, we drop the row.
        The result will take account of all data available:
        - If rating and votes are both present, we calculate the bayesian adjusted rating.
        - If only metascore is present, we use it as the rating.
        - If both bayesian adjusted rating and metascore are present, we average them.
        """
        logger.debug("Weighing ratings based on votes, rating, and metascore.")
        assert self.mean_votes is not None or self.mean_votes != 0, "Mean votes must be valid before weighing ratings."
        assert self.mean_rating is not None or self.mean_rating != 0, "Mean rating must be valid before weighing ratings."
        assert self.mean_metascore is not None or self.mean_metascore != 0, "Mean metascore must be valid before weighing ratings."
        # Calculate the bayesian rating using the formula:
        # bayes_rating = (votes / (votes + mean_votes)) * rating + (mean_votes / (votes + mean_votes)) * mean_rating
        # Use self.mean_votes and self.mean_rating as the mean values.
        self.weighted_df = (
            self.source_df
            .with_columns( # Calculate bayesian rating
                pl.when(
                    pl.col("votes").is_not_null() & pl.col("n_rating").is_not_null()
                ).then(
                    (
                    (pl.col("votes") / (pl.col("votes") + self.mean_votes)) * pl.col("n_rating") +
                    (self.mean_votes / (pl.col("votes") + self.mean_votes)) * (self.mean_rating)
                    )
                ).otherwise(None).alias("bayes_rating")
            )
            .with_columns( # Calculate final rating based on available data
                pl.when(
                    # Case 1: Both bayesian rating and metascore are available - average them
                    pl.col("bayes_rating").is_not_null() & pl.col("n_metascore").is_not_null()
                ).then(
                    (pl.col("bayes_rating") + pl.col("n_metascore")) / 2
                ).when(
                    # Case 2: Only bayesian rating is available
                    pl.col("bayes_rating").is_not_null()
                ).then(
                    pl.col("bayes_rating")
                ).when(
                    # Case 3: Only metascore is available (fallback)
                    pl.col("n_metascore").is_not_null()
                ).then(
                    pl.col("n_metascore")
                ).otherwise(
                    # Case 4: Neither available - will be filtered out
                    None
                ).alias("final_rating")
            )
            .filter(
                # Drop rows where neither rating data nor metascore is available
                pl.col("final_rating").is_not_null()
            )
            .with_columns(
                pl.when(
                    # Only calculate controversy score when both bayesian rating and metascore are available
                    pl.col("bayes_rating").is_not_null() & pl.col("n_metascore").is_not_null()
                ).then(
                    (pl.col("bayes_rating") - pl.col("n_metascore")).abs()
                    * (pl.min_horizontal(pl.col("votes"), self.mean_votes) / self.mean_votes)
                ).otherwise(None).alias("controversy_score")
            )
        )

            

    def collect(self, engine: str) -> DataFrame:
        """
        Collects the lazy DataFrame into a regular DataFrame.
        """
        if self.weighted_df is None:
            raise ValueError("Weighted DataFrame is not computed. Call weigh_ratings() first.")
        return self.weighted_df.collect(engine=engine)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process IMDB ratings dataset.")
    parser.add_argument('--ratings', type=str, default="datasets/dist/movies.parquet", help="Path to title.ratings.parquet")
    parser.add_argument('--output', type=str, default="datasets/dist/movies_processed.parquet", help="Output Parquet file path")

    args = parser.parse_args()

    source_df = read_parquet_file(Path(args.ratings), lazy=True)
    merger = MergeRatings(source_df)
    
    merger.normalize_columns()
    merger.calc_mean_votes()
    merger.calc_mean_rating()
    merger.calc_mean_metascore()
    merger.weigh_ratings()

    processed_df = merger.collect(engine="streaming")
    logger.info(f"Mean final rating: {processed_df['final_rating'].mean()}")
    
    write_parquet_file(processed_df, Path(args.output))
    logger.info(f"Processed ratings saved to {args.output}.")