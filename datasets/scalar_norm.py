import polars as pl
from polars import LazyFrame, DataFrame
from datasets.utils import logger

class ScalarNormalizer:
    """
    Normalizes scalar columns to a 0-1 range using min-max normalization.
    Columns normalized: year, runtime, rating, votes, metascore by default, or user-specified.
    """

    def __init__(self, df: LazyFrame, columns: list[str] | None = None):
        self.source_df: LazyFrame = df
        # Default columns if none provided
        default_cols = ["year", "runtime", "rating", "votes", "metascore"]
        self.columns: list[str] = columns if columns else default_cols
        self.stats: dict[str, tuple[float, float]] = {}

    def calc_stats(self):
        """Calculate min and max values for scalar columns."""
        logger.debug("Calculating min and max for scalar columns.")
        # build expressions for min and max of each column
        stat_exprs = []
        for col in self.columns:
            stat_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            stat_exprs.append(pl.col(col).max().alias(f"{col}_max"))
        stat_df = self.source_df.select(*stat_exprs).collect()

        # extract stats into dict
        self.stats = {}
        for col in self.columns:
            min_val = stat_df[f"{col}_min"][0]
            max_val = stat_df[f"{col}_max"][0]
            self.stats[col] = (min_val, max_val)
            logger.info(f"{col}: min={min_val}, max={max_val}")

    def normalize_columns(self):
        """Normalize scalar columns to 0-1 range."""
        if not self.stats:
            self.calc_stats()
        logger.debug("Normalizing scalar columns to 0-1 range.")
        # build normalization expressions
        norm_exprs = []
        for col in self.columns:
            min_val, max_val = self.stats[col]
            # avoid division by zero when max == min
            if max_val == min_val:
                expr = pl.lit(0.5).alias(f"sn_{col}")
            else:
                expr = (
                    (pl.col(col).cast(pl.Float64) - min_val) / (max_val - min_val)
                ).alias(f"sn_{col}")
            norm_exprs.append(expr)
        self.source_df = self.source_df.with_columns(*norm_exprs)

    def collect(self, engine: str) -> DataFrame:
        """Collects the lazy DataFrame into a regular DataFrame."""
        return self.source_df.collect(engine=engine)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from datasets.utils import read_parquet_file, write_parquet_file

    parser = argparse.ArgumentParser(description="Normalize scalar columns to 0-1 range.")
    parser.add_argument('--input', type=str, required=True, help="Path to input Parquet file.")
    parser.add_argument(
        '--columns',
        type=str,
        default="year,runtime,votes,",
        help="Comma-separated list of columns to normalize (default: year,runtime,votes)."
    )
    parser.add_argument('--output', type=str, required=True, help="Output Parquet file path.")

    args = parser.parse_args()

    source_df = read_parquet_file(Path(args.input), lazy=True)
    cols = [c.strip() for c in args.columns.split(',') if c.strip()]
    normalizer = ScalarNormalizer(source_df, cols)
    normalizer.normalize_columns()
    result_df = normalizer.collect(engine="streaming")
    write_parquet_file(result_df, Path(args.output))
    logger.info(f"Normalized columns {args.columns} written to {args.output}.")
