import polars as pl
import pathlib
import logging
import coloredlogs
from pathlib import Path
from tqdm import tqdm
from numpy import uint32


logger = logging.getLogger("reelraider.datasets")
coloredlogs.install(
    level=logging.INFO,
    logger=logger,
    fmt='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def tid_to_tconst(tid: uint32) -> str:
    return f"tt{str(tid).zfill(7)}"
def tconst_to_tid(tconst: str) -> uint32:
    return uint32(int(tconst[2:]))  # Remove 'tt' prefix and convert to uint32

def read_csv_file(path: Path, lazy: bool = False) -> pl.LazyFrame | pl.DataFrame:
    """
    Reads a CSV or TSV file, autodetecting the separator based on file extension.
    Supports .csv, .csv.gz, .tsv, .tsv.gz.
    Returns a Polars DataFrame.
    """
    logger.debug(f"Reading CSV/TSV file from: {path}")
    suffix = ''.join(path.suffixes).lower()
    if suffix.endswith('.tsv') or suffix.endswith('.tsv.gz'):
        sep = '\t'  # TSV files use tab as separator
    else:
        sep = ','
    # IMDB TSVs do not use quotes for escaping, so disable quote parsing
    if lazy:
        df = pl.scan_csv(
            path,
            separator=sep,
            null_values=['\\N'],
            quote_char=None,
            encoding="utf8-lossy"
        )
    else:
        df = pl.read_csv(
            path,
            separator=sep,
            null_values=['\\N'],
            quote_char=None,
            encoding="utf8-lossy"
        )
    if lazy:
        logger.debug("Loaded lazily.")
    else:
        logger.debug(f"Loaded {df.height} rows.")
    return df

def read_parquet_file(path: Path, lazy: bool = False) -> pl.LazyFrame | pl.DataFrame:
    """
    Reads a Parquet file, optionally lazily.
    Returns a Polars DataFrame or LazyFrame.
    """
    logger.debug(f"Reading Parquet file from: {path}")
    if lazy:
        df = pl.scan_parquet(path)
        logger.debug("Loaded lazily.")
    else:
        df = pl.read_parquet(path)
        logger.debug(f"Loaded {df.height} rows.")
    return df

def write_parquet_file(df: pl.DataFrame, path: Path) -> None:
    """
    Write a Polars DataFrame to Parquet.
    """
    logger.debug(f"Writing DataFrame to Parquet file: {path}")
    df.write_parquet(str(path), compression='snappy')
    logger.debug(f"Wrote {df.height} rows to {path}.")

class CheckpointManager:
    def __init__(self, checkpoint_path: Path = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def set_path(self, path: Path):
        self.checkpoint_path = Path(path)

    def update(self, df):
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path not set.")
        backup_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".back")
        if self.checkpoint_path.exists():
            logger.debug(f"Backing up checkpoint to {backup_path}")
            self.checkpoint_path.replace(backup_path)
        write_parquet_file(df, self.checkpoint_path)

    def load(self) -> pl.DataFrame:
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path not set.")
        return read_parquet_file(self.checkpoint_path)

    def exists(self) -> bool:
        if not self.checkpoint_path:
            return False
        return self.checkpoint_path.exists()