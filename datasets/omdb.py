
import os
import re
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import polars as pl
from polars import DataFrame
import numpy as np
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm as tqdm_asyncio

from datasets.utils import logger, read_parquet_file, write_parquet_file, CheckpointManager


class OmdbAggregator:
    """
    Aggregates OMDB data for a set of IMDb IDs using the OMDB API.
    Optionally enforces a self-imposed rate limit (requests per second).
    """

    def __init__(self, ids_df: DataFrame, max_concurrent: int, api_key: Optional[str] = None, max_rps: Optional[float] = None, checkpoint_path: Optional[str] = None, checkpoint_every: Optional[int] = None):
        self.ids_df: DataFrame = ids_df
        self.omdb_results: List[Dict[str, Any]] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.api_key: Optional[str] = api_key or os.environ.get("OMDB_API_KEY")
        if not self.api_key:
            raise ValueError("OMDB_API_KEY environment variable not set.")
        self.max_rps = max_rps
        self._ratelimit_lock = asyncio.Lock() if max_rps else None
        self._ratelimit_last = 0.0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_manager = None
        if checkpoint_path:
            self.checkpoint_manager = CheckpointManager(Path(checkpoint_path))
            # Try to load checkpoint if exists
            if self.checkpoint_manager.exists():
                logger.info(f"Loading OMDB checkpoint from {checkpoint_path}")
                df = self.checkpoint_manager.load()
                self.omdb_results = df.to_dicts()
                # Remove already-fetched tids from ids_df
                done_tids = set(row["tid"] for row in self.omdb_results)
                self.ids_df = self.ids_df.filter(~self.ids_df["tid"].is_in(list(done_tids)))

    @staticmethod
    def _split_field(value: Any, sep: str = ",") -> List[str]:
        """Split a string field by separator, return list, handle N/A and empty."""
        if not value or value in ("N/A",):
            return []
        return [item.strip() for item in str(value).split(sep) if item.strip()]


    @staticmethod
    def _parse_int(value: Any, default: int = None) -> int | None:
        """Parse integer from string, handle commas and N/A. Returns None if parsing fails or value is missing."""
        if value in (None, "", "N/A"):
            return default
        try:
            return int(str(value).replace(",", ""))
        except Exception:
            return default


    @staticmethod
    def _parse_float(value: Any, default: float = None) -> float | None:
        """Parse float from string, handle N/A. Returns None if parsing fails or value is missing."""
        if value in (None, "", "N/A"):
            return default
        try:
            return float(str(value))
        except Exception:
            return default


    @staticmethod
    def _parse_duration(value: Any, default: int = None) -> int | None:
        """
        Parse a duration string like '173 min', '2h 13min', '1h', '45min', '90', etc. to an integer (minutes).
        Returns None if parsing fails or value is missing.
        """
        if value in (None, "", "N/A"):
            return default
        s = str(value).strip().lower()
        hours = 0
        minutes = 0
        h_match = re.search(r'(\d+)\s*h', s)
        m_match = re.search(r'(\d+)\s*m', s)
        if h_match:
            hours = int(h_match.group(1))
        if m_match:
            minutes = int(m_match.group(1))
        if hours or minutes:
            return hours * 60 + minutes
        # Fallback: just extract the first integer (assume it's minutes)
        match = re.search(r'(\d+)', s)
        if match:
            return int(match.group(1))
        return default

    @staticmethod
    def _parse_string(value: Any, default: str = None) -> str | None:
        """Parse a string value, return None if empty or N/A."""
        if value in (None, "", "N/A"):
            return default
        return str(value).strip() if isinstance(value, str) else default

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def fetch_one(self, tid: np.uint32, client: httpx.AsyncClient):
        """Fetch OMDB data for a single IMDb ID, respecting self-imposed rate limit if set."""
        imdb_id = f"tt{str(tid).zfill(7)}"
        async with self.semaphore:
            # Self-imposed rate limit (if enabled)
            if self.max_rps:
                async with self._ratelimit_lock:
                    now = time.perf_counter()
                    min_interval = 1.0 / self.max_rps
                    elapsed = now - self._ratelimit_last
                    if elapsed < min_interval:
                        await asyncio.sleep(min_interval - elapsed)
                    self._ratelimit_last = time.perf_counter()

            logger.debug(f"Fetching OMDB data for IMDb ID: {imdb_id} (tid={tid})")
            try:
                response = await client.get("/", params={"i": imdb_id, "plot": "full", "r": "json"})
                logger.debug(f"Received response for {imdb_id}: {response.status_code}")
                data = response.json()
            except Exception as e:
                logger.error(f"HTTP error for {imdb_id} (tid={tid}): {e}")
                raise

            if data.get("Response") == "True":
                try:
                    self.omdb_results.append({
                        "tid": tid,
                        "title": self._parse_string(data.get("Title")),
                        "year": self._parse_int(data.get("Year")),
                        "rated": self._parse_string(data.get("Rated")),
                        "runtime": self._parse_duration(data.get("Runtime")),
                        "genre": self._split_field(data.get("Genre")),
                        "director": self._split_field(data.get("Director")),
                        "actors": self._split_field(data.get("Actors")),
                        "plot": self._parse_string(data.get("Plot")),
                        "language": self._split_field(data.get("Language")),
                        "country": self._split_field(data.get("Country")),
                        "awards": self._split_field(data.get("Awards"), sep="&"),
                        "poster": self._parse_string(data.get("Poster")),
                        "rating": self._parse_float(data.get("imdbRating")),
                        "votes": self._parse_int(data.get("imdbVotes")),
                        "metascore": self._parse_int(data.get("Metascore")),
                    })
                    logger.debug(f"OMDB data for {imdb_id} (tid={tid}) fetched successfully.")
                except Exception as e:
                    logger.error(f"Error processing OMDB data for {imdb_id} (tid={tid}): {e}")
                    raise ValueError(f"Error processing OMDB data for {imdb_id} (tid={tid}): {e}")
            else:
                logger.error(f"OMDB API error for {imdb_id} (tid={tid}): {data.get('Error', 'Unknown error')}")
                raise ValueError(f"OMDB API error for {imdb_id} (tid={tid}): {data.get('Error', 'Unknown error')}")

    async def fetch_all(self):
        """
        Fetch OMDB data for all IMDb IDs in the DataFrame.
        Shows a tqdm progress bar with 3-digit decimal percentages and req/s indicator.
        Saves a checkpoint every `checkpoint_every` requests if enabled.
        Only fetches tids not already in checkpoint.
        """
        # Only fetch tids not already in checkpoint
        tids_to_fetch = self.ids_df["tid"].to_list()
        if not tids_to_fetch:
            logger.info("All requested OMDB records already fetched. Skipping fetch.")
            return
        logger.info(f"Starting to fetch {len(tids_to_fetch)} OMDB records with max concurrency {self.semaphore._value}.")
        async with httpx.AsyncClient(
            base_url="https://www.omdbapi.com/",
            params={"apikey": self.api_key}
        ) as client:
            total = len(tids_to_fetch)
            start_time = time.perf_counter()
            bar_format = '{percentage:6.3f}% |{bar}| {n_fmt}/{total_fmt} {rate_fmt} [{elapsed}<{remaining}]'
            checkpoint_every = self.checkpoint_every or 0
            checkpoint_manager = self.checkpoint_manager
            pbar = tqdm_asyncio(total=total, unit="req", ncols=90, bar_format=bar_format)
            completed = 0
            async def run_and_update(tid):
                await self.fetch_one(tid, client)
                nonlocal completed
                completed += 1
                elapsed = time.perf_counter() - start_time
                reqs_per_sec = pbar.n / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix_str(f"{reqs_per_sec:7.3f}")
                pbar.update(1)
                # Save checkpoint if needed
                if checkpoint_manager and checkpoint_every > 0 and completed % checkpoint_every == 0:
                    logger.debug(f"Checkpointing after {completed} requests...")
                    df = self.merge_results()
                    checkpoint_manager.update(df)
            try:
                await asyncio.gather(*(run_and_update(tid) for tid in tids_to_fetch))
                # Final checkpoint at end
                if checkpoint_manager and checkpoint_every > 0:
                    logger.info(f"Final checkpoint after all requests...")
                    df = self.merge_results()
                    checkpoint_manager.update(df)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Saving checkpoint before exiting...")
                if checkpoint_manager and self.omdb_results:
                    df = self.merge_results()
                    checkpoint_manager.update(df)
                    logger.info(f"Checkpoint saved with {len(self.omdb_results)} records.")
                pbar.close()
                raise
            finally:
                pbar.close()

    def merge_results(self):
        """
        Merge the fetched OMDB results into a Polars DataFrame.
        Returns a DataFrame with the aggregated data.
        """
        if self.omdb_results and isinstance(self.omdb_results[0], dict):
            schema = [
                ("tid", pl.UInt32),
                ("title", pl.Utf8),
                ("year", pl.UInt16),
                ("rated", pl.Utf8),
                ("runtime", pl.UInt16),
                ("genre", pl.List(pl.Categorical)),
                ("director", pl.List(pl.Categorical)),
                ("actors", pl.List(pl.Categorical)),
                ("plot", pl.Utf8),
                ("language", pl.List(pl.Categorical)),
                ("country", pl.List(pl.Categorical)),
                ("awards", pl.List(pl.Utf8)),
                ("poster", pl.Utf8),
                ("rating", pl.Float32),
                ("votes", pl.UInt32),
                ("metascore", pl.UInt16),
            ]
            return pl.DataFrame(self.omdb_results, schema=schema)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate OMDB data for IMDb IDs.")
    parser.add_argument('--ids', type=str, default="datasets/dist/ids.parquet", help="Path to Parquet file with IMDb IDs.")
    parser.add_argument('--output', type=str, default="datasets/dist/movies.parquet", help="Output Parquet file path.")
    parser.add_argument('--max-concurrent', type=int, default=64, help="Maximum concurrent requests to OMDB API.")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of records to fetch (optional).")
    parser.add_argument('--api-key', type=str, default=None, help="OMDB API key (overrides environment variable).")
    parser.add_argument('--max-rps', type=float, default=None, help="Maximum requests per second (self-imposed rate limit, optional).")
    parser.add_argument('--checkpoint-path', type=str, default="datasets/dist/movies.cp.parquet", help="Path to save OMDB checkpoint (optional).")
    parser.add_argument('--checkpoint-every', type=int, default=500, help="Save checkpoint every N requests (optional).")
    args = parser.parse_args()

    ids_path = Path(args.ids)
    output_path = Path(args.output)
    ids_df = read_parquet_file(ids_path, lazy=False)
    
    # Apply limit to the original dataset before checkpoint filtering
    if args.limit is not None:
        ids_df = ids_df.head(args.limit)

    aggregator = OmdbAggregator(
        ids_df,
        max_concurrent=args.max_concurrent,
        api_key=args.api_key,
        max_rps=args.max_rps,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every
    )
    
    try:
        asyncio.run(aggregator.fetch_all())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Saving final results...")
    
    # Convert results to DataFrame and save (even if interrupted)
    if aggregator.omdb_results:
        results_df = aggregator.merge_results()
        write_parquet_file(results_df, output_path)
        logger.info(f"OMDB aggregation complete. Results saved to {output_path} ({len(aggregator.omdb_results)} records).")
    else:
        logger.info("No results to save.")