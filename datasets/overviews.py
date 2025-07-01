

# --- Imports ---
import os
import sys
import time
import datetime
import threading
import pathlib
from enum import Enum, auto
from typing import Dict
from queue import Queue, Empty
import requests as rq
import pandas as pd

# --- Config ---

MAX_RETRIES = 2
MAX_WORKERS = 10
BATCH_SIZE = 10
BATCH_INTERVAL = 2
CHECKPOINT_EVERY = 50
RATE_LIMIT_WAIT = 10  # seconds to wait on 429
PARQUET_INPUT_PATH = "dist/merged_raw.parquet"
PARQUET_CHECKPOINT_PATH = "dist/with_overviews_checkpoint.parquet"
PARQUET_OUTPUT_PATH = "dist/with_overviews.parquet"
MOVING_AVG_WINDOW = 10 # Number of batches to consider for moving average ETA

# --- Globals ---

retry_map: Dict[str, int] = {}
id_queue = Queue()
g_429_lock = threading.Lock()
g_429_until = 0
df_lock = threading.Lock()
semaphore = threading.Semaphore(MAX_WORKERS)
batch_counter = 0


# --- Utility Functions ---
def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Source data shape: {df.shape}")
    return df


# --- TMDB API Setup ---
TMDB_TOKEN = os.environ.get('TMDB_API_TOKEN', '')
if not TMDB_TOKEN:
    raise RuntimeError("TMDB_API_TOKEN environment variable is not set!")

tmdb_auth_headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_TOKEN}"
}
tmdb_req_params = {
    "external_source": "imdb_id"
}
class OverviewFetchError(Enum):
    REQUEST_FAILED = auto()
    JSON_DECODE_ERROR = auto()
    NO_DATA = auto()
    NO_OVERVIEW = auto()
    RATELIMITED = auto()
def get_overview(id: str) -> str | OverviewFetchError:
    """Fetch the overview for a given movie ID from the TMDb API using IMDb IDs."""
    res = rq.get(f"https://api.themoviedb.org/3/find/{id}", headers=tmdb_auth_headers, params=tmdb_req_params)
    if res.status_code == 429:  # Too Many Requests
        global g_429_until
        with g_429_lock:
            g_429_until = time.time() + RATE_LIMIT_WAIT  # Wait for RATE_LIMIT_WAIT seconds before retrying
        return OverviewFetchError.RATELIMITED
    elif res.status_code != 200:
        return OverviewFetchError.REQUEST_FAILED
    try:
        data = res.json()
    except Exception:
        return OverviewFetchError.JSON_DECODE_ERROR
    if not data or not isinstance(data, dict):
        return OverviewFetchError.NO_DATA
    movie_results = data.get('movie_results')
    if movie_results and isinstance(movie_results, list) and len(movie_results) > 0:
        overview = movie_results[0].get('overview')
        if overview is not None:
            return overview
        else:
            return OverviewFetchError.NO_OVERVIEW
    else:
        return OverviewFetchError.NO_OVERVIEW



def queue_ids(df: pd.DataFrame) -> None:
    """Queue up IDs from the DataFrame."""
    for id in df['tconst'].unique():
        id_queue.put(id)
        retry_map[id] = 0

def worker(id: str) -> None:
    """Thread worker: fetch and update overview for a single ID."""
    global g_429_lock, g_429_until, source_df
    try:
        with g_429_lock:
            now = time.time()
            if g_429_until > now:
                time.sleep(g_429_until - now)
        res = get_overview(id)
        if isinstance(res, OverviewFetchError):
            retryable = (
                OverviewFetchError.REQUEST_FAILED,
                OverviewFetchError.JSON_DECODE_ERROR,
                OverviewFetchError.RATELIMITED,
            )
            if res in retryable:
                print(f"{res.name.replace('_', ' ').title()} for {id}, retrying...")
                retry_map[id] += 1
                if retry_map[id] <= MAX_RETRIES:
                    id_queue.put(id)
            elif res == OverviewFetchError.NO_DATA:
                print(f"No data for {id}, skipping...")
            elif res == OverviewFetchError.NO_OVERVIEW:
                print(f"No overview for {id}, skipping...")
        else:
            with df_lock:
                source_df.loc[source_df['tconst'] == id, 'overview'] = res
                print(f"Overview for {id} updated successfully.")
    except Exception as e:
        print(f"[ERROR] Unexpected exception in worker for id {id}: {e}")
        retry_map[id] += 1
        if retry_map[id] <= MAX_RETRIES:
            print(f"[ERROR] Re-queuing {id} after crash (retry {retry_map[id]})")
            id_queue.put(id)
        else:
            print(f"[ERROR] Max retries exceeded for {id} after crash. Skipping.")
    finally:
        try:
            id_queue.task_done()
        except Exception:
            pass
        


def process_batch() -> None:
    """Process a batch of IDs using threads, respecting MAX_WORKERS."""
def process_batch() -> list[threading.Thread]:
    batch: list[str] = []
    for _ in range(BATCH_SIZE):
        try:
            batch.append(id_queue.get_nowait())
        except Empty:
            break
    threads: list[threading.Thread] = []
    if batch:
        print(f"Processing batch: {batch}")

        def thread_wrapper(id: str) -> None:
            with semaphore:
                worker(id)

        for id in batch:
            thread = threading.Thread(target=thread_wrapper, args=(id,))
            thread.start()
            threads.append(thread)
        # Use a timeout to prevent deadlocks in case a thread hangs
        for thread in threads:
            thread.join(timeout=BATCH_INTERVAL)
    return threads



# --- Main Execution ---


import pathlib

# --- Resume from checkpoint if available ---
checkpoint_path = pathlib.Path(PARQUET_CHECKPOINT_PATH)
if checkpoint_path.exists():
    print(f"[WARNING] Checkpoint file found at {PARQUET_CHECKPOINT_PATH}. Resuming from checkpoint.")
    source_df = load(PARQUET_CHECKPOINT_PATH)
else:
    source_df = load(PARQUET_INPUT_PATH)
    if 'overview' not in source_df.columns:
        source_df['overview'] = ""

# Only queue IDs that do not have an overview yet
ids_to_queue = source_df[source_df['overview'].isnull() | (source_df['overview'] == "")]['tconst'].unique()
for id in ids_to_queue:
    id_queue.put(id)
    retry_map[id] = 0
print(f"Queued {id_queue.qsize()} IDs for processing.")
total_ids = id_queue.qsize()
start_time = time.time()
batch_times: list[float] = []
try:
    last_batch_end: float = time.time()
    last_threads: list[threading.Thread] = []
    while not id_queue.empty():
        batch_start: float = last_batch_end
        last_threads = process_batch()
        batch_counter += 1
        processed_ids: int = total_ids - id_queue.qsize()
        elapsed: float = time.time() - start_time
        remaining: int = id_queue.qsize()
        # Subtract thread join time from sleep interval, but don't go below 0
        batch_processing_time: float = time.time() - batch_start
        sleep_time: float = max(0, BATCH_INTERVAL - batch_processing_time)
        time.sleep(sleep_time)
        last_batch_end = time.time()
        full_batch_time: float = last_batch_end - batch_start
        batch_times.append(full_batch_time)
        if len(batch_times) > MOVING_AVG_WINDOW:
            batch_times.pop(0)
        avg_recent_batch_time: float = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_batch_time_str: str = f"{avg_recent_batch_time:.2f}s/batch"
        # Omit ETA until at least MOVING_AVG_WINDOW batches are done
        if batch_counter >= MOVING_AVG_WINDOW and avg_recent_batch_time > 0:
            eta: float = remaining * avg_recent_batch_time / BATCH_SIZE
            eta_str: str = str(datetime.timedelta(seconds=int(eta)))
            eta_part: str = f" | ETA: {eta_str}"
        else:
            eta_part = ""
        print(f"Progress: {processed_ids}/{total_ids} processed ({(processed_ids/total_ids)*100:.2f}%) | Remaining: {remaining}{eta_part} | Avg batch: {avg_batch_time_str}")
        if batch_counter % CHECKPOINT_EVERY == 0:
            with df_lock:
                print(f"Checkpoint: saving progress after {batch_counter} batches...")
                source_df.to_parquet(PARQUET_CHECKPOINT_PATH, index=False)
except KeyboardInterrupt:
    print("\nInterrupted! Waiting for all threads to finish...")
    for t in last_threads:
        t.join()
    sys.stdout.flush()
    print("Saving checkpoint before exit...")
    with df_lock:
        source_df.to_parquet(PARQUET_CHECKPOINT_PATH, index=False)
    sys.stdout.flush()
    print("Checkpoint saved. Exiting.")
    sys.stdout.flush()
    exit(0)
print("All batches processed.")
source_df.to_parquet(PARQUET_OUTPUT_PATH, index=False)