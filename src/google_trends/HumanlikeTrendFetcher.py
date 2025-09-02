from __future__ import annotations
import argparse
import sys
import time
import random
import warnings
import pandas as pd
from pathlib import Path
from typing import List
from logging import getLogger, StreamHandler, Formatter, INFO
from rich.console import Console
from tqdm import tqdm
from TrendFetcher import TrendFetcher



# --- Global Variables & Constants ---
CONSOLE = Console()
TRENDS_TIMEFRAME = "2020-08-02 2025-08-01"


def setup_logging_and_warnings():
    """
    Configures logging to be more readable and suppresses noisy FutureWarnings.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    log = getLogger('root')
    log.setLevel(INFO)
    handler = StreamHandler()
    formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    if not log.handlers:
        log.addHandler(handler)


def fetch_and_cache_trends(keywords_to_fetch: List[str], trends_cache_dir: Path, project_root: Path):
    """
    Fetches Google Trends data, using a cache and polite delays to avoid rate-limiting.
    """
    CONSOLE.log(f"Checking cache for {len(keywords_to_fetch)} keywords...")

    non_cached_keywords = [
        kw for kw in keywords_to_fetch
        if not (trends_cache_dir / (
                "".join(c for c in kw if c.isalnum() or c in (' ', '_')).rstrip().replace(' ',
                                                                                          '_') + ".csv")).exists()
    ]
    cached_count = len(keywords_to_fetch) - len(non_cached_keywords)
    CONSOLE.log(f"Found [bold green]{cached_count}[/bold green] cached keywords. "
                f"Fetching [bold yellow]{len(non_cached_keywords)}[/bold yellow] new keywords.")

    if not non_cached_keywords:
        CONSOLE.log("[bold green]All required Google Trends data is already cached.[/bold green]")
        return

    proxies_path = str(project_root / 'proxies.txt')
    req_count = 0
    consecutive_errors_count = 0
    next_break = 40
    for keyword in tqdm(non_cached_keywords, desc="Fetching New Trends"):
        if req_count > 0 and req_count % next_break == 0:
            next_break = random.randint(35, 45)
            break_time = random.uniform(45, 80)
            CONSOLE.log(
                f"\n[bold magenta]Taking a {break_time:.0f}-second break[/bold magenta]")
            time.sleep(break_time)
        time.sleep(random.uniform(2, 5))
        safe_filename = "".join(c for c in keyword if c.isalnum() or c in (' ', '_')).rstrip().replace(' ',
                                                                                                       '_') + ".csv"
        cache_path = trends_cache_dir / safe_filename
        try:
            fetcher = TrendFetcher(keywords=[keyword], timeframe=TRENDS_TIMEFRAME, proxies_path=proxies_path)
            trends_df = fetcher.fetch()
            req_count += 1
            if trends_df is not None and not trends_df.empty:
                trends_df.to_csv(cache_path)
            else:
                cache_path.touch()
            consecutive_errors_count = 0
        except Exception as e:
            consecutive_errors_count += 1
            CONSOLE.log(f"\n[bold red]Failed to fetch '{keyword}'. Skipping. Error: {e}[/bold red]")
            if consecutive_errors_count >= 10:
                CONSOLE.log(
                    "[bold red]Too many consecutive fetch errors (10). Terminating pipeline.[/bold red]")
                sys.exit(1)
            continue
    CONSOLE.log("[bold green]Google Trends data fetching complete.[/bold green]")


def main():
    """Main function to run the trend fetching process from the command line."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    parser = argparse.ArgumentParser(
        description="Fetch Google Trends data in a 'human-like' manner to avoid rate-limiting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--keywords-file', default=str(project_root / 'keywords.csv'),
                        help="Path to the keywords CSV file.")
    parser.add_argument('--trends-cache-dir', default=str(project_root / 'data' / 'google_trends'),
                        help="Directory to store cached Google Trends data.")
    args = parser.parse_args()

    try:
        keywords_df = pd.read_csv(args.keywords_file)
        if 'keyword' not in keywords_df.columns:
            raise ValueError("Keywords CSV must contain a 'keyword' column.")

        keywords_to_fetch = keywords_df['keyword'].dropna().astype(str).tolist()

        fetch_and_cache_trends(
            keywords_to_fetch=keywords_to_fetch,
            trends_cache_dir=Path(args.trends_cache_dir),
            project_root=project_root
        )
        CONSOLE.log("[bold blue]finished successfully.[/bold blue]")

    except FileNotFoundError:
        CONSOLE.log(f"[bold red]Error: Keywords file not found at '{args.keywords_file}'[/bold red]")
    except Exception as e:
        CONSOLE.log(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    setup_logging_and_warnings()
    main()

