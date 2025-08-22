# src/correlations/FindCorrelations.py

from __future__ import annotations

import argparse
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from tqdm import tqdm

# --- Add src directory to path to allow for absolute imports ---
try:
    src_path = Path(__file__).parent.parent.resolve()
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    from google_trends.TrendFetcher import TrendFetcher
except (ImportError, NameError):
    print("Could not import TrendFetcher. Make sure this script is in the correct directory")
    print("and that the google_trends module is available in the 'src' directory.")
    sys.exit(1)

# --- Global Variables & Constants ---
CONSOLE = Console()
TRENDS_TIMEFRAME = "2020-01-01 2025-08-01"


class CorrelationFinder:
    """
    A class to find correlations between Google Trends data and stock market data.
    """

    def __init__(self, stock_dir: str, keywords_file: str, cache_dir: str, results_file: str,
                 project_root: Path):
        self.stock_dir = Path(stock_dir)
        self.keywords_file = Path(keywords_file)
        self.cache_dir = Path(cache_dir)
        self.results_file = Path(results_file)
        self.project_root = project_root
        self.keyword_df: Optional[pd.DataFrame] = None
        self.stock_data: Dict[str, pd.DataFrame] = {}

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        CONSOLE.log("[bold green]CorrelationFinder initialized.[/bold green]")

    def _load_keywords(self):
        """Loads keywords and their categories from the specified CSV file."""
        CONSOLE.log(f"Loading keywords from [cyan]{self.keywords_file}[/cyan]...")
        if not self.keywords_file.exists():
            raise FileNotFoundError(f"Keywords file not found at {self.keywords_file}")

        self.keyword_df = pd.read_csv(self.keywords_file)
        if 'keyword' not in self.keyword_df.columns or 'category' not in self.keyword_df.columns:
            raise ValueError("Keywords CSV must contain 'keyword' and 'category' columns.")

        self.keyword_df.dropna(subset=['keyword'], inplace=True)
        self.keyword_df['keyword'] = self.keyword_df['keyword'].astype(str)

        CONSOLE.log(f"Loaded [bold yellow]{len(self.keyword_df)}[/bold yellow] keywords across "
                    f"[bold yellow]{self.keyword_df['category'].nunique()}[/bold yellow] categories.")

    def _load_and_preprocess_stock_data(self):
        """
        Loads all stock CSVs, intelligently handles the two specific header formats,
        and preprocesses them.
        """
        CONSOLE.log(f"Loading and preprocessing stock data from [cyan]{self.stock_dir}[/cyan]...")
        for file_path in self.stock_dir.glob("*.csv"):
            stock_name = file_path.stem

            if stock_name == "company_info":
                continue

            try:
                # --- FINAL: Robust CSV loading logic based on explicit format descriptions ---
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()

                if first_line.startswith('Date,'):
                    # Format 1 (Simple): Headers are on the first line.
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                elif first_line.startswith('Price,'):
                    # Format 2 (Complex): Multi-line header.
                    # The actual headers are on the first line. Skip the next two metadata lines.
                    df = pd.read_csv(file_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
                    df.index.name = 'Date'  # Rename the index column from 'Price' to 'Date'
                else:
                    CONSOLE.log(f"[yellow]Skipping {stock_name}: Unrecognized CSV header format.[/yellow]")
                    continue

                # The column names in Format 2 are capitalized, so we standardize them
                df.columns = [col.capitalize() for col in df.columns]

                if 'Close' not in df.columns or 'Volume' not in df.columns:
                    CONSOLE.log(
                        f"[yellow]Skipping {stock_name}: Missing 'Close' or 'Volume' column after loading.[/yellow]")
                    continue

                # --- Preprocessing: Calculate Percentage Returns ---
                df['Close_Returns'] = df['Close'].pct_change()
                df['Volume_Returns'] = df['Volume'].pct_change()
                df = df.dropna()
                self.stock_data[stock_name] = df
            except Exception as e:
                CONSOLE.log(f"[bold red]Error processing {stock_name}: {e}[/bold red]")

        CONSOLE.log(
            f"Successfully processed [bold yellow]{len(self.stock_data)}[/bold yellow] stock/basket files.")

    def fetch_and_cache_trends(self, keywords_to_fetch: List[str]):
        """
        Fetches Google Trends data, adding a random delay and periodic breaks to avoid rate-limiting.
        """
        CONSOLE.log(f"Starting Google Trends data fetch for {len(keywords_to_fetch)} keywords...")
        CONSOLE.log(f"Data will be cached in [cyan]{self.cache_dir}[/cyan].")

        proxies_path = str(self.project_root / 'proxies.txt')
        request_counter = 0

        for keyword in tqdm(keywords_to_fetch, desc="Fetching Trends"):
            # --- RATE-LIMITING FIX 1: Take a long break every 50 requests ---
            if request_counter > 0 and request_counter % 50 == 0:
                break_time = random.uniform(30, 60)
                CONSOLE.log(
                    f"\n[bold magenta]Taking a {break_time:.0f}-second break to appear more human...[/bold magenta]")
                time.sleep(break_time)

            # --- RATE-LIMITING FIX 2: Use a longer, more random delay for every request ---
            time.sleep(random.uniform(2, 5))

            safe_filename = "".join(c for c in keyword if c.isalnum() or c in (' ', '_')).rstrip().replace(
                ' ', '_') + ".csv"
            cache_path = self.cache_dir / safe_filename

            if cache_path.exists():
                continue

            try:
                fetcher = TrendFetcher(
                    keywords=[keyword],
                    timeframe=TRENDS_TIMEFRAME,
                    proxies_path=proxies_path
                )
                trends_df = fetcher.fetch()
                request_counter += 1  # Increment counter only on a successful-looking attempt

                if trends_df is not None and not trends_df.empty:
                    trends_df.to_csv(cache_path)
                else:
                    cache_path.touch()
            except Exception as e:
                # --- RATE-LIMITING FIX 3: Don't crash on failure, just log and continue ---
                CONSOLE.log(
                    f"\n[bold red]Failed to fetch '{keyword}' after all retries. Skipping. Error: {e}[/bold red]")
                continue

        CONSOLE.log("[bold green]Google Trends data fetching complete.[/bold green]")

    @staticmethod
    def _calculate_cross_correlation(series1: pd.Series, series2: pd.Series, max_lag: int = 7) -> Tuple[
        int, float]:
        """Calculates the cross-correlation between two series for a given lag range."""
        best_lag = 0
        max_corr = 0.0

        for lag in range(-max_lag, max_lag + 1):
            corr = series1.corr(series2.shift(lag))
            if pd.notna(corr) and abs(corr) > abs(max_corr):
                max_corr = corr
                best_lag = lag

        return best_lag, max_corr

    def run_analysis(self, target_stocks: List[str], target_keywords_df: pd.DataFrame,
                     max_lag: int = 7) -> pd.DataFrame:
        """Runs the correlation analysis for a targeted set of stocks and keywords."""
        CONSOLE.log(f"Starting cross-correlation analysis for [yellow]{len(target_stocks)}[/yellow] stocks "
                    f"against [yellow]{len(target_keywords_df)}[/yellow] keywords...")
        results = []

        stock_progress = tqdm(target_stocks, desc="Analyzing Stocks", total=len(target_stocks))

        for stock_name in stock_progress:
            stock_progress.set_postfix_str(stock_name)
            stock_df = self.stock_data.get(stock_name)
            if stock_df is None:
                continue

            for _, row in target_keywords_df.iterrows():
                keyword, category = row['keyword'], row['category']
                safe_filename = "".join(
                    c for c in keyword if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_') + ".csv"
                cache_path = self.cache_dir / safe_filename

                if not cache_path.exists():
                    continue

                try:
                    trend_df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
                    if trend_df.empty or keyword not in trend_df.columns:
                        continue

                    trend_series = trend_df[keyword]
                    aligned_stock, aligned_trend = stock_df.align(trend_series, join='inner', axis=0)

                    if aligned_stock.empty or len(aligned_stock) < 30:
                        continue

                    for metric in ['Close_Returns', 'Volume_Returns']:
                        best_lag, corr = self._calculate_cross_correlation(
                            aligned_stock[metric], aligned_trend, max_lag
                        )
                        if corr != 0.0:
                            results.append({
                                'Stock': stock_name,
                                'Keyword': keyword,
                                'Keyword_Category': category,
                                'Metric': metric,
                                'Best_Lag_Days': best_lag,
                                'Correlation': corr
                            })
                except Exception:
                    pass

        CONSOLE.log("[bold green]Analysis complete.[/bold green]")
        if not results:
            CONSOLE.log("[bold red]No correlation results were generated.[/bold red]")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df['Abs_Correlation'] = results_df['Correlation'].abs()
        results_df = results_df.sort_values(by='Abs_Correlation', ascending=False).drop(
            columns=['Abs_Correlation'])
        return results_df

    def execute_pipeline(self, target_categories: Optional[List[str]], target_stocks: Optional[List[str]]):
        """Runs the entire pipeline with optional targeting."""
        try:
            self._load_keywords()
            self._load_and_preprocess_stock_data()

            if self.keyword_df is None:
                raise ValueError("Keywords were not loaded correctly.")

            keywords_to_run = self.keyword_df
            if target_categories:
                keywords_to_run = self.keyword_df[self.keyword_df['category'].isin(target_categories)]
                CONSOLE.log(
                    f"Filtered to [yellow]{len(keywords_to_run)}[/yellow] keywords in specified categories.")

            stocks_to_run = list(self.stock_data.keys())
            if target_stocks:
                stocks_to_run = [s for s in self.stock_data if any(t in s for t in target_stocks)]
                CONSOLE.log(
                    f"Filtered to [yellow]{len(stocks_to_run)}[/yellow] stocks matching specified names.")

            if keywords_to_run.empty or not stocks_to_run:
                CONSOLE.log("[bold red]No keywords or stocks to analyze after filtering. Exiting.[/bold red]")
                return

            self.fetch_and_cache_trends(keywords_to_run['keyword'].tolist())

            results_df = self.run_analysis(target_stocks=stocks_to_run, target_keywords_df=keywords_to_run)

            if not results_df.empty:
                CONSOLE.log(f"Saving top results to [cyan]{self.results_file}[/cyan]...")
                results_df.to_csv(self.results_file, index=False)
                CONSOLE.log("[bold green]Pipeline executed successfully![/bold green]")
                CONSOLE.log("Top 10 Results:")
                print(results_df.head(10).to_string())
            else:
                CONSOLE.log("[bold yellow]Pipeline finished, but no results were saved.[/bold yellow]")

        except Exception as e:
            CONSOLE.log(f"[bold red]An error occurred during pipeline execution: {e}[/bold red]")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the correlation analysis from the command line."""
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]
    except NameError:
        project_root = Path.cwd()

    parser = argparse.ArgumentParser(
        description="Find correlations between Google Trends and Stock Market data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--stock-dir', default=str(project_root / 'data' / 'Market'),
                        help="Directory with stock CSVs.")
    parser.add_argument('--keywords-file', default=str(project_root / 'keywords.csv'),
                        help="Path to the keywords CSV file.")
    parser.add_argument('--cache-dir', default=str(project_root / 'data' / 'google_trends'),
                        help="Directory for cached Google Trends data.")
    parser.add_argument('--results-file', default=str(project_root / 'correlation_results.csv'),
                        help="Output file for results.")

    parser.add_argument(
        '--target-categories',
        nargs='+',
        default=None,
        help="Space-separated list of keyword categories to analyze (e.g., geopolitics_israel health_lifestyle)."
    )
    parser.add_argument(
        '--target-stocks',
        nargs='+',
        default=None,
        help="Space-separated list of stock/basket names to analyze (e.g., SupportIsrael Google TAN)."
    )
    args = parser.parse_args()

    finder = CorrelationFinder(
        stock_dir=args.stock_dir,
        keywords_file=args.keywords_file,
        cache_dir=args.cache_dir,
        results_file=args.results_file,
        project_root=project_root
    )
    finder.execute_pipeline(target_categories=args.target_categories, target_stocks=args.target_stocks)


if __name__ == "__main__":
    main()
