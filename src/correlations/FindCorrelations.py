from __future__ import annotations
import argparse
import sys
import time
import random
import warnings
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from logging import getLogger, StreamHandler, Formatter, INFO
from rich.console import Console
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


class CorrelationFinder:
    """
    A class to find and visualize correlations between Google Trends and stock market data.
    """

    def __init__(self, stock_dir: str, keywords_file: str, trends_cache_dir: str, correlation_cache_dir: str,
                 results_file: str, project_root: Path, skip_fetch: bool = False):
        self.skip_fetch = skip_fetch
        self.stock_dir = Path(stock_dir)
        self.keywords_file = Path(keywords_file)
        self.trends_cache_dir = Path(trends_cache_dir)
        self.correlation_cache_dir = Path(correlation_cache_dir)
        self.results_file = Path(results_file)
        self.project_root = project_root
        self.keyword_df: Optional[pd.DataFrame] = None
        self.stock_data: Dict[str, pd.DataFrame] = {}

        self.trends_cache_dir.mkdir(parents=True, exist_ok=True)
        self.correlation_cache_dir.mkdir(parents=True, exist_ok=True)
        CONSOLE.log("[bold green]CorrelationFinder initialized.[/bold green]")

    def _load_keywords(self):
        """Loads keywords and their categories from the specified CSV file."""
        CONSOLE.log(f"Loading keywords from [cyan]{self.keywords_file}[/cyan]...")
        self.keyword_df = pd.read_csv(self.keywords_file)
        self.keyword_df.dropna(subset=['keyword'], inplace=True)
        self.keyword_df['keyword'] = self.keyword_df['keyword'].astype(str)
        CONSOLE.log(f"Loaded [bold yellow]{len(self.keyword_df)}[/bold yellow] keywords.")

    def _load_and_preprocess_stock_data(self):
        """Loads and preprocesses all stock CSVs."""
        CONSOLE.log(f"Loading and preprocessing stock data from [cyan]{self.stock_dir}[/cyan]...")
        for file_path in self.stock_dir.glob("*.csv"):
            stock_name = file_path.stem
            if stock_name == "company_info":
                continue
            try:
                # --- FIX: Robustly parse both known CSV formats ---
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

                df.columns = [col.capitalize() for col in df.columns]

                if 'Close' in df.columns and 'Volume' in df.columns:
                    df['Close_Returns'] = df['Close'].pct_change()
                    df['Volume_Returns'] = df['Volume'].pct_change()
                    self.stock_data[stock_name] = df.dropna()
                else:
                    CONSOLE.log(
                        f"[yellow]Skipping {stock_name}: Missing 'Close' or 'Volume' column after loading.[/yellow]")
            except Exception as e:
                CONSOLE.log(f"[bold red]Error processing {stock_name}: {e}[/bold red]")
        CONSOLE.log(f"Successfully processed [bold yellow]{len(self.stock_data)}[/bold yellow] stock files.")

    def fetch_and_cache_trends(self, keywords_to_fetch: List[str]):
        """Fetches Google Trends data, using a cache and polite delays."""
        non_cached_keywords = [
            kw for kw in keywords_to_fetch
            if not (self.trends_cache_dir / (
                    "".join(c for c in kw if c.isalnum() or c in (' ', '_')).rstrip().replace(' ',
                                                                                              '_') + ".csv")).exists()
        ]
        CONSOLE.log(
            f"Found [bold green]{len(keywords_to_fetch) - len(non_cached_keywords)}[/bold green] cached trends. "
            f"Fetching [bold yellow]{len(non_cached_keywords)}[/bold yellow] new trends.")
        if not non_cached_keywords:
            return

        proxies_path = str(self.project_root / 'proxies.txt')
        req_count = 0
        consecutive_errors_count = 0
        next_break = 40
        for keyword in tqdm(non_cached_keywords, desc="Fetching New Trends"):
            if req_count > 0 and req_count % next_break == 0:
                next_break = random.randint(35, 45)
                break_time = random.uniform(45, 90)
                CONSOLE.log(
                    f"\n[bold magenta]Taking a {break_time:.0f}-second break to appear more human...[/bold magenta]")
                time.sleep(break_time)
            time.sleep(random.uniform(2, 5))
            safe_filename = "".join(c for c in keyword if c.isalnum() or c in (' ', '_')).rstrip().replace(
                ' ', '_') + ".csv"
            cache_path = self.trends_cache_dir / safe_filename
            try:
                fetcher = TrendFetcher(keywords=[keyword], timeframe=TRENDS_TIMEFRAME,
                                       proxies_path=proxies_path)
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
                if consecutive_errors_count >= 3:
                    CONSOLE.log(
                        "[bold red]Too many consecutive fetch errors (3). Terminating pipeline.[/bold red]")
                    sys.exit(1)
                continue

    @staticmethod
    def _calculate_cross_correlation(series1: pd.Series, series2: pd.Series, max_lag: int = 7) -> Tuple[
        int, float]:
        """Calculates the best cross-correlation between two series."""
        best_lag, max_corr = 0, 0.0
        if series1.std() == 0 or series2.std() == 0:
            return 0, 0.0
        for lag in range(-max_lag, max_lag + 1):
            corr = series1.corr(series2.shift(lag))
            if pd.notna(corr) and abs(corr) > abs(max_corr):
                max_corr, best_lag = corr, lag
        return best_lag, max_corr

    def run_analysis(self, target_stocks: List[str], target_keywords_df: pd.DataFrame,
                     max_lag: int = 7) -> pd.DataFrame:
        """Runs correlation analysis, using a cache for results."""
        CONSOLE.log(f"Starting analysis for [yellow]{len(target_stocks)}[/yellow] stocks...")
        results = []

        # --- Analyze cached keywords only ---
        cached_trend_files = {p.stem.replace('_', ' '): p for p in self.trends_cache_dir.glob("*.csv")}
        keywords_to_analyze_df = target_keywords_df[
            target_keywords_df['keyword'].isin(cached_trend_files.keys())]
        CONSOLE.log(
            f"Found [green]{len(cached_trend_files)}[/green] cached trends to analyze against stocks.")

        for stock_name in tqdm(target_stocks, desc="Analyzing Stocks"):
            stock_df = self.stock_data.get(stock_name)
            if stock_df is None:
                continue
            for _, row in keywords_to_analyze_df.iterrows():
                keyword, category = row['keyword'], row['category']
                for metric in ['Close_Returns', 'Volume_Returns']:
                    corr_cache_filename = f"{stock_name}_{keyword}_{metric}.json".replace(' ', '_')
                    corr_cache_path = self.correlation_cache_dir / corr_cache_filename
                    if corr_cache_path.exists():
                        with open(corr_cache_path, 'r') as f:
                            results.append(json.load(f))
                        continue
                    try:
                        trend_df = pd.read_csv(cached_trend_files[keyword], index_col='date',
                                               parse_dates=True)
                        if trend_df.empty or keyword not in trend_df.columns:
                            continue

                        trend_series = trend_df[keyword].astype(float).fillna(0)
                        aligned_stock, aligned_trend = stock_df.align(trend_series, join='inner', axis=0)
                        if len(aligned_stock) < 30:
                            continue

                        best_lag, corr = self._calculate_cross_correlation(aligned_stock[metric],
                                                                           aligned_trend, max_lag)
                        if corr != 0.0:
                            result_data = {'Stock': stock_name, 'Keyword': keyword,
                                           'Keyword_Category': category,
                                           'Metric': metric, 'Best_Lag_Days': best_lag, 'Correlation': corr}
                            results.append(result_data)
                            with open(corr_cache_path, 'w') as f:
                                json.dump(result_data, f)
                    except Exception:
                        continue

        if not results: return pd.DataFrame()
        results_df = pd.DataFrame(results)
        results_df['Abs_Correlation'] = results_df['Correlation'].abs()
        return results_df.sort_values(by='Abs_Correlation', ascending=False).drop(columns=['Abs_Correlation'])

    def plot_top_correlations(self, results_df: pd.DataFrame, top_n: int = 10):
        """Generates and saves interactive plots for the top N correlations."""
        if results_df.empty:
            return
        CONSOLE.log(f"Generating plots for the top {top_n} correlations...")
        plot_dir = self.project_root / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Get top N positive and top N negative correlations
        top_positive = results_df.nlargest(top_n, 'Correlation')
        top_negative = results_df.nsmallest(top_n, 'Correlation')
        top_results = pd.concat([top_positive, top_negative]).drop_duplicates()

        for _, row in top_results.iterrows():
            stock, keyword, metric, lag, corr = row['Stock'], row['Keyword'], row['Metric'], row[
                'Best_Lag_Days'], row['Correlation']

            stock_df = self.stock_data[stock]
            safe_kw_filename = "".join(c for c in keyword if c.isalnum() or c in (' ', '_')).rstrip().replace(
                ' ', '_') + ".csv"
            trend_df = pd.read_csv(self.trends_cache_dir / safe_kw_filename, index_col='date',
                                   parse_dates=True)
            trend_series = trend_df[keyword].astype(float).fillna(0)

            # Shift trend data by the best lag for visualization
            shifted_trend = trend_series.shift(lag)

            aligned_stock, aligned_trend = stock_df.align(shifted_trend, join='inner', axis=0)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=aligned_stock.index, y=aligned_stock[metric], name=f"{stock} {metric}"),
                secondary_y=False)
            fig.add_trace(go.Scatter(x=aligned_trend.index, y=aligned_trend,
                                     name=f"Trend: '{keyword}' (Shifted {lag}d)"), secondary_y=True)

            title = f"<b>{stock} vs '{keyword}'</b><br>Correlation = {corr:.3f} (Lag: {lag} days)"
            fig.update_layout(title_text=title, title_x=0.5)
            fig.update_yaxes(title_text=f"<b>{metric}</b>", secondary_y=False)
            fig.update_yaxes(title_text="<b>Google Trend Score</b>", secondary_y=True)

            plot_filename = f"{stock}_{keyword}_{metric}.html".replace(' ', '_')
            fig.write_html(plot_dir / plot_filename)
        CONSOLE.log(
            f"[bold green]Saved {len(top_results)} plots to the '{plot_dir.name}' directory.[/bold green]")

    def execute_pipeline(self, target_categories: Optional[List[str]], target_stocks: Optional[List[str]]):
        """Runs the entire pipeline."""
        try:
            self._load_keywords()
            self._load_and_preprocess_stock_data()
            if self.keyword_df is None: return

            keywords_to_run = self.keyword_df
            if target_categories:
                keywords_to_run = self.keyword_df[self.keyword_df['category'].isin(target_categories)]

            stocks_to_run = list(self.stock_data.keys())
            if target_stocks:
                stocks_to_run = [s for s in self.stock_data if any(t in s for t in target_stocks)]

            if not self.skip_fetch:
                self.fetch_and_cache_trends(keywords_to_run['keyword'].tolist())
            else:
                CONSOLE.log("[bold yellow]Skipping Google Trends fetching as requested.[/bold yellow]")

            results_df = self.run_analysis(stocks_to_run, keywords_to_run)

            if not results_df.empty:
                CONSOLE.log(f"Saving top results to [cyan]{self.results_file}[/cyan]...")
                results_df.to_csv(self.results_file, index=False)
                CONSOLE.log("[bold green]Pipeline executed successfully![/bold green]")
                CONSOLE.log("Top 10 Results:")
                print(results_df.head(10).to_string())
                self.plot_top_correlations(results_df)
            else:
                CONSOLE.log("[bold yellow]Pipeline finished, but no results were saved.[/bold yellow]")
        except Exception as e:
            CONSOLE.log(f"[bold red]An error occurred: {e}[/bold red]")
            import traceback
            traceback.print_exc()


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    parser = argparse.ArgumentParser(
        description="Find correlations between Google Trends and Stock Market data.")
    parser.add_argument('--stock-dir', default=str(project_root / 'data' / 'Market'),
                        help="Directory with stock CSVs.")
    parser.add_argument('--keywords-file', default=str(project_root / 'keywords.csv'),
                        help="Path to keywords CSV.")
    parser.add_argument('--trends-cache-dir', default=str(project_root / 'data' / 'google_trends'),
                        help="Cache for Google Trends data.")
    parser.add_argument('--corr-cache-dir', default=str(project_root / 'data' / 'correlations'),
                        help="Cache for correlation results.")
    parser.add_argument('--results-file', default=str(project_root / 'correlation_results.csv'),
                        help="Output file for results.")
    parser.add_argument('--target-categories', nargs='+', default=None,
                        help="Specific keyword categories to analyze.")
    parser.add_argument('--target-stocks', nargs='+', default=None, help="Specific stocks to analyze.")
    args = parser.parse_args()

    skip_fetch = False
    try:
        if input("Skip fetching new trends? (Y/N): ").strip().upper() == 'Y':
            skip_fetch = True
    except (EOFError, KeyboardInterrupt):
        CONSOLE.log("\n[yellow]No user input. Proceeding with fetching.[/yellow]")

    finder = CorrelationFinder(
        stock_dir=args.stock_dir, keywords_file=args.keywords_file, trends_cache_dir=args.trends_cache_dir,
        correlation_cache_dir=args.corr_cache_dir, results_file=args.results_file,
        project_root=project_root, skip_fetch=skip_fetch
    )
    finder.execute_pipeline(target_categories=args.target_categories, target_stocks=args.target_stocks)


if __name__ == "__main__":
    setup_logging_and_warnings()
    main()
