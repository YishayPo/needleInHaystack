from __future__ import annotations
import sys
import warnings
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from logging import getLogger, StreamHandler, Formatter, INFO
from rich.console import Console
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Global Variables & Constants ---
CONSOLE = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOCK_DIR = PROJECT_ROOT / "data" / "Market"
KEYWORDS_FILE = PROJECT_ROOT / "keywords.csv"
TRENDS_CACHE_DIR = PROJECT_ROOT / "data" / "google_trends"
CORR_CACHE_DIR = PROJECT_ROOT / "data" / "correlations"
RESULTS_FILE = PROJECT_ROOT / "correlation_results.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"

SPARSITY_DISCARD_THRESHOLD = 0.8  # 80% -> discard completely


def setup_logging_and_warnings():
    """Configures logging and suppresses noisy warnings."""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    log = getLogger('root')
    log.setLevel(INFO)
    handler = StreamHandler()
    handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    if not log.handlers:
        log.addHandler(handler)


class CorrelationFinder:
    """Finds and visualizes correlations between pre-downloaded Google Trends and stock data."""

    def __init__(self):
        self.keyword_df: Optional[pd.DataFrame] = None
        self.stock_data_daily: Dict[str, pd.DataFrame] = {}
        self.stock_data_weekly: Dict[str, pd.DataFrame] = {}

        TRENDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CORR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        CONSOLE.log("[bold green]CorrelationFinder initialized.[/bold green]")

    def _load_keywords(self):
        CONSOLE.log(f"Loading keywords from [cyan]{KEYWORDS_FILE}[/cyan]...")
        self.keyword_df = pd.read_csv(KEYWORDS_FILE).dropna(subset=['keyword'])
        self.keyword_df['keyword'] = self.keyword_df['keyword'].astype(str)
        CONSOLE.log(f"Loaded [bold yellow]{len(self.keyword_df)}[/bold yellow] keywords.")

    def _trim_or_fix_stock_data(self, df: pd.DataFrame, stock_name: str) -> pd.DataFrame:
        """Trim from first corrupted streak >= 6 days, tolerate shorter ones."""
        df = df.copy()
        corrupted = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])
        corrupted |= (df['Volume'] == 0)

        if not corrupted.any():
            return df  # nothing wrong

        # find streaks of consecutive corrupted days
        streak_count = 0
        first_bad_day = None
        for date, is_bad in corrupted.items():
            if is_bad:
                if streak_count == 0:
                    first_bad_day = date
                streak_count += 1
                if streak_count >= 6:
                    CONSOLE.log(f"[yellow]{stock_name}: trimming data from {first_bad_day.date()} onward "
                                f"(>=6 consecutive corrupted days).[/yellow]")
                    return df.loc[:first_bad_day - pd.Timedelta(days=1)]
            else:
                streak_count = 0
                first_bad_day = None

        # if no long streak found, tolerate everything
        return df

    def _load_and_preprocess_stock_data(self):
        """Loads stock CSVs and creates daily + weekly return series."""
        CONSOLE.log(f"Loading stock data from [cyan]{STOCK_DIR}[/cyan]...")
        for file_path in STOCK_DIR.glob("*.csv"):
            stock_name = file_path.stem
            if stock_name == "company_info":
                continue
            try:
                first_line = file_path.open('r', encoding='utf-8').readline().strip()
                if first_line.startswith('Date,'):
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                elif first_line.startswith('Price,'):
                    df = pd.read_csv(file_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
                    df.index.name = 'Date'
                else:
                    continue

                df.columns = [col.capitalize() for col in df.columns]
                if 'Close' not in df.columns or 'Volume' not in df.columns:
                    continue

                # clean corrupted data
                df = self._trim_or_fix_stock_data(df, stock_name)

                df['Close_Returns'] = df['Close'].pct_change()
                df['Volume_Returns'] = df['Volume'].pct_change()
                df_daily = df.dropna()
                self.stock_data_daily[stock_name] = df_daily
                self.stock_data_weekly[stock_name] = df_daily[['Close_Returns', 'Volume_Returns']].resample(
                    'W').sum().dropna()
            except Exception as e:
                CONSOLE.log(f"[bold red]Error processing {stock_name}: {e}[/bold red]")

        CONSOLE.log(f"Processed [bold yellow]{len(self.stock_data_daily)}[/bold yellow] stock files.")

    @staticmethod
    def _preprocess_trend_series(trend_series: pd.Series) -> Tuple[pd.Series, float]:
        """Remove zeros, apply Z-score normalization, return series and sparsity ratio."""
        sparsity = (trend_series == 0).mean()
        nonzero = trend_series[trend_series != 0]
        if nonzero.empty:
            return pd.Series(dtype=float), sparsity
        # z-score
        standardized = (nonzero - nonzero.mean()) / nonzero.std(ddof=0)
        return standardized, sparsity

    @staticmethod
    def _calculate_cross_correlation(series1: pd.Series, series2: pd.Series, max_lag: int) -> Tuple[
        int, float]:
        """Find best lag correlation in weeks."""
        if series1.std() == 0 or series2.std() == 0:
            return 0, 0.0
        best_lag, max_corr = 0, 0.0
        for lag in range(-max_lag, max_lag + 1):
            corr = series1.corr(series2.shift(lag))
            if pd.notna(corr) and abs(corr) > abs(max_corr):
                best_lag, max_corr = lag, corr
        return best_lag, max_corr

    def run_analysis(self, target_stocks: List[str], target_keywords_df: pd.DataFrame) -> pd.DataFrame:
        """Run correlation analysis on weekly data with caching."""
        results = []
        self.missing_keywords: List[str] = []  # keep track of empty trend files
        cached_trend_files = {p.stem.replace('_', ' '): p for p in TRENDS_CACHE_DIR.glob("*.csv")}
        keywords_to_analyze_df = target_keywords_df[
            target_keywords_df['keyword'].isin(cached_trend_files.keys())]

        for stock_name in tqdm(target_stocks, desc="Analyzing Stocks"):
            stock_df_weekly = self.stock_data_weekly.get(stock_name)
            if stock_df_weekly is None:
                continue
            for _, row in keywords_to_analyze_df.iterrows():
                keyword, category = row['keyword'], row['category']
                for metric in ['Close_Returns', 'Volume_Returns']:
                    cache_file = CORR_CACHE_DIR / f"{stock_name}_{keyword}_{metric}.json".replace(' ', '_')
                    if cache_file.exists():
                        try:
                            results.append(json.loads(cache_file.read_text()))
                            continue
                        except json.JSONDecodeError:
                            pass
                    try:
                        try:
                            trend_df = pd.read_csv(cached_trend_files[keyword], index_col='date',
                                                   parse_dates=True)
                        except pd.errors.EmptyDataError:
                            # file completely empty
                            self.missing_keywords.append(keyword)
                            continue

                        if trend_df.empty or keyword not in trend_df.columns:
                            self.missing_keywords.append(keyword)
                            continue

                        trend_series, sparsity = self._preprocess_trend_series(
                            trend_df[keyword].astype(float).fillna(0))

                        if sparsity > SPARSITY_DISCARD_THRESHOLD:
                            result = {'Stock': stock_name, 'Keyword': keyword, 'Keyword_Category': category,
                                      'Metric': metric, 'Best_Lag_Weeks': 0, 'Correlation': 0.0}
                            results.append(result)
                            cache_file.write_text(json.dumps(result))
                            continue

                        max_lag = 1 if sparsity > (SPARSITY_DISCARD_THRESHOLD * 0.6) else 3

                        aligned_stock, aligned_trend = stock_df_weekly.align(trend_series, join='inner',
                                                                             axis=0)
                        if aligned_trend.empty:
                            continue
                        lag, corr = self._calculate_cross_correlation(aligned_stock[metric], aligned_trend,
                                                                      max_lag)
                        result = {'Stock': stock_name, 'Keyword': keyword, 'Keyword_Category': category,
                                  'Metric': metric, 'Best_Lag_Weeks': lag, 'Correlation': corr}
                        results.append(result)
                        cache_file.write_text(json.dumps(result))
                    except Exception as e:
                        CONSOLE.log(f"[red]Error processing {stock_name}-{keyword}: {e}[/red]")
                        continue

        # save missing keywords file at project root
        if hasattr(self, "missing_keywords") and self.missing_keywords:
            missing_file = PROJECT_ROOT / "missing_keywords.txt"
            with open(missing_file, "w", encoding="utf-8") as f:
                for kw in sorted(set(self.missing_keywords)):
                    f.write(f"{kw}\n")

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df['Abs_Correlation'] = df['Correlation'].abs()
        return df.sort_values(by='Abs_Correlation', ascending=False).drop(columns=['Abs_Correlation'])

    def plot_top_correlations(self, results_df: pd.DataFrame, top_n: int = 10):
        """Generate and save interactive plots for top correlations."""
        if results_df.empty:
            return
        top_results = pd.concat([
            results_df.nlargest(top_n, 'Correlation'),
            results_df.nsmallest(top_n, 'Correlation')
        ]).drop_duplicates()

        for _, row in top_results.iterrows():
            stock, keyword, metric, lag, corr = row['Stock'], row['Keyword'], row['Metric'], row[
                'Best_Lag_Weeks'], row['Correlation']
            stock_df = self.stock_data_weekly[stock]
            trend_df = pd.read_csv(TRENDS_CACHE_DIR / f"{keyword.replace(' ', '_')}.csv", index_col='date',
                                   parse_dates=True)
            trend_series, _ = self._preprocess_trend_series(trend_df[keyword].astype(float).fillna(0))
            shifted_trend = trend_series.shift(lag)
            aligned_stock, aligned_trend = stock_df.align(shifted_trend, join='inner', axis=0)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=aligned_stock.index, y=aligned_stock[metric], name=f"{stock} {metric}"),
                secondary_y=False)
            fig.add_trace(
                go.Scatter(x=aligned_trend.index, y=aligned_trend, name=f"Trend '{keyword}' (Lag {lag}w)"),
                secondary_y=True)
            fig.update_layout(title=f"{stock} vs '{keyword}' — Correlation {corr:.3f} (Lag {lag}w)",
                              title_x=0.5)

            fig.write_html(PLOTS_DIR / f"{stock}_{keyword}_{metric}.html".replace(' ', '_'))

        CONSOLE.log(f"[bold green]Saved {len(top_results)} plots to '{PLOTS_DIR}'[/bold green]")

    def execute_pipeline(self):
        """Full pipeline: load data, run analysis, save results, plot."""
        self._load_keywords()
        self._load_and_preprocess_stock_data()
        if self.keyword_df is None or not self.stock_data_daily:
            return

        results_df = self.run_analysis(list(self.stock_data_daily.keys()), self.keyword_df)
        if results_df.empty:
            CONSOLE.log("[bold yellow]No correlations found.[/bold yellow]")
            return

        results_df.to_csv(RESULTS_FILE, index=False)
        CONSOLE.log(f"[bold green]Results saved to {RESULTS_FILE}[/bold green]")
        print(results_df.head(10).to_string())
        self.plot_top_correlations(results_df)


def main():
    setup_logging_and_warnings()
    finder = CorrelationFinder()
    finder.execute_pipeline()


if __name__ == "__main__":
    main()
