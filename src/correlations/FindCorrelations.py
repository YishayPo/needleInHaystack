from __future__ import annotations
import warnings
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from logging import getLogger, StreamHandler, Formatter, INFO
from rich.console import Console
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.seterr(all="ignore")

# --- Global Variables & Constants ---
CONSOLE = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOCK_DIR = PROJECT_ROOT / "data" / "Market"
KEYWORDS_FILE = PROJECT_ROOT / "keywords.csv"
TRENDS_CACHE_DIR = PROJECT_ROOT / "data" / "google_trends"
RESULTS_FILE = PROJECT_ROOT / "correlation_results.csv"
PLOTS_DIR = PROJECT_ROOT / "plots" / "random_correlations"

SPARSITY_DISCARD_THRESHOLD = 0.8  # 80% -> discard completely
CORR_MIN, CORR_MAX = 0.3, 0.8  # logical correlation range
IDLE_STOCK_CAP = 26  # max amount of days when stock can stay unchanged


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
        self.missing_keywords: List[str] = []

        TRENDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        CONSOLE.log("[bold green]CorrelationFinder initialized.[/bold green]")

    def _load_keywords(self):
        CONSOLE.log(f"Loading keywords from [cyan]{KEYWORDS_FILE}[/cyan]...")
        self.keyword_df = pd.read_csv(KEYWORDS_FILE).dropna(subset=['keyword'])
        self.keyword_df['keyword'] = self.keyword_df['keyword'].astype(str)
        CONSOLE.log(f"Loaded [bold yellow]{len(self.keyword_df)}[/bold yellow] keywords.")

    def _trim_or_fix_stock_data(self, df: pd.DataFrame, stock_name: str) -> pd.DataFrame:
        """Trim from first corrupted streak >= IDLE_STOCK_CAP days."""
        df = df.copy()
        corrupted = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])
        corrupted |= (df['Volume'] == 0)

        if not corrupted.any():
            return df

        streak_count = 0
        first_bad_day = None
        for date, is_bad in corrupted.items():
            if is_bad:
                if streak_count == 0:
                    first_bad_day = date
                streak_count += 1
                if streak_count >= IDLE_STOCK_CAP:
                    CONSOLE.log(f"[yellow]{stock_name}: trimming from {first_bad_day.date()} onward[/yellow]")
                    return df.loc[:first_bad_day - pd.Timedelta(days=1)]
            else:
                streak_count = 0
                first_bad_day = None
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
        """Remove zeros, apply Z-score normalization safely, return series and sparsity ratio."""
        sparsity = (trend_series == 0).mean()
        nonzero = trend_series[trend_series != 0]
        if nonzero.empty:
            return pd.Series(dtype=float), sparsity

        std = nonzero.std(ddof=0)
        if std == 0 or np.isnan(std):
            standardized = pd.Series(0.0, index=nonzero.index)
        else:
            standardized = (nonzero - nonzero.mean()) / std

        return standardized, sparsity

    @staticmethod
    def _calculate_cross_correlation(series1: pd.Series, series2: pd.Series, max_lag: int) -> Tuple[
        int, float]:
        if len(series1.dropna()) < 2 or len(series2.dropna()) < 2:
            return 0, 0.0
        if series1.std() == 0 or series2.std() == 0:
            return 0, 0.0

        best_lag, max_corr = 0, 0.0
        for lag in range(-max_lag, max_lag + 1):
            corr = series1.corr(series2.shift(lag))
            if pd.notna(corr) and abs(corr) > abs(max_corr):
                best_lag, max_corr = lag, corr
        return best_lag, max_corr

    def _should_skip(self, stock: str, keyword: str, metric: str, done: set) -> bool:
        """Check if this correlation already exists (incremental update)."""
        return (stock, keyword, metric) in done

    def _load_trend_series(self, keyword: str, cached_trend_files: Dict[str, Path]) -> Tuple[
        pd.Series, float]:
        """Load and preprocess Google Trends series, handle missing/empty files."""
        try:
            trend_df = pd.read_csv(cached_trend_files[keyword], index_col='date', parse_dates=True)
            if trend_df.empty or keyword not in trend_df.columns:
                self.missing_keywords.append(keyword)
                return pd.Series(dtype=float), 1.0
            return self._preprocess_trend_series(trend_df[keyword].astype(float).fillna(0))
        except pd.errors.EmptyDataError:
            self.missing_keywords.append(keyword)
            return pd.Series(dtype=float), 1.0
        except Exception as e:
            CONSOLE.log(f"[red]Trend load error for {keyword}: {e}[/red]")
            return pd.Series(dtype=float), 1.0

    def _compute_correlation(self, stock_df_weekly: pd.DataFrame, trend_series: pd.Series,
                             metric: str, sparsity: float) -> Tuple[int, float]:
        """Compute best lag correlation between stock returns and trend."""
        if trend_series.empty or sparsity > SPARSITY_DISCARD_THRESHOLD:
            return 0, 0.0
        max_lag = 1 if sparsity > (SPARSITY_DISCARD_THRESHOLD * 0.6) else 3
        aligned_stock, aligned_trend = stock_df_weekly.align(trend_series, join='inner', axis=0)
        if aligned_trend.empty:
            return 0, 0.0
        return self._calculate_cross_correlation(aligned_stock[metric], aligned_trend, max_lag)

    def _process_metric(self, stock_name: str, stock_df_weekly: pd.DataFrame, keyword: str,
                        category: str, metric: str, done: set,
                        cached_trend_files: Dict[str, Path]) -> Optional[dict]:
        """Process one metric for stock-keyword pair."""
        if self._should_skip(stock_name, keyword, metric, done):
            return None

        trend_series, sparsity = self._load_trend_series(keyword, cached_trend_files)
        lag, corr = self._compute_correlation(stock_df_weekly, trend_series, metric, sparsity)

        if CORR_MIN <= abs(corr) <= CORR_MAX:
            return {
                'Stock': stock_name, 'Keyword': keyword, 'Keyword_Category': category,
                'Metric': metric, 'Best_Lag_Weeks': lag, 'Correlation': corr
            }
        return None

    def _process_stock_keyword(self, stock_name: str, stock_df_weekly: pd.DataFrame,
                               keyword: str, category: str, done: set,
                               cached_trend_files: Dict[str, Path]) -> List[dict]:
        """Process both metrics (Close_Returns, Volume_Returns) for one stock-keyword."""
        results = []
        for metric in ['Close_Returns', 'Volume_Returns']:
            result = self._process_metric(stock_name, stock_df_weekly, keyword, category, metric, done,
                                          cached_trend_files)
            if result:
                results.append(result)
        return results

    def run_analysis(self, target_stocks: List[str], target_keywords_df: pd.DataFrame,
                     existing_results: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Main loop: iterate stocks → keywords → metrics, build correlations."""
        results = []
        cached_trend_files = {p.stem.replace('_', ' '): p for p in TRENDS_CACHE_DIR.glob("*.csv")}
        keywords_to_analyze_df = target_keywords_df[
            target_keywords_df['keyword'].isin(cached_trend_files.keys())]

        already_done = set()
        if existing_results is not None and not existing_results.empty:
            already_done = set(
                zip(existing_results['Stock'], existing_results['Keyword'], existing_results['Metric']))

        for stock_name in tqdm(target_stocks, desc="Analyzing Stocks"):
            stock_df_weekly = self.stock_data_weekly.get(stock_name)
            if stock_df_weekly is None:
                continue
            for _, row in keywords_to_analyze_df.iterrows():
                keyword, category = row['keyword'], row['category']
                results.extend(self._process_stock_keyword(stock_name, stock_df_weekly,
                                                           keyword, category, already_done,
                                                           cached_trend_files))

        return pd.DataFrame(results)

    def _save_missing_keywords(self):
        if self.missing_keywords:
            missing_file = PROJECT_ROOT / "missing_keywords.txt"
            with open(missing_file, "w", encoding="utf-8") as f:
                for kw in sorted(set(self.missing_keywords)):
                    f.write(f"{kw}\n")

    def _plot_top_for_stock(self, stock: str, top_df: pd.DataFrame):
        for _, row in top_df.iterrows():
            stock_df = self.stock_data_weekly[stock]
            keyword, metric, lag, corr = row['Keyword'], row['Metric'], row['Best_Lag_Weeks'], row[
                'Correlation']
            try:
                trend_df = pd.read_csv(
                    TRENDS_CACHE_DIR / f"{keyword.replace(' ', '_')}.csv",
                    index_col='date', parse_dates=True
                )
                trend_series, _ = self._preprocess_trend_series(trend_df[keyword].astype(float).fillna(0))
                shifted_trend = trend_series.shift(lag)
                aligned_stock, aligned_trend = stock_df.align(shifted_trend, join='inner', axis=0)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=aligned_stock.index, y=aligned_stock[metric],
                                         name=f"{stock} {metric}"), secondary_y=False)
                fig.add_trace(go.Scatter(x=aligned_trend.index, y=aligned_trend,
                                         name=f"Trend '{keyword}' (Lag {lag}w)"), secondary_y=True)
                fig.update_layout(
                    title=f"{stock} vs '{keyword}' — Correlation {corr:.3f} (Lag {lag}w)",
                    title_x=0.5
                )
                fig.write_html(PLOTS_DIR / f"{stock}_{keyword}_{metric}.html".replace(' ', '_'))
            except Exception as e:
                CONSOLE.log(f"[red]Plot error {stock}-{keyword}: {e}[/red]")

    def execute_pipeline(self):
        self._load_keywords()
        self._load_and_preprocess_stock_data()
        if self.keyword_df is None or not self.stock_data_daily:
            return

        all_stocks = list(self.stock_data_daily.keys())
        existing_df = None
        if RESULTS_FILE.exists():
            try:
                existing_df = pd.read_csv(RESULTS_FILE)
            except Exception:
                existing_df = None

        new_results = self.run_analysis(all_stocks, self.keyword_df, existing_df)

        self._save_missing_keywords()

        if new_results.empty and (existing_df is None or existing_df.empty):
            CONSOLE.log("[bold yellow]No correlations found in range 0.3–0.8[/bold yellow]")
            return

        final_df = pd.concat([existing_df, new_results],
                             ignore_index=True) if existing_df is not None else new_results
        final_df.to_csv(RESULTS_FILE, index=False)
        CONSOLE.log(f"[bold green]Results saved to {RESULTS_FILE}[/bold green]")

        for stock in all_stocks:
            stock_df = final_df[final_df['Stock'] == stock].copy()
            if stock_df.empty:
                print(f"{stock}: no correlation found")
                continue
            stock_df['Abs_Correlation'] = stock_df['Correlation'].abs()
            top3 = stock_df.nlargest(3, 'Abs_Correlation')
            print(f"\nTop correlations for {stock}:")
            print(
                top3[['Stock', 'Keyword', 'Metric', 'Best_Lag_Weeks', 'Correlation']].to_string(index=False))
            self._plot_top_for_stock(stock, top3)


def main():
    setup_logging_and_warnings()
    finder = CorrelationFinder()
    finder.execute_pipeline()


if __name__ == "__main__":
    main()
