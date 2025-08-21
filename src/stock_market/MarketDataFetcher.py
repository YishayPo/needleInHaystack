"""
fetch_stock_data.py — Flexible stock & custom ETF data downloader using yfinance

Usage (CLI)
-----------
$ python fetch_stock_data.py --config config.json

The script reads a JSON configuration that defines what to download, how, and
where to save the resulting CSVs. It supports:
    • Single tickers (e.g. indexes, large caps)
    • Lists of tickers that will be treated as a custom equally-weighted ETF.
    • Company information including headquarters data

Configuration schema
--------------------
{
    "output_dir": "./data",                # directory to place CSVs (will be created)
    "company_info_file": "company_info.csv",  # optional: filename for company data
    "entries": [
        {
            "name": "S&P500",                 # used for the CSV filename
            "ticker": "^GSPC",                # single string → single security/index
            "period": "5y",                   # yfinance period (mutually exclusive w/ start/end)
            "interval": "1d",                 # yfinance interval
            "include_company_info": false     # optional: fetch company data (default false)
        },
        {
            "name": "LargeTech",
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],  # list → custom ETF
            "start": "2020-01-01",            # explicit date range allowed too
            "end": "2025-06-01",
            "interval": "1wk",                # weekly data
            "aggregate": "value",             # optional: value | mean | none (default none)
            "include_company_info": true      # fetch company info for all tickers
        }
    ]
}

Aggregate behavior for custom ETF entries
-----------------------------------------
* "value"   - Computes a synthetic price series by summing **(normalized first-day=1)**
                adjusted closes, then rescales to a base of 100 (a quick index proxy).
* "mean"    - Arithmetic mean of the adjusted closes.
* "none"    - Writes one column per component ticker (default).

Company Information
-------------------
When include_company_info is true, the script will fetch and save:
- Company name, sector, industry
- Headquarters location (address, city, state, zip, country)
- Contact info (phone, website)
- Key metrics (market cap, employees, etc.)


"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from rich.console import Console
from rich.progress import track
import yfinance as yf


console = Console()


@dataclass
class EntryConfig:
    name: str
    ticker: Optional[str] = None  # single ticker OR …
    tickers: Optional[List[str]] = field(default_factory=list)

    # timeframe – exactly one of (period) OR (start & end)
    period: Optional[str] = None
    start: Optional[str] = None  # YYYY‑MM‑DD
    end: Optional[str] = None

    interval: str = "1d"
    aggregate: str = "none"  # for custom ETF – value | mean | none
    join: str = "inner"      # how to align date indexes across tickers
    include_company_info: bool = True  # whether to fetch company data

    description: Optional[str] = ""

    # --------------------------------------------------------------------- #
    def is_custom_etf(self) -> bool:
        return bool(self.tickers)

    def validate(self):
        if not self.name:
            raise ValueError("Each entry must have a non‑empty 'name'.")
        if (self.ticker is None) == (not self.tickers):
            raise ValueError("Specify either 'ticker' or 'tickers', not both.")
        if self.period and (self.start or self.end):
            raise ValueError("Use 'period' OR 'start'/'end', not both.")
        if self.start and not self.end:
            raise ValueError("'start' requires 'end'.")
        if self.aggregate not in {"value", "mean", "none"}:
            raise ValueError("aggregate must be one of 'value', 'mean', 'none'.")
        if self.join not in {"inner", "outer"}:
            raise ValueError("join must be 'inner' or 'outer'.")

    # ------------------------------------------------------------------
    @property
    def yf_kwargs(self) -> Dict[str, str]:
        if self.period:
            return dict(period=self.period, interval=self.interval)
        return dict(start=self.start, end=self.end, interval=self.interval)

    @property
    def all_tickers(self) -> List[str]:
        """Return all tickers (whether single or multiple)."""
        if self.ticker:
            return [self.ticker]
        return self.tickers


@dataclass
class Config:
    output_dir: str
    entries: List[EntryConfig]
    company_info_file: str = "company_info.csv"

    @staticmethod
    def load(path: Union[str, Path]) -> "Config":
        with open(path, "r", encoding="utf‑8") as fp:
            raw = json.load(fp)
        entries = [EntryConfig(**e) for e in raw.get("entries", [])]
        if not entries:
            raise ValueError("Config must contain at least one entry.")
        for e in entries:
            e.validate()
        output_dir = raw.get("output_dir", "./data")
        company_info_file = raw.get("company_info_file", "company_info.csv")
        return Config(output_dir=output_dir, entries=entries, company_info_file=company_info_file)


PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]
VOL_COL = "Volume"


def fetch_company_info(ticker: str) -> Dict[str, any]:
    """Fetch company information including headquarters data."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract relevant company information
        company_data = {
            'ticker': ticker,
            'name': info.get('longName', ''),
            'short_name': info.get('shortName', ''),

            # Headquarters information
            'hq_address': info.get('address1', ''),
            'hq_address2': info.get('address2', ''),
            'hq_city': info.get('city', ''),
            'hq_state': info.get('state', ''),
            'hq_zip': info.get('zip', ''),
            'hq_country': info.get('country', ''),

            # Contact information
            'phone': info.get('phone', ''),
            'website': info.get('website', ''),

            # Business information
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'business_summary': info.get('longBusinessSummary', '')[:500],  # Truncate for CSV

            # Key metrics
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'employees': info.get('fullTimeEmployees', 0),
            'founded': info.get('founded', ''),

            # Financial metrics
            'currency': info.get('currency', ''),
            'exchange': info.get('exchange', ''),
            'quote_type': info.get('quoteType', ''),

            # Additional metrics
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0),
        }

        return company_data
    except Exception as e:
        console.log(f"[yellow]Warning: Could not fetch company info for {ticker}: {e}")
        return {'ticker': ticker, 'error': str(e)}


def fetch_single_ticker(entry: EntryConfig) -> pd.DataFrame:
    """Download a *single* ticker via yfinance and return the raw dataframe."""
    console.log(f"Downloading {entry.ticker} ({entry.name}) …")
    df = yf.download(entry.ticker, progress=False, **entry.yf_kwargs)
    if df.empty:
        console.log(f"[bold red]No data returned for {entry.ticker}")
    return df


# ------------------------------------------------------------------------- #

def _aggregate_series(cat: pd.DataFrame, method: str, normalize: bool) -> pd.Series:
    """Helper: aggregate per-column stacked frame *cat* using *method*.

    Parameters
    ----------
    cat : pd.DataFrame  (columns = tickers, index = dates)
    method : "mean" | "value"
    normalize : bool        - if True, first normalize each column to its first value
                            before applying the aggregation (for price columns when
                            method == "value").
    """
    if cat.empty or cat.iloc[0].isna().all():
        # Nothing to aggregate – return an empty Series so we can skip this column later
        return pd.Series(dtype='float64')

    if method == "mean":
        return cat.mean(axis=1)

    # "value" → sum after optional normalization
    if normalize:
        cat = cat.div(cat.iloc[0])
    summed = cat.sum(axis=1)
    if normalize:
        summed = 100 * summed / summed.iloc[0]
    return summed


def fetch_custom_etf(entry: EntryConfig) -> pd.DataFrame:
    console.log(
        f"Building custom ETF '{entry.name}' from {len(entry.tickers)} tickers …"
    )

    ticker_frames: List[pd.DataFrame] = []
    for t in track(entry.tickers, description="Tickers"):
        data = yf.download(t, progress=False, **entry.yf_kwargs)
        if data.empty:
            console.log(f"[yellow] No data for {t}, skipping.")
            continue
        ticker_frames.append(data)

    if not ticker_frames:
        raise RuntimeError(
            f"No data available for any ticker in {entry.tickers}."
        )

    # ---------------- un‑aggregated mode ----------------
    if entry.aggregate == "none":
        prefixed = [df.add_prefix(f"{t}_") for df, t in zip(ticker_frames, entry.tickers)]
        out = pd.concat(prefixed, axis=1, join=entry.join).sort_index()
        return out

    # ---------------- aggregated mode -------------------
    agg_columns: Dict[str, pd.Series] = {}

    # price‑like columns (Open … Close [+ Adj Close])
    for col in PRICE_COLS:
        # skip Adj Close if *any* ticker lacks it (NYSE indices, some crypto …)
        if col == "Adj Close" and not all(col in df.columns for df in ticker_frames):
            continue
        cat = pd.concat(
            [df[col] for df in ticker_frames if col in df.columns],
            axis=1,
            join=entry.join,
        )
        normalize = (entry.aggregate == "value") and (col != VOL_COL)
        agg_columns[col] = _aggregate_series(cat, entry.aggregate, normalize)

    # volume (always raw numbers)
    if VOL_COL in ticker_frames[0].columns:
        cat = pd.concat(
            [df[VOL_COL] for df in ticker_frames if VOL_COL in df.columns],
            axis=1,
            join=entry.join,
        )
        agg_columns[VOL_COL] = _aggregate_series(cat, entry.aggregate, False)

    aggregated_df = pd.concat(agg_columns, axis=1)
    aggregated_df.columns.name = None  # flat column index for CSV
    return aggregated_df


# ------------------------------- IO -------------------------------------- #

def save_csv(df: pd.DataFrame, entry_name: str, out_dir: Path):
    if df.empty:
        console.log(f"[yellow] Skipping write for {entry_name} (empty frame).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{entry_name}.csv"
    df.to_csv(file_path, index_label="Date")
    console.log(f"Saved {file_path.relative_to(Path.cwd())}")


def save_company_info(company_data_list: List[Dict], out_dir: Path, filename: str):
    """Save company information to CSV."""
    if not company_data_list:
        return

    df = pd.DataFrame(company_data_list)
    # Order columns logically
    column_order = [
        'ticker', 'name', 'short_name',
        'hq_address', 'hq_address2', 'hq_city', 'hq_state', 'hq_zip', 'hq_country',
        'phone', 'website', 'sector', 'industry',
        'market_cap', 'enterprise_value', 'employees', 'currency', 'exchange',
        'pe_ratio', 'forward_pe', 'dividend_yield', 'beta',
        '52_week_high', '52_week_low', 'business_summary'
    ]
    # Only include columns that exist in the dataframe
    df = df[[col for col in column_order if col in df.columns]]

    file_path = out_dir / filename
    df.to_csv(file_path, index=False)
    console.log(f"Saved company info to {file_path.relative_to(Path.cwd())}")


# ------------------------------- CLI ------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Download stock/index/portfolio data via yfinance."
    )
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file.")
    parser.add_argument("--company_data", required=False, action="store_true", help="Fetch only company data.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    only_company_info = args.company_data
    out_dir = Path(cfg.output_dir).expanduser().resolve()

    # Collect all company data if any entry requests it
    all_company_data = []
    processed_tickers = set()

    for entry in cfg.entries:
        if only_company_info and not entry.include_company_info:
            console.log(f"Skipping {entry.name} (not fetching company info).")
            continue
        elif not only_company_info:
            # Fetch price data
            if entry.is_custom_etf():
                df = fetch_custom_etf(entry)
            else:
                df = fetch_single_ticker(entry)
            save_csv(df, entry.name, out_dir)

        # Fetch company info if requested
        if entry.include_company_info:
            console.log(f"Fetching company info for {entry.name} …")
            for ticker in track(entry.all_tickers, description="Company info"):
                if ticker not in processed_tickers and not ticker.startswith('^'):
                    # Skip index tickers (^GSPC, etc.) as they don't have company info
                    company_info = fetch_company_info(ticker)
                    if 'error' not in company_info:
                        all_company_data.append(company_info)
                        processed_tickers.add(ticker)

    # Save all company data to a single CSV
    if all_company_data:
        save_company_info(all_company_data, out_dir, cfg.company_info_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        console.log(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)