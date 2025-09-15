import pandas as pd
import time
from requests_html import HTMLSession

# ETFs requested
ETFS = [
    "XLK",
    "XLV",
    "XLF",
    "XLY",
    "XLP",
    "XLI",
    "XLE",
    "XLU",
    "XLB",
    "XLC",
    "XLRE",
    "IBB",
    "KBE",
    "KRE",
    "IYT",
    "ITB",
    "^GSPC",
    "ICLN",
    "TAN",
    "PHO",
    "EIS",
    "ISRA",
    "FRDM",
    "RSX",
    "ERUS",
    "SHE",
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
]

BASE = "https://stockanalysis.com/etf/{ticker}/holdings/"


def fetch_top50_from_stockanalysis(ticker):
    url = BASE.format(ticker=ticker.lower())
    session = HTMLSession()
    response = session.get(url)
    response.html.render(timeout=30, sleep=1)  # render JS to load the table
    # Parse table rows
    rows = response.html.find("table tbody tr")
    symbols = []
    for tr in rows:
        # symbol is in the 2nd td (first is rank)
        tds = tr.find("td")
        if len(tds) >= 2:
            sym = tds[1].text.strip().split()[0]  # symbol text
            if sym and sym.isalpha():
                symbols.append(sym)
        if len(symbols) >= 50:
            break
    # try static HTML tables if rendering fails
    if not symbols:
        try:
            dfs = pd.read_html(response.html.html)
            for df in dfs:
                if "Symbol" in df.columns:
                    symbols = df["Symbol"].astype(str).tolist()[:50]
                    break
        except Exception:
            pass
    return symbols


def main():
    out_rows = []
    for etf in ETFS:
        try:
            syms = fetch_top50_from_stockanalysis(etf)
            if not syms:
                print(f"[warn] No symbols parsed for {etf}; check the page manually.")
            for s in syms:
                out_rows.append({"ETF": etf, "Ticker": s})
            print(f"{etf}: collected {len(syms)} symbols")
            time.sleep(1.5)  # be polite
        except Exception as e:
            print(f"[error] {etf}: {e}")
    out = pd.DataFrame(out_rows, columns=["ETF", "Ticker"])
    out.to_csv("etf_holdings_top50.csv", index=False)
    print("Wrote etf_holdings_top50.csv")


if __name__ == "__main__":
    main()
