"""Daily NEPSE stock screener.

Fetches live data from merolagani.com (per-company fundamentals) and ranks
listed equities by undervalued fundamentals and bonus history.

Ranking logic (mirrors the original nepsealpha-based screener):
  * Undervalued (0-2): P/E below sector median + P/BV below sector median
  * Ratios Summary: Strong / Medium / Weak heuristic from EPS, P/E, P/BV
  * Bonus %: latest fiscal year's bonus share percentage
  * Sorted by Undervalued desc, Ratio desc, LTP asc, Bonus % desc
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

RATIO_MAP = {"Strong": 1, "Medium": 0, "Weak": -1}


def _num(text):
    # drop trailing annotations like "(FY:082-083, Q:4)" before parsing
    text = (text or "").split("(")[0]
    cleaned = re.sub(r"[^\d.\-]", "", text)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _field(html, label):
    """Extract the <td> value following the <th> containing `label`."""
    m = re.search(
        r"<th[^>]*>.*?" + re.escape(label) + r".*?</th>\s*<td[^>]*>(.*?)</td>",
        html,
        re.I | re.S,
    )
    if not m:
        return None
    return re.sub(r"<[^>]+>", "", m.group(1)).strip()


def fetch_company(symbol):
    """Scrape one company's fundamentals from merolagani. Returns dict or None."""
    for attempt in range(3):
        try:
            r = SESSION.get(
                f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}",
                timeout=30,
            )
            if r.status_code != 200:
                return None
            html = r.text
            sector = _field(html, "Sector")
            if not sector:
                return None
            bonus = _num(_field(html, "% Bonus"))
            return {
                "Symbol": symbol,
                "Sector": sector,
                "LTP": _num(_field(html, "Market Price")),
                "EPS": _num(_field(html, "EPS")),
                "PE": _num(_field(html, "P/E Ratio")),
                "PBV": _num(_field(html, "PBV")),
                "Bonus %": bonus if bonus is not None else -1,
            }
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def get_symbols():
    """Live symbol list from sharesansar's today-share-price table."""
    r = SESSION.get("https://www.sharesansar.com/today-share-price", timeout=30)
    (df,) = pd.read_html(StringIO(r.text))
    symbols = df["Symbol"].astype(str).str.strip().tolist()
    # keep equities only: drop debentures / instruments with digits in symbol
    return sorted({s for s in symbols if s and not re.search(r"\d", s)})


def ratios_summary(row, pe_median, pbv_median):
    """Heuristic replacement for nepsealpha's Ratios Summary."""
    if row["EPS"] is None or row["EPS"] <= 0:
        return "Weak"
    if row["PE"] is None or row["PBV"] is None:
        return "Medium"
    if row["PE"] <= pe_median and row["PBV"] <= pbv_median:
        return "Strong"
    return "Medium"


def main():
    symbols = get_symbols()
    print(f"found {len(symbols)} symbols", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, row in enumerate(pool.map(fetch_company, symbols)):
            if row:
                rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  scraped {i + 1}/{len(symbols)}", flush=True)
                time.sleep(1)

    df = pd.DataFrame(rows)
    print(f"got fundamentals for {len(df)} companies", flush=True)

    df = df.dropna(subset=["LTP", "Sector"])
    df = df[df["Sector"].str.strip() != ""]

    # sector medians for the undervalued-vs-sector comparison
    df["PE_pos"] = df["PE"].where(df["PE"] > 0)
    med = df.groupby("Sector").agg(pe_med=("PE_pos", "median"), pbv_med=("PBV", "median"))

    def undervalued(row):
        score = 0
        m = med.loc[row["Sector"]]
        if row["PE"] is not None and row["PE"] > 0 and row["PE"] < m["pe_med"]:
            score += 1
        if row["PBV"] is not None and row["PBV"] > 0 and row["PBV"] < m["pbv_med"]:
            score += 1
        return score

    df["Undervalued"] = df.apply(undervalued, axis=1)
    df["Ratios Summary"] = df.apply(
        lambda r: ratios_summary(r, med.loc[r["Sector"]]["pe_med"], med.loc[r["Sector"]]["pbv_med"]),
        axis=1,
    )
    df["Ratio"] = df["Ratios Summary"].map(RATIO_MAP)

    df["Symbol"] = df["Symbol"].apply(
        lambda s: f"[{s}](https://merolagani.com/CompanyDetail.aspx?symbol={s})"
    )
    df = df.sort_values(
        by=["Undervalued", "Ratio", "LTP", "Bonus %"],
        ascending=[False, False, True, False],
    )
    df = df.loc[:, ["Symbol", "Ratios Summary", "Sector", "LTP", "Undervalued", "Bonus %"]]

    npt = timezone(timedelta(hours=5, minutes=45))
    updated = datetime.now(npt).strftime("%Y-%m-%d %H:%M NPT")
    header = (
        "# auto-nepse\n\n"
        "Daily NEPSE stock screener — auto-ranks Nepal Stock Exchange stocks "
        "by undervalued fundamentals and bonus history.\n\n"
        f"_Last updated: {updated} · Source: merolagani.com · "
        f"{len(df)} companies ranked_\n\n"
        "| Undervalued = P/E and P/BV below sector median | "
        "Ratios Summary: Strong = profitable and cheap vs sector |\n\n"
    )
    Path("README.md").write_text(header + df.to_markdown(index=False) + "\n")
    print(f"wrote README.md with {len(df)} ranked stocks", flush=True)


if __name__ == "__main__":
    main()
