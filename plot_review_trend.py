# plot_review_trend.py

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

DEFAULT_YEAR_FOR_YEARLESS_DATES = 2026
CURRENT_YEAR = datetime.now().year  # Dynamically get the current year

def parse_review_date(raw_date: str) -> datetime:
    """
    Parse review dates like:
      - Apr 1
      - September 4
      - 09/04/2023
      - Apr 1, 2024

    If no year is present, determine if the month is in the future.
    If it is, use the current year. If not, use the previous year.
    """
    raw_date = raw_date.strip()

    has_year = bool(re.search(r"\b\d{4}\b", raw_date))

    default_date = datetime(DEFAULT_YEAR_FOR_YEARLESS_DATES, 1, 1)

    parsed = date_parser.parse(
        raw_date,
        default=default_date,
        fuzzy=True,
        dayfirst=False,
    )


    if not has_year:
        today = datetime.now()
        if parsed.month > today.month or (parsed.month == today.month and parsed.day > today.day):
            # Date is in the future, use the current year
            parsed = parsed.replace(year=CURRENT_YEAR - 1)

    return parsed

def extract_release_date(html_text: str) -> Optional[datetime]:
    """
    Look for a <page-media-reviews-manager> block, parse its JSON props,
    and return media.theaterReleaseDate as a datetime if present.
    Returns None when the block, the JSON, or the field is missing.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    manager = soup.find("page-media-reviews-manager")
    if manager is None:
        return None

    script = manager.find("script", attrs={"data-json": "props"})
    if script is None or not script.string:
        return None

    try:
        props = json.loads(script.string)
    except json.JSONDecodeError:
        return None

    raw_date = props.get("media", {}).get("theaterReleaseDate")
    if not raw_date:
        return None

    try:
        return date_parser.parse(raw_date)
    except (ValueError, TypeError):
        return None


def extract_reviews(html_text: str) -> pd.DataFrame:
    soup = BeautifulSoup(html_text, "html.parser")

    rows = []

    for card in soup.find_all("review-card-critic"):
        timestamp_el = card.find(attrs={"slot": "timestamp"})
        score_el = card.find("score-icon-critics")

        if timestamp_el is None or score_el is None:
            continue

        raw_date = timestamp_el.get_text(strip=True)
        sentiment = score_el.get("sentiment", "").strip().upper()

        try:
            review_date = parse_review_date(raw_date)
        except Exception:
            continue

        rows.append(
            {
                "date": review_date.date(),
                "raw_date": raw_date,
                "sentiment": sentiment,
                "is_positive": sentiment == "POSITIVE",
            }
        )

    if not rows:
        raise ValueError("No usable reviews were found in the HTML file.")

    return pd.DataFrame(rows)


def build_cumulative_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            reviews_on_date=("is_positive", "size"),
            positives_on_date=("is_positive", "sum"),
        )
        .sort_values("date")
    )

    daily["cumulative_reviews"] = daily["reviews_on_date"].cumsum()
    daily["cumulative_positive"] = daily["positives_on_date"].cumsum()
    daily["percent_positive_to_date"] = (
        daily["cumulative_positive"] / daily["cumulative_reviews"] * 100
    )

    return daily


def plot_trend(
    trend: pd.DataFrame,
    output_path: str,
    release_date: Optional[datetime] = None,
) -> None:
    plt.figure(figsize=(10, 6))

    min_y = min(trend["percent_positive_to_date"])
    plt.plot(
        trend["date"],
        trend["percent_positive_to_date"],
        marker="o",
    )

    if release_date is not None:
        plt.axvline(
            release_date.date(),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Theater release ({release_date.date()})",
        )
        plt.legend()

    plt.xlabel("Date")
    plt.ylabel("Percent positive reviews up to date")
    plt.title("Cumulative Positive Review Percentage Over Time")
    plt.ylim(min_y, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative Rotten Tomatoes-style review sentiment over time."
    )
    parser.add_argument("html_file", help="Path to saved HTML source file")
    parser.add_argument(
        "--output",
        default="review_trend.png",
        help="Output image filename. Default: review_trend.png",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save the computed trend as a CSV file",
    )
    parser.add_argument(
        "--release-date",
        default=None,
        help=(
            "Theater release date to draw as a vertical line "
            "(e.g. 'Dec 14, 2018' or '2018-12-14'). "
            "Overrides the value extracted from the HTML."
        ),
    )

    args = parser.parse_args()

    html_text = Path(args.html_file).read_text(encoding="utf-8")

    reviews = extract_reviews(html_text)
    trend = build_cumulative_trend(reviews)

    if args.release_date:
        try:
            release_date = date_parser.parse(args.release_date)
        except (ValueError, TypeError):
            raise SystemExit(
                f"Could not parse --release-date value: {args.release_date!r}"
            )
    else:
        release_date = extract_release_date(html_text)

    if args.csv:
        trend.to_csv(args.csv, index=False)

    plot_trend(trend, args.output, release_date=release_date)

    print(f"Parsed {len(reviews)} reviews.")
    print(f"Saved graph to {args.output}")

    if release_date is not None:
        print(f"Marked theater release date: {release_date.date()}")

    if args.csv:
        print(f"Saved trend data to {args.csv}")


if __name__ == "__main__":
    main()
