# plot_review_trend.py

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

DEFAULT_YEAR_FOR_YEARLESS_DATES = 2026
CURRENT_YEAR = datetime.now().year  # Dynamically get the current year
DEFAULT_DAYS_AFTER_THEATER_RELEASE = 7

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

def _parse_props_date(props: dict, field: str) -> Optional[datetime]:
    raw_date = props.get("media", {}).get(field)
    if not raw_date:
        return None
    try:
        return date_parser.parse(raw_date)
    except (ValueError, TypeError):
        return None


def extract_release_dates(
    html_text: str,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Look for a <page-media-reviews-manager> block, parse its JSON props,
    and return (theaterReleaseDate, streamingReleaseDate) as datetimes.
    Either value is None when the block, the JSON, or the field is missing.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    manager = soup.find("page-media-reviews-manager")
    if manager is None:
        return None, None

    script = manager.find("script", attrs={"data-json": "props"})
    if script is None or not script.string:
        return None, None

    try:
        props = json.loads(script.string)
    except json.JSONDecodeError:
        return None, None

    return (
        _parse_props_date(props, "theaterReleaseDate"),
        _parse_props_date(props, "streamingReleaseDate"),
    )


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
    theater_release_date: Optional[datetime] = None,
    streaming_release_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> None:
    if end_date is not None:
        trend = trend[trend["date"] <= end_date.date()]

    if trend.empty:
        raise SystemExit("No review data falls within the requested date range.")

    plt.figure(figsize=(10, 6))

    min_y = min(trend["percent_positive_to_date"])
    plt.plot(
        trend["date"],
        trend["percent_positive_to_date"],
        marker="o",
    )

    has_label = False
    if theater_release_date is not None:
        plt.axvline(
            theater_release_date.date(),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Theater release ({theater_release_date.date()})",
        )
        has_label = True

    if streaming_release_date is not None:
        plt.axvline(
            streaming_release_date.date(),
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Streaming release ({streaming_release_date.date()})",
        )
        has_label = True

    if has_label:
        plt.legend()

    plt.xlabel("Date")
    plt.ylabel("Percent positive reviews up to date")
    plt.title("Cumulative Positive Review Percentage Over Time")
    plt.ylim(min_y, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.show()


def _parse_cli_date(value: str, flag_name: str) -> datetime:
    try:
        return date_parser.parse(value)
    except (ValueError, TypeError):
        raise SystemExit(f"Could not parse {flag_name} value: {value!r}")


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
    parser.add_argument(
        "--streaming-release-date",
        default=None,
        help=(
            "Streaming release date to draw as a vertical line "
            "(e.g. 'Apr 28, 2026' or '2026-04-28'). "
            "Overrides the value extracted from the HTML."
        ),
    )
    end_group = parser.add_mutually_exclusive_group()
    end_group.add_argument(
        "--end-date",
        default=None,
        help=(
            "Last date to include in the plot "
            "(e.g. 'Dec 21, 2018' or '2018-12-21'). "
            f"Defaults to {DEFAULT_DAYS_AFTER_THEATER_RELEASE} days after the "
            "theater release date when one is available."
        ),
    )
    end_group.add_argument(
        "--full-history",
        action="store_true",
        help=(
            "Plot the entire review history instead of cutting off "
            "one week after the theater release date."
        ),
    )

    args = parser.parse_args()

    html_text = Path(args.html_file).read_text(encoding="utf-8")

    reviews = extract_reviews(html_text)
    trend = build_cumulative_trend(reviews)

    extracted_theater, extracted_streaming = extract_release_dates(html_text)

    if args.release_date:
        theater_release_date = _parse_cli_date(args.release_date, "--release-date")
    else:
        theater_release_date = extracted_theater

    if args.streaming_release_date:
        streaming_release_date = _parse_cli_date(
            args.streaming_release_date, "--streaming-release-date"
        )
    else:
        streaming_release_date = extracted_streaming

    if args.full_history:
        end_date: Optional[datetime] = None
    elif args.end_date:
        end_date = _parse_cli_date(args.end_date, "--end-date")
    elif theater_release_date is not None:
        end_date = theater_release_date + timedelta(
            days=DEFAULT_DAYS_AFTER_THEATER_RELEASE
        )
    else:
        end_date = None

    if args.csv:
        trend.to_csv(args.csv, index=False)

    plot_trend(
        trend,
        args.output,
        theater_release_date=theater_release_date,
        streaming_release_date=streaming_release_date,
        end_date=end_date,
    )

    print(f"Parsed {len(reviews)} reviews.")
    print(f"Saved graph to {args.output}")

    if theater_release_date is not None:
        print(f"Marked theater release date: {theater_release_date.date()}")

    if streaming_release_date is not None:
        print(f"Marked streaming release date: {streaming_release_date.date()}")

    if end_date is not None:
        print(f"Plot ends at: {end_date.date()}")
    else:
        print("Plot shows the full review history.")

    if args.csv:
        print(f"Saved trend data to {args.csv}")


if __name__ == "__main__":
    main()
