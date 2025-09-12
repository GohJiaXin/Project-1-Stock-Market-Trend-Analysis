import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stock_analysis import compute_sma, compute_daily_returns, compute_runs, max_profit_stock_ii

sns.set_style("whitegrid")  # set seaborn theme


def load_prices(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # fix common variants
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})

    assert "date" in df.columns, f"CSV must have a Date column, found: {df.columns}"
    assert "close" in df.columns, f"CSV must have a Close column, found: {df.columns}"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").set_index("date")

    return df


def plot_close_sma(df: pd.DataFrame, window: int, outpath: str):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=df.index, y="close", label="Close", linewidth=2)
    sns.lineplot(data=df, x=df.index, y=compute_sma(df["close"], window),
                 label=f"SMA ({window})", linewidth=2)
    plt.title(f"Closing Price vs SMA ({window})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_runs(df: pd.DataFrame, outpath: str):
    d = df["close"].copy()
    up_mask = d.diff() > 0
    down_mask = d.diff() < 0

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=d.index, y=d.values, label="Close", linewidth=2)
    sns.scatterplot(x=d.index[up_mask], y=d[up_mask], marker="^",
                    color="green", s=60, label="Up days")
    sns.scatterplot(x=d.index[down_mask], y=d[down_mask], marker="v",
                    color="red", s=60, label="Down days")

    plt.title("Closing Price with Up/Down Markers")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Stock Market Trend Analysis")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--sma-window", type=int, default=5, help="Window size for SMA")
    parser.add_argument("--outdir", default="outputs", help="Directory to save outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_prices(args.csv)
    df[f"sma_{args.sma_window}"] = compute_sma(df["close"], args.sma_window)
    df["daily_return"] = compute_daily_returns(df["close"])

    runs, counts, longest = compute_runs(df["close"])
    profit, trades = max_profit_stock_ii(df["close"])

    # save plots
    plot_close_sma(df, args.sma_window, os.path.join(args.outdir, "close_sma.png"))
    plot_runs(df, os.path.join(args.outdir, "runs_highlight.png"))

    # save summary
    summary = {
        "up_runs": counts["up_runs"],
        "down_runs": counts["down_runs"],
        "up_days": counts["up_days"],
        "down_days": counts["down_days"],
        "longest_up_run": None if longest["up"] is None else {
            "length": int(longest["up"].length),
            "start": str(longest["up"].start_date.date()),
            "end": str(longest["up"].end_date.date())
        },
        "longest_down_run": None if longest["down"] is None else {
            "length": int(longest["down"].length),
            "start": str(longest["down"].start_date.date()),
            "end": str(longest["down"].end_date.date())
        },
        "max_profit_stock_ii": profit,
        "number_of_trades": len(trades),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Done! Outputs saved to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
