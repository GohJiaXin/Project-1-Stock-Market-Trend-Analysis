import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from stock_analysis import compute_sma, compute_daily_returns, compute_runs, max_profit_stock_ii


def load_prices(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names
    df.columns = [c.strip().title() for c in df.columns]
    assert "Date" in df.columns and "Close" in df.columns, "CSV must have Date and Close columns."
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.set_index("Date")
    # make sure numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def plot_close_sma(df: pd.DataFrame, window: int, outpath: str):
    plt.figure()
    df["Close"].plot(label="Close")
    compute_sma(df["Close"], window).plot(label=f"SMA ({window})")
    plt.title(f"Close vs SMA ({window})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_runs(df: pd.DataFrame, outpath: str):
    d = df["Close"].copy()
    up_mask = d.diff() > 0
    down_mask = d.diff() < 0

    plt.figure()
    d.plot(label="Close")
    plt.plot(d.index[up_mask], d[up_mask], marker="^", linestyle="None", label="Up days")
    plt.plot(d.index[down_mask], d[down_mask], marker="v", linestyle="None", label="Down days")
    plt.title("Close with Up/Down Day Markers")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Stock Market Trend Analysis")
    parser.add_argument("--csv", required=True, help="Path to input CSV with Date, Open, High, Low, Close, Volume")
    parser.add_argument("--sma-window", type=int, default=5, help="Window size for SMA")
    parser.add_argument("--outdir", default="outputs", help="Directory to save outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_prices(args.csv)
    df[f"SMA_{args.sma_window}"] = compute_sma(df["Close"], args.sma_window)
    df["Daily_Return"] = compute_daily_returns(df["Close"])

    runs, counts, longest = compute_runs(df["Close"])
    profit, trades = max_profit_stock_ii(df["Close"])

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
