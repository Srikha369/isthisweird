import pandas as pd
import numpy as np
from scipy import stats


#compare segments of data before and after a given date to see if there is a significant change in the metric of interest (e.g. signups, conversion rate)
def compare_segments(df: pd.DataFrame, split_date: str, metric_col: str = "signups"):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]) 
    split = pd.to_datetime(split_date) #data split date to compare segments before and after 

    results = []
    for ch in df["channel"].unique():
        sub = df[df["channel"] == ch] #makes a independent dataframe for each channel 
        before = sub[sub["date"] < split]
        after = sub[sub["date"] >= split]

        before_mean = before[metric_col].mean()  
        after_mean = after[metric_col].mean()
        pct_change = (after_mean - before_mean) / before_mean if before_mean != 0 else np.nan #percent change 

        # Two-sample t-test: is the after-period mean statistically significantly different from the before-period mean for this channel?

        t_stat, p_value = stats.ttest_ind(before[metric_col], after[metric_col], equal_var=False)

        results.append({
            "channel": ch,
            "before_mean": round(before_mean, 1),
            "after_mean": round(after_mean, 1),
            "pct_change": round(pct_change * 100, 1),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
        })

    return pd.DataFrame(results).sort_values("pct_change")

#wrapper to compare segments for both metrics and package one verdict 
def segment_comparison_tool(df: pd.DataFrame, split_date: str) -> dict:
    """
    Runs compare_segments on both signups and conversion_rate, and returns
    a structured summary an agent can reason over directly.
    """
    signups_result = compare_segments(df, split_date, metric_col="signups")
    conversion_result = compare_segments(df, split_date, metric_col="conversion_rate")

    # Which channels show a significant drop in each metric?
    signups_flagged = signups_result[
        (signups_result["significant"]) & (signups_result["pct_change"] < 0)
    ]["channel"].tolist()

    conversion_flagged = conversion_result[
        (conversion_result["significant"]) & (conversion_result["pct_change"] < 0)
    ]["channel"].tolist()

    return {
        "split_date": split_date,
        "signups_by_channel": signups_result.to_dict(orient="records"),
        "conversion_by_channel": conversion_result.to_dict(orient="records"),
        "channels_with_signups_drop": signups_flagged,
        "channels_with_conversion_drop": conversion_flagged,
        "interpretation": _interpret(signups_flagged, conversion_flagged, df["channel"].unique().tolist()),
    }


def _interpret(signups_flagged: list, conversion_flagged: list, all_channels: list) -> str:
    """Turn the flagged-channel lists into a plain-English hint for the agent."""
    if not signups_flagged and not conversion_flagged:
        return "No channel shows a significant drop in signups or conversion rate."

    if set(signups_flagged) == set(all_channels) and not conversion_flagged:
        return "Signups dropped across ALL channels with no conversion-rate change — likely a sitewide traffic issue."

    if set(conversion_flagged) == set(all_channels):
        return "Conversion rate dropped across ALL channels — likely a sitewide funnel/product issue, not a traffic issue."

    if signups_flagged and not conversion_flagged:
        return f"Signups dropped in {signups_flagged} with no conversion-rate change — likely a traffic-volume issue isolated to {signups_flagged}."

    if conversion_flagged and not signups_flagged:
        return f"Conversion rate dropped in {conversion_flagged} with traffic unaffected — likely a funnel/product issue isolated to {conversion_flagged}."

    return f"Mixed signal: signups dropped in {signups_flagged}, conversion dropped in {conversion_flagged} — needs further investigation."


def detect_changepoints(df: pd.DataFrame, metric_col: str = "signups",
                         channel: str = None, min_segment_days: int = 7) -> dict:
    """
    Scans for the date that best splits the series into two segments with
    the most different means. Returns the best candidate changepoint.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    if channel:
        data = data[data["channel"] == channel]
    else:
        # aggregate across channels if no specific channel given
        data = data.groupby("date")[metric_col].sum().reset_index()

    data = data.sort_values("date").reset_index(drop=True)
    dates = data["date"].tolist()
    values = data[metric_col].values
    n = len(values)

    best_date, best_t_stat, best_p_value = None, 0, 1.0

    # Try every candidate split point (skipping the edges, where segments
    # would be too short to test reliably) and keep whichever split
    # produces the strongest statistical separation.
    for i in range(min_segment_days, n - min_segment_days):
        before, after = values[:i], values[i:]
        t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)
        if abs(t_stat) > abs(best_t_stat):
            best_date, best_t_stat, best_p_value = dates[i], t_stat, p_value

    return {
        "channel_queried": channel or "all channels combined",
        "metric": metric_col,
        "detected_changepoint": best_date.strftime("%Y-%m-%d") if best_date is not None else None,
        "t_stat": round(best_t_stat, 3),
        "p_value": round(best_p_value, 4),
        "significant": best_p_value < 0.05,
    }


MARKETING_EVENTS_LOG = [
    {"date": "2026-02-14", "event_type": "spend_change", "channel": "paid",
     "description": "Q1 budget freeze — paid spend cut ~50%"},
    {"date": "2026-02-18", "event_type": "spend_change", "channel": "paid",
     "description": "Marketing cut paid ad spend ~80%"},
    {"date": "2026-01-25", "event_type": "spend_change", "channel": "paid",
     "description": "Paid spend increased 15% for a new campaign test"},
    {"date": "2026-02-08", "event_type": "deploy", "channel": None,
     "description": "Payment processor migration deployed"},
    {"date": "2026-02-20", "event_type": "deploy", "channel": None,
     "description": "Signup flow refactor deployed"},
    {"date": "2026-03-01", "event_type": "deploy", "channel": None,
     "description": "Unrelated minor UI deploy, no functional changes"},
]


def check_event_alignment(candidate_date: str, channel: str = None,
                           tolerance_days: int = 2) -> dict:
    """
    Checks whether any known event falls within `tolerance_days` of the
    candidate changepoint date, optionally filtered to a specific channel.
    """
    candidate = pd.to_datetime(candidate_date)
    matches = []

    for event in MARKETING_EVENTS_LOG:
        event_date = pd.to_datetime(event["date"])
        days_diff = (event_date - candidate).days

        if abs(days_diff) <= tolerance_days:
            # event applies if it's channel-specific and matches, or
            # it's a sitewide event (channel=None) which affects everything
            if channel is None or event["channel"] in (channel, None):
                matches.append({**event, "days_from_candidate": days_diff})

    if matches:
        interpretation = "; ".join(
            f"{m['description']} on {m['date']} ({m['days_from_candidate']:+d} days from candidate)"
            for m in matches
        )
    else:
        interpretation = f"No known event found within {tolerance_days} days of {candidate_date}."

    return {
        "candidate_date": candidate_date,
        "channel_queried": channel,
        "matching_events": matches,
        "interpretation": interpretation,
    }


#SANITY CHECK 
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    data_dir = Path(__file__).parent / "data" / "scenarios"
    df = pd.read_csv(data_dir / "scenario_01.csv")

    # Should find the strong changepoint around 2026-02-15 for scenario_01
    cp = detect_changepoints(df, metric_col="signups", channel="referral")
    print(cp)

    # Should find NO matching event (referral collapse has no logged cause)
    events = check_event_alignment(cp["detected_changepoint"], channel="referral")
    print(events["interpretation"])




