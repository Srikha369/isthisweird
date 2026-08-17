import numpy as np
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass, asdict

CHANNELS = ["organic", "paid", "referral"]

#base traffic and conversation rates per channel 
BASE_TRAFFIC = {"organic": 4000, "paid": 2500, "referral": 900}
BASE_CONVERSION = {"organic": 0.045, "paid": 0.06, "referral": 0.09}

#incident data structure 
@dataclass
class Incident:
    incident_type: str            # "traffic_collapse" | "conversion_bug" | "sitewide_bug" | "pricing_change" | "spend_cut"
    start_date: str               # ISO date, e.g. "2026-02-15"
    affected_channel: str | None  # None if it hits all channels
    magnitude: float              # severity, 0-1
    description: str              # plain-English ground truth explanation

#because b2b signups are generally lower on weekends, we will use weekly seasonality to make sure agent can detect actual anomaly and not common noise 
def _weekly_seasonality(day_of_week: int) -> float:
    #week starts on monday (0) and sunday is 6
    weekday_multipliers = [1.05, 1.08, 1.08, 1.05, 0.95, 0.6, 0.55]
    return weekday_multipliers[day_of_week]



#BASELINE GENERATOR 

def generate_baseline(start_date: str, n_days: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    rows = []
    #for each date, we will go through each channel and generate traffic, conversation rate, signups and make dataframe containing it  
    for d in dates:
        seasonal = _weekly_seasonality(d.weekday()) #float multiplier for seasonality 
        for ch in CHANNELS:

            traffic = BASE_TRAFFIC[ch] * seasonal * rng.normal(1.0, 0.06)
            traffic = max(traffic, 0)

            conv_rate = BASE_CONVERSION[ch] * rng.normal(1.0, 0.08)
            conv_rate = max(conv_rate, 0)

            signups = rng.binomial(int(traffic), min(conv_rate, 1.0))
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "channel": ch,
                "traffic": float(round(traffic)),
                "conversion_rate": round(conv_rate, 4),
                "signups": float(signups),
            })
    return pd.DataFrame(rows)

#INCIDENT INJECTION --------------------------------------------------------------------------------------------------

def inject_incident(df: pd.DataFrame, incident: Incident) -> pd.DataFrame:
    df = df.copy()
    start = pd.to_datetime(incident.start_date) #date when incident starts
    mask_date = pd.to_datetime(df["date"]) >= start #boolean mask for dates after incident start date

    if incident.incident_type == "traffic_collapse":
        # specific channel traffic drops sharply 
        #could be caused by a campaign pause, SEO penalty or referral partner pulling out, etc. 
        ch = incident.affected_channel 
        mask = mask_date & (df["channel"] == ch) 
        df.loc[mask, "traffic"] = df.loc[mask, "traffic"] * (1 - incident.magnitude) 
        df.loc[mask, "signups"] = np.round(
            df.loc[mask, "traffic"] * df.loc[mask, "conversion_rate"]
        )

    elif incident.incident_type == "conversion_bug":
        # traffic is normal but the conversation rate is lower in one specific channel 
        #could be caused by a broken signup form or checkout bug 
        ch = incident.affected_channel
        mask = mask_date & (df["channel"] == ch) 
        df.loc[mask, "conversion_rate"] = df.loc[mask, "conversion_rate"] * (1 - incident.magnitude)
        df.loc[mask, "signups"] = np.round(
            df.loc[mask, "traffic"] * df.loc[mask, "conversion_rate"]
        )

    elif incident.incident_type == "sitewide_bug":
        # conversion drops across all channels at once 
        #could be caused by global outage, broken deploy affecting whole signup flow 
        mask = mask_date
        df.loc[mask, "conversion_rate"] = df.loc[mask, "conversion_rate"] * (1 - incident.magnitude)
        df.loc[mask, "signups"] = np.round(
            df.loc[mask, "traffic"] * df.loc[mask, "conversion_rate"]
        )

    elif incident.incident_type == "pricing_change":
        # conversion drops gradually across all channels like from a price increase
        # softens demand rather than causing an abrupt cliff
        mask = mask_date
        days_since = (pd.to_datetime(df["date"]) - start).dt.days
        ramp = np.clip(days_since / 14, 0, 1)  # ramps in over 2 weeks
        df.loc[mask, "conversion_rate"] = df.loc[mask, "conversion_rate"] * (
            1 - incident.magnitude * ramp[mask]
        )
        df.loc[mask, "signups"] = np.round(
            df.loc[mask, "traffic"] * df.loc[mask, "conversion_rate"]
        )

    elif incident.incident_type == "spend_cut":
        # paid traffic drops because budget was cut 
        # in data looks identical to traffic_collapse in the data in "paid" channel specifically 
        mask = mask_date & (df["channel"] == "paid")
        df.loc[mask, "traffic"] = df.loc[mask, "traffic"] * (1 - incident.magnitude)
        df.loc[mask, "signups"] = np.round(
            df.loc[mask, "traffic"] * df.loc[mask, "conversion_rate"]
        )

    else:
        raise ValueError(f"Unknown incident type: {incident.incident_type}")

    return df

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#make scenario that takes everything and combines

def make_scenario(scenario_id: str, incident: Incident, dataset_start_date="2026-01-01", n_days=90, seed = 0):
    baseline = generate_baseline(dataset_start_date, n_days=n_days, seed=seed)
    injected = inject_incident(baseline, incident)

    metadata = {
        "scenario_id": scenario_id,
        "ground_truth": asdict(incident),
        "n_days": n_days,
        "start_date": dataset_start_date,
    }

    return injected, metadata

#save scenario function

def save_scenario(scenario_id: str, df: pd.DataFrame, metadata: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True) 
    df.to_csv(out_dir / f"{scenario_id}.csv", index=False)  #save dataframe as csv
    with open(out_dir / f"{scenario_id}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4) #save metadata as json

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#MAINBLOCK 

#definitions of 10 defined incidents
# need to fix percent signs in strings  

SCENARIOS = [
    Incident("traffic_collapse", "2026-02-15", "referral", 0.7,
             "Referral partner pulled their integration, referral traffic collapsed ~70 percent"),
    Incident("conversion_bug", "2026-02-10", "paid", 0.6,
             "Broken signup form on the paid landing page tanked paid conversion ~60 percent"),
    Incident("sitewide_bug", "2026-02-20", None, 0.35,
             "Global signup flow outage cut conversion ~35 percent across every channel"),
    Incident("pricing_change", "2026-02-05", None, 0.25,
             "Price increase gradually softened conversion ~25 percent across all channels over 2 weeks"),
    Incident("spend_cut", "2026-02-18", "paid", 0.8,
             "Marketing cut paid ad spend, paid traffic dropped ~80 percent"),
    Incident("traffic_collapse", "2026-02-12", "organic", 0.4,
             "Search algorithm update hit organic traffic, down ~40 percent"),
    Incident("conversion_bug", "2026-02-22", "organic", 0.5,
             "A/B test misconfiguration broke organic conversion ~50 percent"),
    Incident("sitewide_bug", "2026-02-08", None, 0.5,
             "Payment processor outage cut conversion ~50 percent sitewide"),
    Incident("pricing_change", "2026-02-25", None, 0.3,
             "New pricing tier confused users, conversion softened ~30 percent over 2 weeks"),
    Incident("spend_cut", "2026-02-14", "paid", 0.5,
             "Q1 budget freeze cut paid spend, paid traffic down ~50 percent"),
]

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "data" / "scenarios"
    for i, incident in enumerate(SCENARIOS, start=1):
        scenario_id = f"scenario_{i:02d}"
        df, meta = make_scenario(scenario_id, incident, seed=i)
        save_scenario(scenario_id, df, meta, out_dir)
        print(f"Saved {scenario_id}: {incident.incident_type} "
              f"({incident.affected_channel or 'all channels'}) starting {incident.start_date}")
    print(f"\nDone. {len(SCENARIOS)} scenarios saved to {out_dir}")