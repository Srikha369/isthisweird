import pandas as pd
import json
from pathlib import Path
from agent_loop import investigate

DATA_DIR = Path(__file__).parent / "data" / "scenarios"

def load_all_scenarios():
    #loads all scenarios regardles of how many there are as the dataframe and metadata
    scenarios = []
    for meta_file in sorted(DATA_DIR.glob("*_metadata.json")):
        scenario_id = meta_file.stem.replace("_metadata", "")
        df = pd.read_csv(DATA_DIR / f"{scenario_id}.csv")
        with open(meta_file) as f:
            meta = json.load(f)
        scenarios.append((scenario_id, df, meta))
    return scenarios

# Maps ground-truth incident_type values to the words/phrases we'd expect
# a correct diagnosis to use, since the agent won't necessarily use our
# exact internal label names.
INCIDENT_TYPE_KEYWORDS = {
    "traffic_collapse": ["traffic", "collapse", "drop in traffic", "traffic decline"],
    "conversion_bug": ["conversion", "bug", "funnel", "form", "checkout"],
    "sitewide_bug": ["sitewide", "site-wide", "global", "outage", "all channels"],
    "pricing_change": ["pricing", "price"],
    "spend_cut": ["spend", "budget", "ad spend", "marketing spend"],
}

def grade_diagnosis(diagnosis_text: str, ground_truth: dict) -> dict:
    text_lower = diagnosis_text.lower()

    incident_type = ground_truth["incident_type"]
    affected_channel = ground_truth["affected_channel"]

    # Did the diagnosis mention the right kind of cause?
    keywords = INCIDENT_TYPE_KEYWORDS[incident_type]
    type_correct = any(kw in text_lower for kw in keywords)

    # Did it mention the right channel? (sitewide incidents have no
    # specific channel, so we skip this check for those.)
    if affected_channel is None:
        channel_correct = True  # nothing specific to get right
    else:
        channel_correct = affected_channel.lower() in text_lower

    return {
        "type_correct": type_correct,
        "channel_correct": channel_correct,
        "fully_correct": type_correct and channel_correct,
    }


def run_eval(anomaly_description_template="Signups appear to have dropped noticeably in February"):
    scenarios = load_all_scenarios()
    results = []

    for scenario_id, df, meta in scenarios:
        print(f"Investigating {scenario_id}...")
        ground_truth = meta["ground_truth"]

        outcome = investigate(df, anomaly_description=anomaly_description_template)
        grade = grade_diagnosis(outcome["diagnosis"], ground_truth)

        results.append({
            "scenario_id": scenario_id,
            "true_incident_type": ground_truth["incident_type"],
            "true_channel": ground_truth["affected_channel"],
            "num_tool_calls": outcome["num_tool_calls"],
            "tools_used": [c["tool"] for c in outcome["tool_calls_made"]],
            "diagnosis": outcome["diagnosis"],
            **grade,
        })

    return pd.DataFrame(results)

def summarize(results: pd.DataFrame):
    total = len(results)
    fully_correct = results["fully_correct"].sum()
    type_correct = results["type_correct"].sum()

    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY: {fully_correct}/{total} fully correct "
          f"({fully_correct/total*100:.0f}%)")
    print(f"Incident type correct (channel may be wrong): {type_correct}/{total}")
    print(f"{'='*50}\n")

    print("Per-scenario breakdown:")
    for _, row in results.iterrows():
        status = "✓" if row["fully_correct"] else "✗"
        print(f"  {status} {row['scenario_id']}: true={row['true_incident_type']}/"
              f"{row['true_channel']}, tool_calls={row['num_tool_calls']}, "
              f"tools={row['tools_used']}")

    print("\nMissed scenarios (worth reading the full diagnosis for these):")
    for _, row in results[~results["fully_correct"]].iterrows():
        print(f"\n--- {row['scenario_id']} ---")
        print(f"True cause: {row['true_incident_type']} on {row['true_channel']}")
        print(f"Agent said: {row['diagnosis'][:300]}...")

if __name__ == "__main__":
    results = run_eval()
    summarize(results)

    out_path = Path(__file__).parent / "eval_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


