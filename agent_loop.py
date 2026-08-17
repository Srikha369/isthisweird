from dotenv import load_dotenv
load_dotenv()

import anthropic
import json
import stat_tools


client = anthropic.Anthropic()  # assumes ANTHROPIC_API_KEY is set as an env var
MODEL = "claude-sonnet-4-6"
MAX_TOOL_CALLS = 5

TOOLS = [
    {
        "name": "segment_comparison_tool",
        "description": (
            "Compares signups and conversion rate across channels (organic, paid, referral) "
            "before vs. after a given date. Use this FIRST to check whether an anomaly is "
            "concentrated in one channel or affects all channels equally, and whether it's a "
            "traffic problem (signups drop, conversion stable) or a funnel problem "
            "(conversion drops, traffic stable)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "split_date": {
                    "type": "string",
                    "description": "Candidate date (YYYY-MM-DD) to split the data into before/after periods",
                }
            },
            "required": ["split_date"],
        },
    },
    {
        "name": "detect_changepoints",
        "description": (
            "Scans a time series to find the date with the strongest statistical break in trend. "
            "Use this to pin down WHEN an anomaly started, optionally scoped to one channel."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_col": {
                    "type": "string",
                    "enum": ["signups", "conversion_rate"],
                    "description": "Which metric to scan for a changepoint",
                },
                "channel": {
                    "type": ["string", "null"],
                    "description": "Restrict to one channel (organic/paid/referral), or null for all channels combined",
                },
            },
            "required": ["metric_col"],
        },
    },
    {
        "name": "check_event_alignment",
        "description": (
            "Checks whether a candidate date lines up with a known business event (marketing "
            "spend change, deploy, pricing change). Use this to distinguish between an unexplained "
            "anomaly and one that's caused by a known internal action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_date": {"type": "string", "description": "Date (YYYY-MM-DD) to check"},
                "channel": {
                    "type": ["string", "null"],
                    "description": "Channel to check for channel-specific events, or null",
                },
            },
            "required": ["candidate_date"],
        },
    },
]

TOOL_FUNCTIONS = {
    "segment_comparison_tool": lambda df, **kwargs: stat_tools.segment_comparison_tool(df, **kwargs),
    "detect_changepoints": lambda df, **kwargs: stat_tools.detect_changepoints(df, **kwargs),
    "check_event_alignment": lambda df, **kwargs: stat_tools.check_event_alignment(**kwargs),
}

SYSTEM_PROMPT = """You are an automated data investigator. A metric anomaly has been \
reported. Your job is to determine the most likely root cause using the tools available.

Investigation process:
1. Start by running segment_comparison_tool to see if the anomaly is isolated to one \
channel or affects all channels, and whether it's a traffic or conversion issue.
2. Use detect_changepoints to pin down exactly when the anomaly started, scoped to \
whichever channel(s) segment_comparison_tool flagged.
3. Use check_event_alignment on that date to see if a known business event explains it.
4. Once you have enough evidence, STOP calling tools and give your final diagnosis.

Your final diagnosis must state: the most likely root cause, your confidence level \
(high/medium/low), and the specific evidence from the tools that supports it. If the \
evidence is ambiguous or contradictory, say so honestly rather than guessing."""


def investigate(df, anomaly_description: str, max_tool_calls: int = MAX_TOOL_CALLS) -> dict:
    messages = [
        {"role": "user", "content": f"Anomaly reported: {anomaly_description}. Investigate and diagnose the root cause."}
    ]

    tool_call_log = []  # keep a record of what the agent did, for the eval harness later

    for _ in range(max_tool_calls):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't ask for a tool, it's done investigating — stop here.
        if response.stop_reason != "tool_use":
            break

        # Otherwise, execute every tool call the model requested this turn,
        # and feed the results back so it can decide what to do next.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                
                try:
                    result = TOOL_FUNCTIONS[tool_name](df, **tool_input)
                except Exception as e:
                    result = {"error": str(e)}

                tool_call_log.append({"tool": tool_name, "input": tool_input, "output": result})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        messages.append({"role": "user", "content": tool_results})

    else:
        # Loop exhausted max_tool_calls without the model concluding on its own —
        # force a final answer rather than silently returning nothing.
        messages.append({
            "role": "user",
            "content": "You've reached the maximum number of tool calls. Give your final diagnosis now based on the evidence gathered so far."
        })
        response = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages
        )

    final_text = "".join(block.text for block in response.content if block.type == "text")

    return {
        "diagnosis": final_text,
        "tool_calls_made": tool_call_log,
        "num_tool_calls": len(tool_call_log),
    }

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    data_dir = Path(__file__).parent / "data" / "scenarios"
    df = pd.read_csv(data_dir / "scenario_05.csv")  # spend_cut on paid — the trap case

    result = investigate(df, anomaly_description="Total signups dropped noticeably in mid-to-late February")
    print(result["diagnosis"])
    print("\nTool calls made:", result["num_tool_calls"])
    for call in result["tool_calls_made"]:
        print(" -", call["tool"], call["input"])

