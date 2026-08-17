
# Is This Weird?

An autonomous agent that diagnoses the root cause of KPI anomalies. Automates the manual investigation work data teams do every 
time a dashboard metric moves unexpectedly.

## The problem

Everytime a metric drops unexpectedly on a dashboard, without an agent, 
someone has to manually dig through segments, timing, and recent changes
 to figure out why. This investigative pattern is fairly predictable, 
 allowing us to explore how advanced automation can get.  
 This project builds an agent that performs that investigation 
autonomously. Given an anomalous metric, it forms hypotheses, tests 
each one using real statistical methods, and 
produces a confidence-rated diagnosis with supporting evidence.

## How it works

Given an anomaly, the agent runs an investigation loop (up to 5 tool 
calls) using three statistical tools. 

1. **Segment comparison** — Welch's t-test comparing each channel's 
   signups and conversion rate before vs. after a candidate date, to 
   determine whether the anomaly is isolated to one channel and 
   whether it's a traffic problem or a conversion/funnel problem.

2. **Changepoint detection** — scans the time series for the date 
   that produces the strongest statistical break between "before" and 
   "after" segments, to pinpoint exactly when the anomaly started.

3. **Event alignment** — checks whether the detected changepoint lines 
   up with a known business event (a marketing spend change, a deploy, 
   a pricing update), to distinguish an anomaly with a known internal 
   cause from a genuinely unexplained one.

The LLM (Claude) decides which to call, in what order, 
and when it has enough evidence to conclude. Meanwhile,  
the actual statistics are computed deterministically in Python, instead of being inferred by the model. 

So it can be called as a service, A FastAPI wrapper exposes the agent as a `POST /investigate` endpoint. 


## Evaluation

Tested against 10 synthetic scenarios, each with a different injected 
root cause and known ground truth (traffic collapse, conversion bug, 
sitewide outage, gradual pricing change, or marketing spend cut).

On average, **8-10 / 10** of the scenarios were correctly diagnosed.

Full per-scenario results are in `eval_results.csv`.


## Example investigation

**Anomaly reported:** Sudden signups drop reported.


---

**Agent's process:**
1. **Changepoint detection** | `detect_changepoints` (signups) | Statistically significant break on **Feb 14, 2026** (p=0.0042, t=2.94) |
2.  **Channel segmentation** | `segment_comparison_tool` (split: Feb 14) | **Referral signups dropped −71%** (highly significant, p≈0); organic (−2.5%) and paid (+1.6%) are flat |
3. **Traffic vs. funnel** | `segment_comparison_tool` (conversion rates) | Conversion rates are **stable across all channels** — this is a **traffic volume problem, not a funnel/UX problem** |
4. **Event alignment** | `check_event_alignment` (Feb 14, global) | A **Q1 budget freeze cut paid spend by ~50%** on the exact same date |


**Agent's Diagnosis:** 

The signup drop is **entirely explained by a collapse in referral channel traffic** starting February 14. Crucially:
- **Organic and paid channels are unaffected** in signup volume
- **Conversion rates are stable**, ruling out a broken signup flow, pricing change, or UX regression
- The **Q1 paid budget freeze** coincides precisely with the changepoint. While the paid channel itself didn't drop (possibly due to efficiency gains), the referral program likely depends on paid traffic as a seeding mechanism — fewer paid users entering the funnel means fewer users to refer others

- **High confidence** that the referral channel is the source of the drop and that it's a traffic problem
- **Medium confidence** on the paid budget freeze as the *cause* of the referral collapse — the correlation is exact (same date), but no direct referral-specific event was logged. It's possible a **referral program partner outage, link breakage, or partner campaign ending** is the true trigger and simply went unlogged



**Ground truth:** "Referral partner pulled their integration, referral traffic collapsed ~70 percent in mid February." -- True. 



## Running it

\`\`\`bash
conda create -n anomaly-agent python=3.12
conda activate anomaly-agent
pip install -r requirements.txt

### Add your Anthropic API key to a .env file:
echo "ANTHROPIC_API_KEY=your-key-here" > .env

### Generate the synthetic scenarios
python synthetic_generator.py

### Run the agent on a single scenario
python agent_loop.py

### Run the full evaluation
python eval.py

### Or start the API
uvicorn isthisweirdapp:app --reload
\`\`\`

## Known limitations

- Changepoint detection struggles with gradual (non-sharp) trend shifts
- The events log is hand-curated and synthetic; a production version 
  would need to pull from real systems (deploy logs, ad platform APIs)
- Grading in the eval harness uses keyword matching on the agent's 
  free-text diagnosis rather than structured output. A stricter eval would have the agent return a 
  structured JSON verdict instead of prose





