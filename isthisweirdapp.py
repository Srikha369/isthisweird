from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
from pathlib import Path
from agent_loop import investigate

app = FastAPI(title="Is This Weird?")

DATA_DIR = Path(__file__).parent / "data" / "scenarios"

class InvestigateRequest(BaseModel):
    scenario_id: str
    anomaly_description: str = "Signups appear to have dropped noticeably"

class InvestigateResponse(BaseModel):
    scenario_id: str
    diagnosis: str
    num_tool_calls: int
    tools_used: list[str]


@app.post("/investigate", response_model=InvestigateResponse)
def investigate_endpoint(request: InvestigateRequest):
    csv_path = DATA_DIR / f"{request.scenario_id}.csv"
    if not csv_path.exists():
        return {"error": f"Unknown scenario_id: {request.scenario_id}"}

    df = pd.read_csv(csv_path)
    outcome = investigate(df, anomaly_description=request.anomaly_description)

    return InvestigateResponse(
        scenario_id=request.scenario_id,
        diagnosis=outcome["diagnosis"],
        num_tool_calls=outcome["num_tool_calls"],
        tools_used=[c["tool"] for c in outcome["tool_calls_made"]],
    )

@app.get("/health")
def health():
    return {"status": "ok"}






