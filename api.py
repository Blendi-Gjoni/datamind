import os
import json
import tempfile
from pathlib import Path
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_tools.data_pipeline import DataPipeline
from data_tools.data_store import DataStore
from agents.orchestrator import build_orchestrator

app = FastAPI(title="Data Analyser API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DB_PATH    = str(BASE_DIR / "store.db")
DATA_DIR   = str(BASE_DIR / "data/")
API_KEY    = os.getenv("OPENAI_API_KEY")

pipeline = DataPipeline()
store = DataStore(db_path=DB_PATH, data_dir=DATA_DIR)
orchestrator = build_orchestrator(
    db_path=DB_PATH,
    data_dir=DATA_DIR,
    openai_api_key=API_KEY,
)


class AnalyseRequest(BaseModel):
    dataset_id: str
    question: str


class StepInfo(BaseModel):
    name: str
    success: bool
    duration_ms: float
    error: str | None = None


class AnalyseResponse(BaseModel):
    success: bool
    dataset_id: str
    question: str
    insight: str | None
    chart: dict | None
    steps: list[StepInfo]
    error: str | None = None

@app.get("/")
def serve_ui():
    return FileResponse("index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()

    # Try common encodings before giving up
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            contents.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise HTTPException(status_code=422, detail="Could not decode CSV — unsupported encoding.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        dataset_id = pipeline.ingest_csv(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    entry = pipeline.datasets[dataset_id]
    store.save(dataset_id, entry["dataframe"], entry["schema"], entry["metadata"])

    return {
        "dataset_id": dataset_id,
        "filename":   file.filename,
        "schema":     entry["schema"],
        "metadata":   entry["metadata"],
    }

@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = orchestrator.run(dataset_id=req.dataset_id, question=req.question)

    chart_dict = None
    if result.figure is not None:
        fig_dict = result.figure.to_dict()

        for trace in fig_dict.get("data", []):
            for key, value in trace.items():
                if isinstance(value, np.ndarray):
                    trace[key] = value.tolist()

                elif isinstance(value, dict) and "bdata" in value and "dtype" in value:
                    arr = np.frombuffer(
                        __import__("base64").b64decode(value["bdata"]),
                        dtype=np.dtype(value["dtype"])
                    )
                    trace[key] = arr.tolist()

        chart_dict = json.loads(json.dumps(fig_dict))
        print(chart_dict)

    steps = [
        StepInfo(
            name=s.name,
            success=s.success,
            duration_ms=round(s.duration_ms, 1),
            error=s.error,
        )
        for s in result.steps
    ]

    return AnalyseResponse(
        success=result.success,
        dataset_id=req.dataset_id,
        question=req.question,
        insight=result.insight,
        chart=chart_dict,
        steps=steps,
        error=result.error,
    )

@app.get("/datasets")
def list_datasets():
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT dataset_id, metadata, created_at FROM datasets ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "dataset_id": r[0],
            "metadata":   json.loads(r[1]),
            "created_at": r[2],
        }
        for r in rows
    ]