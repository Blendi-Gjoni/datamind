import time
import traceback
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import plotly.graph_objects as go
from data_tools.data_pipeline import DataPipeline
from data_tools.data_store import DataStore
from agents.planner_agent import PlannerAgent
from agents.query_agent import QueryAgent
from agents.visualization_agent import VisualizationAgent
from agents.insight_agent import InsightAgent

@dataclass
class StepResult:
    name: str
    success: bool
    duration_ms: float
    output: dict = field(default_factory=dict)
    error: str | None = None

@dataclass
class AnalysisResult:
    question: str
    dataset_id: str
    success: bool
    figure: go.Figure | None = None
    insight: InsightAgent | None = None
    steps: list[StepResult] = field(default_factory=list)
    error: str | None = None

    @property
    def plan(self) -> dict | None:
        for s in self.steps:
            if s.name == "planner" and s.success:
                return s.output.get("plan")
        return None

    @property
    def dataframe(self) -> pd.DataFrame | None:
        for s in self.steps:
            if s.name == "query" and s.success:
                return s.output.get("dataframe")
        return None

class Orchestrator:
    def __init__(
        self,
        store: DataStore,
        planner: PlannerAgent,
        query_agent: QueryAgent,
        viz_agent: VisualizationAgent,
        insight_agent: InsightAgent,
    ):
        self.store = store
        self.planner = planner
        self.query_agent = query_agent
        self.viz_agent = viz_agent
        self.insight_agent = insight_agent

    def run(self, dataset_id: str, question: str) -> AnalysisResult:
        result = AnalysisResult(
            question=question,
            dataset_id=dataset_id,
            success=False,
        )
        # Fetch schema
        step = self.run_step("schema_fetch", lambda: self.fetch_schema(dataset_id))
        result.steps.append(step)
        if not step.success:
            result.error = step.error
            return result
        schema = step.output["schema"]
        metadata = step.output["metadata"]

        # Plan
        step = self.run_step("planner", lambda: self.plan(question, schema, metadata))
        result.steps.append(step)
        if not step.success:
            result.error = step.error
            return result
        plan = step.output["plan"]

        """
        import json
        print(json.dumps(plan, indent=2))
        """

        #Fetch dataframe
        viz_cols = [plan["visualization"].get("x"), plan["visualization"].get("y"), plan["visualization"].get("color_by")]
        query_cols = self.extract_query_columns(plan["query"])
        needed_cols = list({c for c in viz_cols + query_cols if c}) or None
        step = self.run_step("dataframe_fetch", lambda: self._fetch_df(dataset_id, needed_cols))
        result.steps.append(step)
        if not step.success:
            result.error = step.error
            return result
        df = step.output["dataframe"]

        #Query
        step = self.run_step("query", lambda: self.query_agent.run(df, plan["query"]))
        result.steps.append(step)
        if not step.success:
            result.error = step.error
            return result
        transformed_df = step.output["dataframe"]

        #Visualization
        step = self.run_step("visualization", lambda: self.viz_agent.run(transformed_df, plan["visualization"]))
        result.steps.append(step)
        if step.success:
            result.figure = step.output.get("figure")
        """print(json.dumps(result.figure.to_json(), indent=2))"""

        #Insight
        step = self.run_step("insight", lambda: self.insight_agent.run(transformed_df, plan["insight"], question))
        result.steps.append(step)
        if step.success:
            result.insight = step.output.get("insight")

        result.success = True
        return result



    def run_step(self, name: str, fn) -> StepResult:
        t0 = time.perf_counter()
        try:
            output = fn()
            duration_ms = (time.perf_counter() - t0) * 1000
            return StepResult(name=name, success=True, duration_ms=duration_ms, output=output or {})
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            return StepResult(name=name, success=False, duration_ms=duration_ms, error=f"{type(e).__name__}: {e}")

    def fetch_schema(self, dataset_id: str) -> dict:
        schema, metadata = self.store.get_schema(dataset_id)
        return {"schema": schema, "metadata": metadata}

    def plan(self, question: str, schema: dict, metadata: dict) -> dict:
        plan = self.planner.plan(question=question, schema=schema, metadata=metadata)
        return {"plan": plan}

    def _fetch_df(self, dataset_id: str, columns: list | None) -> dict:
        df = self.store.get_dataframe(dataset_id, columns=columns)
        return {"dataframe": df}

    def extract_query_columns(self, query_plan: dict) -> list[str]:
        cols =[]
        for step in query_plan.get("steps", []):
            params = step.get("params", {})
            for key in ("column", "agg_column", "date_column", "index", "values"):
                if val := params.get(key):
                    cols.append(val)
            for key in ("group_by", "columns"):
                if val := params.get(key):
                    cols.extend(val if isinstance(val, list) else [val])
        return cols


def build_orchestrator(
        db_path: str = "store.db",
        data_dir: str = "data/",
        openai_api_key: str | None = None,
        model: str = "gpt-4o",
) -> Orchestrator:
    return Orchestrator(
        store=DataStore(db_path=db_path, data_dir=data_dir),
        planner=PlannerAgent(api_key=openai_api_key, model=model),
        query_agent=QueryAgent(),
        viz_agent=VisualizationAgent(),
        insight_agent=InsightAgent(api_key=openai_api_key, model=model),
    )

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    import os
    from pathlib import Path
    from data_tools.data_pipeline import DataPipeline
    from data_tools.data_store import DataStore

    api_key = os.getenv("OPENAI_API_KEY")

    # Absolute path so it works regardless of where you run from
    BASE_DIR = Path(__file__).parent.parent  # points to AI_Data_Analyser/
    csv_path = BASE_DIR / "agents" / "sample.csv"

    pipeline = DataPipeline()
    store    = DataStore(
        db_path  = str(BASE_DIR / "store.db"),
        data_dir = str(BASE_DIR / "data/"),
    )

    dataset_id = pipeline.ingest_csv(str(csv_path))
    entry      = pipeline.datasets[dataset_id]
    store.save(dataset_id, entry["dataframe"], entry["schema"], entry["metadata"])

    orchestrator = build_orchestrator(
        db_path        = str(BASE_DIR / "store.db"),
        data_dir       = str(BASE_DIR / "data/"),
        openai_api_key = api_key,
    )

    result = orchestrator.run(
        dataset_id = dataset_id,
        question   = "What are the top 5 product categories by total sales?",
    )

    print("\nPipeline steps:")
    for step in result.steps:
        status = "OK" if step.success else "FAIL"
        print(f"  [{status}] {step.name:<20} {step.duration_ms:>7.1f}ms"
              + (f"  — {step.error}" if step.error else ""))

    if result.insight:
        print(f"\nInsight:\n{result.insight}")

    if result.figure:
        result.figure.show()

    if not result.success:
        print(f"\nPipeline failed: {result.error}")