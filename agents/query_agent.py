import pandas as pd
from data_tools.data_operations import QUERY_TOOLS

class QueryAgent:
    def _normalise_step(self, step: dict) -> dict:
        if "tool" in step:
            return step
        tool_name = next((k for k in step), None)
        return {"tool": tool_name, "params": step.get(tool_name, {})}

    def run(self, df: pd.DataFrame, query_plan: dict) -> dict:
        steps = [self._normalise_step(s) for s in query_plan.get("steps", [])]
        result_df = df.copy()
        applied = []

        for step in steps:
            if "tool" in step:
                tool_name = step.get("tool")
                params = step.get("params", {})
            else:
                tool_name = next((k for k in step if k != "params"), None)
                params = step.get(tool_name, {}) if tool_name else {}

            if not tool_name:
                continue

            if tool_name not in QUERY_TOOLS:
                raise ValueError(f"Query Agent: unknown tool: {tool_name}")

            try:
                result_df = QUERY_TOOLS[tool_name](result_df, **params)
                applied.append({"tool": tool_name, "params": params, "rows": len(result_df)})
            except Exception as e:
                raise RuntimeError(f"Query Agent: {tool_name} failed: {e}") from e

        return {
            "dataframe": result_df,
            "steps_applied": applied,
            "row_count": len(result_df),
            "columns": list(result_df.columns),
        }