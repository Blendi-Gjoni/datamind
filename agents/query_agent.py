import pandas as pd
from data_tools.data_operations import QUERY_TOOLS

class QueryAgent:
    def run(self, df: pd.DataFrame, query_plan: dict) -> dict:
        steps = query_plan.get("steps", [])
        result_df = df.copy()
        applied = []

        for step in steps:
            tool_name = step.get("tool")
            params = step.get("params", {})

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