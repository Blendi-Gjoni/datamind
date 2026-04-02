import pandas as pd
from data_tools.visualization_operations import make_chart

class VisualizationAgent:
    def run(self, df: pd.DataFrame, viz_plan: dict) -> dict:
        chart_type = viz_plan.get('chart_type', "none")

        if chart_type == "none":
             return {"figure": None, "chart_type": "none", "skipped": True}


        for key in ("x", "y"):
            col = viz_plan.get(key)
            if col and col not in df.columns:
                raise ValueError(
                    f"VisualizationAgent: column '{col}' not in dataframe. "
                    f"Available: {list(df.columns)}"
                )

        fig = make_chart(df, viz_plan)

        return {
            "figure": fig,
            "chart_type": chart_type,
            "x": viz_plan.get("x"),
            "y": viz_plan.get("y"),
            "title": viz_plan.get("title"),
            "skipped": fig is None,
        }