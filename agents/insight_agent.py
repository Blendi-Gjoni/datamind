import pandas as pd
from openai import OpenAI
from .query_agent import QueryAgent
from .visualization_agent import VisualizationAgent

INSIGHT_SYSTEM_PROMPT = """
You are a data analyst writing a concise insight for a non-technical audience.
You receive a summary of a dataset query result and must explain what it means
in 2-4 sentences. Be specific — mention actual values, trends, or comparisons.
Do not use jargon. Do not repeat the question. Just give the insight.
""".strip()

class InsightAgent:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def run(self, df: pd.DataFrame, insight_plan: dict, original_question: str) -> dict:
        focus = insight_plan.get("focus", "summarize the data")
        highlight_cols = insight_plan.get("highlight_columns", [])

        summary = self.build_summary(df, highlight_cols)

        user_message = (
            f"Original question: {original_question}\n\n"
            f"Focus: {focus}\n\n"
            f"Data summary:\n{summary}"
        )

        response = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        insight_text = response.choices[0].message.content.strip()

        return {
            "insight": insight_text,
            "focus": focus,
            "rows_analyzed": len(df),
        }

    def build_summary(self, df: pd.DataFrame, highlight_cols: list) -> str:
        lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}"]

        cols_to_summarize = highlight_cols if highlight_cols else list(df.columns)[:5]

        for col in cols_to_summarize:
            if col not in df.columns:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                lines.append(
                    f"{col}: min={series.min():.2f}, max={series.max():.2f}, "
                    f"mean={series.mean():.2f}, sum={series.sum():.2f}"
                )
            else:
                top = series.value_counts().head(5).to_dict()
                lines.append(f"{col} (top values): {top}")

        lines.append("\nSample rows:")
        lines.append(df.head(10).to_string(index=False))

        return "\n".join(lines)

"""
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    import os

    api_key = os.getenv("OPENAI_API_KEY")

    # Simulate a planner output
    plan = {
        "query": {
            "steps": [
                {
                    "tool": "group_aggregate",
                    "params": {"group_by": ["category"], "agg_column": "sales", "agg_fn": "sum"}
                },
                {
                    "tool": "top_n",
                    "params": {"column": "sales", "n": 5, "ascending": False}
                }
            ]
        },
        "visualization": {
            "chart_type": "bar",
            "x": "category",
            "y": "sales",
            "color_by": None,
            "title": "Top 5 categories by sales"
        },
        "insight": {
            "focus": "Which categories dominate sales and by how much",
            "highlight_columns": ["category", "sales"]
        }
    }

    # Dummy dataset
    df = pd.DataFrame({
        "category": ["Electronics", "Clothing", "Electronics", "Books", "Clothing",
                     "Electronics", "Toys", "Books", "Toys", "Clothing"],
        "sales": [450, 120, 890, 60, 340, 210, 95, 80, 110, 270],
        "region": ["North"] * 5 + ["South"] * 5,
    })

    question = "What are the top 5 product categories by total sales?"

    # Run each agent
    query_agent = QueryAgent()
    query_result = query_agent.run(df, plan["query"])
    print("Query result:")
    print(query_result["dataframe"])

    viz_agent = VisualizationAgent()
    viz_result = viz_agent.run(query_result["dataframe"], plan["visualization"])
    print(f"\nChart type: {viz_result['chart_type']}")
    if viz_result["figure"]:
        viz_result["figure"].show()

    insight_agent = InsightAgent(api_key=api_key)
    insight_result = insight_agent.run(query_result["dataframe"], plan["insight"], question)
    print(f"\nInsight:\n{insight_result['insight']}")"""
