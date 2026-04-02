import json
from openai import OpenAI

PLANNER_SYSTEM_PROMPT = """
You are a data analysis planner. Given a user question and a dataset schema, 
you produce a structured JSON execution plan for a pipeline of specialized agents.

The schema describes column names and their types: "numeric", "datetime", or "string".

Your output must be a valid JSON object with exactly this structure:
{
  "query": {
    "steps": [
      {
        "tool": "<tool_name>",
        "params": { ... }
      }
    ]
  },
  "visualization": {
    "chart_type": "<bar|line|scatter|pie|histogram|heatmap|none>",
    "x": "<column_name or null>",
    "y": "<column_name or null>",
    "color_by": "<column_name or null>",
    "title": "<chart title>"
  },
  "insight": {
    "focus": "<what the insight agent should explain>",
    "highlight_columns": ["<col1>", "<col2>"]
  }
}

Available query tools and their params:
- filter_rows:       { "column": str, "op": "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains", "value": any }
- group_aggregate:   { "group_by": [str], "agg_column": str, "agg_fn": "sum" | "mean" | "count" | "min" | "max" }
- sort_rows:         { "column": str, "ascending": bool }
- select_columns:    { "columns": [str] }
- top_n:             { "column": str, "n": int, "ascending": bool }
- value_counts:      { "column": str }
- pivot_table:       { "index": str, "columns": str, "values": str, "agg_fn": str }
- date_resample:     { "date_column": str, "freq": "D" | "W" | "M" | "Y", "agg_column": str, "agg_fn": str }

Rules:
- Steps are executed in order; each step receives the output of the previous step.
- Only reference columns that exist in the schema.
- If no transformation is needed, use an empty steps list: [].
- If visualization is not meaningful, set chart_type to "none".
- Output ONLY the JSON object, no explanation, no markdown fences.
""".strip()

class PlannerAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def plan(self, question: str, schema: dict, metadata: dict = None) -> dict:
        user_message = self.build_user_message(question, schema, metadata)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].messsage.content

        return parse_and_validate(raw, schema)

    def build_user_message(self, question: str, schema: dict, metadata: dict) -> str:
        schema_str = json.dumps(schema, indent=2)
        meta_str = ""
        if metadata:
            meta_str = f"\n\nDataset metadata:\n{json.dumps(metadata, indent=2)}"
        return (
            f"Question: {question}\n\n"
            f"Schema: {schema_str}\n\n"
            f"{meta_str}"
        )

    def parse_and_validate(self, raw: str, schema: dict) -> dict:
        try:
            plan = json.loads(raw)
        except json.decoder.JSONDecodeError as e:
            raise ValueError(f"Planner returned invalid JSON: {e}\n\nRaw output:\n{raw}")

        required_keys = {"query", "visualization", "insight"}
        missing = required_keys - plan.keys()
        if missing:
            raise ValueError(f"Plan is missing required keys: {missing}")
        if "steps" not in plan["query"]:
            raise ValueError(f"Plan 'query' section must contain 'steps'.")

        known_columns = set(schema.keys())
        for step in plan["query"]["steps"]:
            for param_key in ("column", "agg_column", "group_by", "index", "columns", "values", "date_column"):
                val = step.get("params", {}).get(param_key)
                if val is None:
                    continue
                cols = val if isinstance(val, list) else [val]
                for col in cols:
                    if col not in known_columns:
                        print(f"[PlannerAgent] Warning: column '{col}' not found in schema.")
        return plan