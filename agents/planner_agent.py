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
    "chart_type": "<bar|line|scatter|pie|histogram|none>",
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
- filter_rows:     { "column": str, "op": "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains", "value": any }
- group_aggregate: { "group_by": [str], "agg_column": str, "agg_fn": "sum" | "mean" | "count" | "min" | "max" }
- sort_rows:       { "column": str, "ascending": bool }
- select_columns:  { "columns": [str] }
- top_n:           { "column": str, "n": int, "ascending": bool }
- value_counts:    { "column": str }
- pivot_table:     { "index": str, "columns": str, "values": str, "agg_fn": str }
- date_resample:   { "date_column": str, "freq": "D" | "W" | "ME" | "QE" | "YE", "agg_column": str, "agg_fn": str }
- melt_columns:    { "id_cols": [str], "value_cols": [str], "var_name": str, "value_name": str }

COLUMN RULES — read these first, they override everything else:
- ONLY use column names that appear exactly in the schema. Do not infer, guess, abbreviate, or rename any column.
- Never reference a column called "year", "month", "date", "time", or any derived name unless it appears verbatim in the schema.
- If a column you need does not exist in the schema, find the closest real column or set chart_type to "none".
- Before writing any step or visualization, mentally verify every column name against the schema.

QUERY RULES:
- Steps execute in order. Each step receives the output of the previous step.
- If no transformation is needed, use an empty steps list: [].
- Each step must use exactly this format: {"tool": "<tool_name>", "params": {...}}. Never use {"<tool_name>": {...}} flat format.
- Every step in the steps list must have a valid tool name from the available tools list. Never include a step with a null, empty, or invented tool name. If no transformation is needed, use an empty steps list [] instead.
- For aggregation (sum, mean, min, max), ONLY use columns typed "numeric" in the schema. Never aggregate "datetime" or "string" columns.
- For group_by, use "string" typed columns. Never group by "datetime" or "numeric" columns directly.
- For top N questions, ALWAYS group_aggregate first to get totals per group, then apply top_n. Never apply top_n on raw unaggregated data.
- For filtering to a single group then aggregating (e.g. "average profit in Electronics"), use filter_rows first, then group_aggregate with an empty group_by list [].
- For time series questions, use date_resample on a "datetime" column. Frequencies: "D" daily, "W" weekly, "ME" monthly, "QE" quarterly, "YE" yearly. Never use "M", "Y", or "Q".
- For wide-format datasets where the same variable spans multiple columns (e.g. pop1980, pop2000, pop2010), use melt_columns first to convert to long format before aggregating or plotting.
- For distribution questions, use value_counts for string columns or histogram chart_type for numeric columns.
- For comparison questions across two string dimensions, consider pivot_table.
- For questions that only need sorting and selecting (e.g. "show me the 10 most expensive products"), use sort_rows then top_n — no group_aggregate needed.

VISUALIZATION RULES:
- ALWAYS set x and y to real column names from the schema or from the output of your query steps. Never leave x or y as null when chart_type is not "none".
- If you cannot confidently determine x and y, set chart_type to "none" — do not guess.
- After melt_columns, x is the var_name column and y is the value_name column.
- chart_type selection guide:
  - bar: comparing totals or counts across categories
  - line: trends over time or ordered sequences
  - scatter: correlation between two numeric columns
  - pie: share or proportion of a whole (use only when there are fewer than 8 categories)
  - histogram: distribution of a single numeric column
  - none: when the result is a single number, a table with many columns, or visualization adds no value
- Do not use pie if there are more than 7 distinct values — use bar instead.
- color_by should only be set if it meaningfully segments the data and the column is a "string" type with fewer than 10 distinct values. Otherwise set it to null.

INSIGHT RULES:
- focus should describe specifically what pattern, ranking, or trend the insight agent should explain — not just restate the question.
- highlight_columns should list only the columns directly relevant to the answer, maximum 3.

Output ONLY the JSON object. No explanation, no markdown fences, no commentary.
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

        raw = response.choices[0].message.content

        return self.parse_and_validate(raw, schema)

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