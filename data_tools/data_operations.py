import pandas as pd

def filter_rows(df: pd.DataFrame, column: str, op: str, value) -> pd.DataFrame:
    ops = {
        "==": lambda c, v: c == v,
        "!=": lambda c, v: c != v,
        ">": lambda c, v: c > v,
        "<": lambda c, v: c < v,
        ">=": lambda c, v: c >= v,
        "<=": lambda c, v: c <= v,
        "contains": lambda c, v: c.str.contains(str(v), case=False, na=False),
    }
    if op not in ops:
        raise ValueError(f"Unknown filter op '{op}'. Valid: {list(ops)}")
    return df[ops[op](df[column], value)].copy()

def group_aggregate(df: pd.DataFrame, group_by: str, agg_column: str, agg_fn: str) -> pd.DataFrame:
    valid_fns = {"sum", "mean", "count", "min", "max"}
    if agg_fn not in valid_fns:
        raise ValueError(f"Unknown aggregate function '{agg_fn}'. Valid: {valid_fns}")
    return df.groupby(group_by, as_index=False).agg({agg_column: agg_fn})

def sort_rows(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    return df.sort_values(column, ascending=ascending).reset_index(drop=True)

def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[columns].copy()

def top_n(df: pd.DataFrame, column: str, n: int, ascending: bool = True) -> pd.DataFrame:
    return df.nlargest(n, column) if not ascending else df.nsmallest(n, column)

def value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    vc = df[column].value_counts().reset_index()
    vc.columns = [column, "count"]
    return vc

def pivot_table(df: pd.DataFrame, index: str, columns: str, values: str, agg_fn: str) -> pd.DataFrame:
    return pd.pivot_table(df, index=index, columns=columns, values=values, agg_fn=agg_fn).reset_index()

def date_resample(df: pd.DataFrame, date_column: str, freq: str, agg_column: str, agg_fn: str) -> pd.DataFrame:
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    resampled = df.set_index(date_column).resample(freq)[agg_column].agg(agg_fn).reset_index()
    return resampled

QUERY_TOOLS = {
    "filter_rows":     filter_rows,
    "group_aggregate": group_aggregate,
    "sort_rows":       sort_rows,
    "select_columns":  select_columns,
    "top_n":           top_n,
    "value_counts":    value_counts,
    "pivot_table":     pivot_table,
    "date_resample":   date_resample,
}