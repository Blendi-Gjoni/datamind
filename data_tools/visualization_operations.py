import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def make_chart(df: pd.DataFrame, viz_plan: dict) -> go.Figure | None:
    chart_type = viz_plan.get("chart_type", "none")
    x = viz_plan.get("x")
    y = viz_plan.get("y")
    color = viz_plan.get("color_by")
    title = viz_plan.get("title", "")

    if chart_type == "none" or not x or not y:
        return None

    kwargs = dict(data_frame=df, x=x, y=y, title=title)
    if color and color in df.columns:
        kwargs["color"] = color

    chart_map = {
        "bar": px.bar,
        "line": px.line,
        "scatter": px.scatter,
        "histogram": lambda **kw: px.histogram(kw["data_frame"], x=kw["x"], title=kw["title"]),
        "pie": lambda **kw: px.pie(kw["data_frame"], names=kw["x"], values=kw["y"], title=kw["title"]),
    }

    if chart_type not in chart_map:
        raise ValueError(f"Unknown chart type: '{chart_type}'")

    fig = chart_map[chart_type](**kwargs)
    fig.update_layout(margin=dict(t=50, b=20, l=10, r=20))
    return fig