"""
Plotly Visualization Engine.
Generates polished, responsive, and publication-ready analytical charts.
"""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.logger import logger


def build_plotly_chart(
    df: pd.DataFrame,
    chart_type: str = "bar",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: str = "Data Analysis Visualization",
) -> Optional[go.Figure]:
    """
    Build an interactive Plotly figure based on query results DataFrame and parameters.
    """
    if df is None or df.empty:
        logger.warning("Cannot generate chart from empty DataFrame.")
        return None

    cols = list(df.columns)
    if len(cols) == 0:
        return None

    # Autodetect columns if missing
    if not x_col or x_col not in cols:
        x_col = cols[0]
    if not y_col or y_col not in cols:
        y_col = cols[1] if len(cols) > 1 else cols[0]

    norm_type = str(chart_type).lower().strip()

    try:
        if norm_type == "bar":
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=title,
                text_auto=".2s",
                color_discrete_sequence=["#3366FF"],
            )
            fig.update_traces(marker_line_width=0, opacity=0.9)

        elif norm_type == "line":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                markers=True,
                color_discrete_sequence=["#00CC96"],
            )

        elif norm_type == "scatter":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=title,
                size_max=15,
                color_discrete_sequence=["#AB63FA"],
            )

        elif norm_type == "pie":
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                title=title,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )

        else:
            # Default to bar
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#3366FF"],
            )

        # Apply modern responsive styling
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            title=dict(font=dict(size=16, family="Inter, sans-serif")),
            xaxis=dict(title=x_col.replace("_", " ").title()),
            yaxis=dict(title=y_col.replace("_", " ").title()),
            hoverlabel=dict(bgcolor="white", font_size=13),
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to generate Plotly chart: {str(e)}")
        return None
