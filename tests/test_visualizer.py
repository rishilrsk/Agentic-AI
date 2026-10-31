"""
Unit tests for Phase 7: Plotly Visualization Engine.
"""

import pandas as pd
import plotly.graph_objects as go

from src.visualizer import build_plotly_chart


def test_build_bar_chart():
    df = pd.DataFrame({"category": ["A", "B", "C"], "revenue": [100, 200, 150]})
    fig = build_plotly_chart(df, chart_type="bar", x_col="category", y_col="revenue")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Data Analysis Visualization"


def test_build_line_chart():
    df = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "sales": [10, 25, 40]})
    fig = build_plotly_chart(df, chart_type="line", x_col="month", y_col="sales")
    assert isinstance(fig, go.Figure)


def test_build_pie_chart():
    df = pd.DataFrame({"region": ["North", "South"], "orders": [50, 80]})
    fig = build_plotly_chart(df, chart_type="pie", x_col="region", y_col="orders")
    assert isinstance(fig, go.Figure)


def test_empty_dataframe_chart():
    fig = build_plotly_chart(pd.DataFrame(), chart_type="bar")
    assert fig is None
