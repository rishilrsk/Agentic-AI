"""
Unit tests for Phase 5: LLM Client and Prompt Engineering.
"""

from src.llm import (
    LLMClient,
    PLANNER_SYSTEM_PROMPT,
    SQL_GENERATOR_SYSTEM_PROMPT,
    SQL_CORRECTION_SYSTEM_PROMPT,
    INSIGHT_SYSTEM_PROMPT,
)


def test_prompt_formatting():
    """Verify that all system prompt templates format without missing placeholders."""
    p1 = PLANNER_SYSTEM_PROMPT.format(schema_context="Table: test (id INT, val TEXT)")
    assert "Table: test" in p1

    p2 = SQL_GENERATOR_SYSTEM_PROMPT.format(
        table_name="sales", schema_context="id INT, revenue FLOAT"
    )
    assert "sales" in p2
    assert "revenue FLOAT" in p2

    p3 = SQL_CORRECTION_SYSTEM_PROMPT.format(
        table_name="sales",
        schema_context="id INT",
        failed_query="SELECT col FROM sales",
        error_message="Column col does not exist",
    )
    assert "Column col does not exist" in p3

    p4 = INSIGHT_SYSTEM_PROMPT.format(
        user_query="Find top products",
        executed_sql="SELECT * FROM sales",
        query_results_text="product: Mouse, rev: 100",
    )
    assert "Find top products" in p4


def test_llm_client_fallback_planner():
    """Test LLMClient structured JSON generation in offline/fallback mode."""
    client = LLMClient()
    plan = client.generate_json(
        prompt="Analyze revenue by category",
        system_prompt=PLANNER_SYSTEM_PROMPT.format(schema_context="category TEXT, revenue FLOAT"),
    )
    assert isinstance(plan, dict)
    assert "intent" in plan
    assert "steps" in plan
    assert plan["requires_sql"] is True


def test_llm_client_fallback_sql_generation():
    """Test SQL generation in fallback mode produces valid SELECT query."""
    client = LLMClient()
    sql = client.generate(
        prompt="What is the total revenue by category?",
        system_prompt=SQL_GENERATOR_SYSTEM_PROMPT.format(
            table_name="dataset", schema_context="| category | categorical | object |\n| revenue | numeric | float64 |"
        ),
    )
    assert sql.upper().startswith("SELECT")
    assert "FROM dataset" in sql


def test_dynamic_query_differentiation():
    """Ensure different analytical questions produce distinct and tailored SQL queries."""
    from src.llm.query_planner import generate_dynamic_sql

    schema = (
        "| product_name | categorical | object |\n"
        "| category | categorical | object |\n"
        "| city | categorical | object |\n"
        "| revenue | numeric | float64 |\n"
        "| units_sold | numeric | int64 |\n"
        "| customer_rating | numeric | float64 |\n"
        "| order_date | datetime | object |\n"
    )

    # Question 1: Product by units sold
    q1 = generate_dynamic_sql("Identify the top 5 products by units sold", schema)
    assert "product_name" in q1
    assert "units_sold" in q1
    assert "LIMIT 5" in q1

    # Question 2: Rating by city
    q2 = generate_dynamic_sql("What is the average customer rating per city?", schema)
    assert "city" in q2
    assert "customer_rating" in q2
    assert "AVG" in q2

    # Question 3: Monthly revenue
    q3 = generate_dynamic_sql("Compare monthly revenue across all orders", schema)
    assert "order_date" in q3
    assert "revenue" in q3

    # Assert that all three queries are completely different!
    assert q1 != q2
    assert q2 != q3
    assert q1 != q3

