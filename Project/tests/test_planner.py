"""
tests/test_planner.py
Unit tests for the agent planner / intent classification.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.agents.planner import classify_intent, AgentPlan

COLUMNS = ["age", "salary", "department", "experience", "score"]


class TestClassifyIntent:
    def test_missing_values_intent(self):
        plan = classify_intent("Which column has most null values?", COLUMNS)
        assert plan.action == "missing_values"

    def test_missing_values_keyword(self):
        plan = classify_intent("How many missing values does age have?", COLUMNS)
        assert plan.action == "missing_values"

    def test_correlation_intent(self):
        plan = classify_intent("Are there any strong correlations?", COLUMNS)
        assert plan.action == "correlations"

    def test_outlier_intent(self):
        plan = classify_intent("Show me outliers in the dataset", COLUMNS)
        assert plan.action == "outliers"

    def test_duplicate_intent(self):
        plan = classify_intent("How many duplicate rows are there?", COLUMNS)
        assert plan.action == "duplicates"

    def test_quality_intent(self):
        plan = classify_intent("What is the data quality score?", COLUMNS)
        assert plan.action == "data_quality"

    def test_visualization_histogram(self):
        plan = classify_intent("Show me a histogram of salary", COLUMNS)
        assert plan.action == "generate_visualization"
        assert plan.chart_type == "histogram"
        assert plan.column == "salary"

    def test_visualization_scatter(self):
        plan = classify_intent("Scatter plot of age vs salary", COLUMNS)
        assert plan.action == "generate_visualization"

    def test_visualization_heatmap(self):
        plan = classify_intent("Show correlation heatmap", COLUMNS)
        assert plan.action == "generate_visualization"
        assert plan.chart_type == "heatmap"

    def test_visualization_boxplot(self):
        plan = classify_intent("Boxplot of age column", COLUMNS)
        assert plan.action == "generate_visualization"
        assert plan.chart_type == "boxplot"

    def test_report_intent(self):
        plan = classify_intent("Generate a PDF report", COLUMNS)
        assert plan.action == "generate_report"

    def test_column_statistics_intent(self):
        plan = classify_intent("What is the mean of salary?", COLUMNS)
        assert plan.action == "column_statistics"
        assert plan.column == "salary"

    def test_general_question(self):
        plan = classify_intent("What is the dataset about?", COLUMNS)
        assert plan.action == "retrieve_and_answer"

    def test_column_detection(self):
        plan = classify_intent("Tell me about the age column", COLUMNS)
        assert plan.column == "age"

    def test_no_column_when_absent(self):
        plan = classify_intent("Which column has outliers?", COLUMNS)
        # action is outliers, no specific column mentioned
        assert plan.action == "outliers"


class TestAgentPlan:
    def test_defaults(self):
        plan = AgentPlan(action="retrieve_and_answer")
        assert plan.column is None
        assert plan.chart_type is None
        assert plan.x_col is None
        assert plan.y_col is None
        assert plan.extra == {}