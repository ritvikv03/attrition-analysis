import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "monthly_income": [3000, 5000, 4000, 6000, 4500, 7000],
            "job_satisfaction": [1, 2, 1, 3, 2, 3],
            "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(sample_df):
    result = attrition_by_department(sample_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values(sample_df):
    result = attrition_by_overtime(sample_df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    # 3 overtime employees, all 3 are leavers
    assert yes_row["employees"] == 3
    assert yes_row["leavers"] == 3
    assert yes_row["attrition_rate"] == 100.0

    no_row = result[result["overtime"] == "No"].iloc[0]
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    # Leavers: 3000, 4000, 4500 → mean = 3833.33
    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    assert yes_avg == round((3000 + 4000 + 4500) / 3, 2)

    # Stayers: 5000, 6000, 7000 → mean = 6000.0
    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert no_avg == 6000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_uses_group_headcount(sample_df):
    result = satisfaction_summary(sample_df)
    # Satisfaction 1: employees 1 & 3 → both leavers → 100%
    row1 = result[result["job_satisfaction"] == 1].iloc[0]
    assert row1["total_employees"] == 2
    assert row1["leavers"] == 2
    assert row1["attrition_rate"] == 100.0

    # Satisfaction 3: employees 4 & 6 → no leavers → 0%
    row3 = result[result["job_satisfaction"] == 3].iloc[0]
    assert row3["total_employees"] == 2
    assert row3["leavers"] == 0
    assert row3["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_by_satisfaction(sample_df):
    result = satisfaction_summary(sample_df)
    scores = result["job_satisfaction"].tolist()
    assert scores == sorted(scores)
