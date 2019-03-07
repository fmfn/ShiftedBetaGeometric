import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime
from axia.util import SubscriptionData
from axia.report import ReportActuals


_data1 = [
    [date(2017, 6, 10), np.nan, 10, 10, "a"],
    [date(2017, 6, 10), np.nan, 10, 10, "a"],
    [date(2017, 6, 10), np.nan, 10, 10, "a"],
    [date(2017, 6, 10), np.nan, 10, 10, "a"],
    [date(2017, 6, 10), np.nan, 10, 10, "a"],
    [date(2017, 6, 10), np.nan, 10, 10, "a"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c"],
]

_data2 = [
    [date(2017, 6, 10), np.nan, 10, 10, "a1"],
    [date(2017, 6, 10), np.nan, 10, 10, "a1"],
    [date(2017, 6, 10), np.nan, 10, 10, "a1"],
    [date(2018, 2, 10), np.nan, 20, 20, "a2"],
    [date(2018, 2, 10), np.nan, 20, 20, "a2"],
    [date(2018, 2, 10), np.nan, 20, 20, "a2"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b1"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b1"],
    [date(2017, 6, 10), date(2018, 6, 21), 10, 10, "b1"],
    [date(2018, 2, 10), date(2019, 2, 21), 20, 20, "b2"],
    [date(2018, 2, 10), date(2019, 2, 21), 20, 20, "b2"],
    [date(2018, 2, 10), date(2019, 2, 21), 20, 20, "b2"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c1"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c1"],
    [date(2017, 6, 10), date(2017, 12, 21), 10, 10, "c1"],
    [date(2018, 2, 10), date(2018, 8, 21), 20, 20, "c2"],
    [date(2018, 2, 10), date(2018, 8, 21), 20, 20, "c2"],
    [date(2018, 2, 10), date(2018, 8, 21), 20, 20, "c2"],
]


def get_test_data(subscriber_data):
    test_data = pd.DataFrame(
        data=subscriber_data,
        columns=[
            "start_date",
            "end_date",
            "subscription_initial",
            "subscription_current",
            "category",
        ],
        index=pd.Series(
            data=range(1, len(subscriber_data) + 1),
            name="account_id",
        ),
    )
    test_data["start_date"] = pd.to_datetime(test_data["start_date"])
    test_data["end_date"] = pd.to_datetime(test_data["end_date"])
    return test_data


def test_report_observed_ltv_simple():
    data = SubscriptionData(
        data=get_test_data(_data1),
        subscription_initial="subscription_initial",
        subscription_current="subscription_current",
    )
    report = ReportActuals(subscription_data=data)

    actuals = report.observed_ltv()
    assert actuals.loc["mean", "alive"] == 12.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == 120.0 / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == 120.0 / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == 100
    assert actuals.loc["mean", "value_to_date_base"] == 100

    actuals = report.observed_ltv(age=7)
    assert actuals.loc["mean", "alive"] == 12.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == 120.0 / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == 120.0 / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == (70 + 70 + 60) / 3
    assert actuals.loc["mean", "value_to_date_base"] == (70 + 70 + 60) / 3

    actuals = report.observed_ltv(age=6)
    assert actuals.loc["mean", "alive"] == 18.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == 180.0 / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == 180.0 / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == 60
    assert actuals.loc["mean", "value_to_date_base"] == 60

    actuals = report.observed_ltv(age=13)
    assert actuals.loc["mean", "alive"] == 6.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == 60.0 / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == 60.0 / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == (130 + 120 + 60) / 3
    assert actuals.loc["mean", "value_to_date_base"] == (130 + 120 + 60) / 3


def test_report_observed_ltv_mixed():
    data = SubscriptionData(
        data=get_test_data(_data2),
        subscription_initial="subscription_initial",
        subscription_current="subscription_current",
    )
    report = ReportActuals(subscription_data=data)

    actuals = report.observed_ltv()
    assert actuals.loc["mean", "alive"] == 12.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == (6 * 10 + 6 * 20) / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == (6 * 10 + 6 * 20) / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == (6 * 120 + 6 * 240 + 3 * 60 + 3 * 120) / 18
    assert actuals.loc["mean", "value_to_date_base"] == (6 * 120 + 6 * 240 + 3 * 60 + 3 * 120) / 18

    actuals = report.observed_ltv(age=7)
    assert actuals.loc["mean", "alive"] == 12.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == (6 * 10 + 6 * 20 + 6 * 0) / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == (6 * 10 + 6 * 20 + 6 * 0) / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == (6 * 70 + 6 * 140 + 3 * 60 + 3 * 120) / 18.0
    assert actuals.loc["mean", "value_to_date_base"] == (6 * 70 + 6 * 140 + 3 * 60 + 3 * 120) / 18.0

    actuals = report.observed_ltv(age=6)
    assert actuals.loc["mean", "alive"] == 18.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == (9 * 10 + 9 * 20) / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == (9 * 10 + 9 * 20) / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == (9 * 60 + 9 * 120) / 18.0
    assert actuals.loc["mean", "value_to_date_base"] == (9 * 60 + 9 * 120) / 18.0

    actuals = report.observed_ltv(age=13)
    assert actuals.loc["mean", "alive"] == 6.0 / 18.0
    assert actuals.loc["mean", "subscription_value_base"] == (3 * 10 + 3 * 20) / 18.0
    assert actuals.loc["mean", "subscription_value_linear"] == (3 * 10 + 3 * 20) / 18.0
    assert actuals.loc["mean", "value_to_date_linear"] == (60 + 120 + 120 + 240 + 130 + 260) / 6.0
    assert actuals.loc["mean", "value_to_date_base"] == (60 + 120 + 120 + 240 + 130 + 260) / 6.0


def test_report_observed_ltv_by_cohort():
    data = SubscriptionData(
        data=get_test_data(_data2),
        subscription_initial="subscription_initial",
        subscription_current="subscription_current",
    )
    report = ReportActuals(subscription_data=data)
    actuals = report.observed_ltv_by_cohort()

    print(report.observed_ltv_by_cohort().T)
    for (start_date, value) in zip(["2017-06-01", "2018-02-01"], [10.0, 20.0]):
        assert (
            actuals.loc[start_date, "cohort_size"] ==
            9
        )
        assert (
            actuals.loc[start_date, "alive"] ==
            pytest.approx(6 / 9)
        )
        assert (
            actuals.loc[start_date, "mean_months_alive"] ==
            pytest.approx((12 + 12 + 6) / 3)
        )
        assert (
            actuals.loc[start_date, "mean_subscription_value_base"] ==
            pytest.approx(value * (6 / 9))
        )
        assert (
            actuals.loc[start_date, "mean_subscription_value_linear"] ==
            pytest.approx(value * (6 / 9))
        )
        assert (
            actuals.loc[start_date, "mean_value_to_date_base"] ==
            pytest.approx(value * ((12 + 12 + 6) / 3))
        )
        assert (
            actuals.loc[start_date, "mean_value_to_date_linear"] ==
            pytest.approx(value * ((12 + 12 + 6) / 3))
        )


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    pytest.main([__file__])
