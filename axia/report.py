import numpy as np
import pandas as pd
from axia.util import SubscriptionData


class _BaseReport:
    pass


class ReportActuals(_BaseReport):
    """ReportActuals will streamline reporting."""

    def __init__(self, subscription_data: SubscriptionData):
        self._data = subscription_data

    def observed_ltv(self, age=12, start_date_cutoff=None):
        if start_date_cutoff is not None:
            data = self._data.cdf[
                self._data.cdf["start_date"] >= start_date_cutoff]
        else:
            data = self._data.cdf

        return (
            data
            .query("age == {age}".format(age=age - 1))
            [[
                "alive",
                "subscription_value_base",
                "value_to_date_base",
                "subscription_value_linear",
                "value_to_date_linear"
            ]]
            .describe(percentiles=[0.1, 0.5, 0.9])
            .loc[["mean", "10%", "50%", "90%"]]
        )

    def observed_ltv_by_cohort(self, age=12, start_date_cutoff=None,
                               smoothing=0):
        if start_date_cutoff is not None:
            data = self._data.cdf[
                self._data.cdf["start_date"] >= start_date_cutoff]
        else:
            data = self._data.cdf

        report = (
            data
            .query("age == {age}".format(age=age - 1))
            .groupby("start_date")
            .agg({
                "age": "count",
                "alive": "mean",
                "months_alive": "mean",
                "subscription_value_base": "mean",
                "value_to_date_base": "mean",
                "subscription_value_linear": "mean",
                "value_to_date_linear": "mean",
            })
        )

        report = report.rename(columns={
            "age": "cohort_size",
            "months_alive": "mean_months_alive",
            "subscription_value_base": "mean_subscription_value_base",
            "value_to_date_base": "mean_value_to_date_base",
            "subscription_value_linear": "mean_subscription_value_linear",
            "value_to_date_linear": "mean_value_to_date_linear",
        })

        if smoothing:
            report = report.rolling(window=smoothing).mean()

        return report

    def plot_cohort_trends(self, age=12, start_date_cutoff=None,
                           smoothing=0):
        import matplotlib.pyplot as plt
        actuals = self.observed_ltv_by_cohort(
            age=age,
            start_date_cutoff=start_date_cutoff,
            smoothing=smoothing,
        )

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12.5), sharex=False)
        fig.suptitle("{age} months actuals".format(age=age))

        actuals[["cohort_size"]].plot(ax=ax[0][0], c='black')
        actuals[["mean_subscription_value_base"]].plot(ax=ax[0][1])
        actuals[["mean_subscription_value_linear"]].plot(ax=ax[0][1])
        actuals[["mean_value_to_date_base"]].plot(ax=ax[1][1])
        actuals[["mean_value_to_date_linear"]].plot(ax=ax[1][1])

        ax01_2 = ax[1][0].twinx()
        a, = ax[1][0].plot(
            actuals.index,
            actuals["alive"],
            color='purple',
            label='alive',
        )
        b, = ax01_2.plot(
            actuals.index,
            actuals["mean_months_alive"],
            color='green',
            label='mean_months_alive',
        )
        p = [a, b]
        ax[1][0].legend(
            p,
            [p_.get_label() for p_ in p],
            loc='lower left',
        )
        ax01_2.set_xbound((actuals.index[0], actuals.index[-1]))

        for rax in ax:
            for cax in rax:
                cax.grid(True)
