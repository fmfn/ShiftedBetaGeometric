import numpy as np
import pandas as pd
from datetime import date, datetime


class SubscriptionData:
    def __init__(
        self,
        data,
        start_date_col=None,
        end_date_col=None,
        age_col=None,
        alive_col=None,
        subscription_initial=None,
        subscription_current=None,
        additional_cols=None,
        split_at=0.8,
        ):
        """
        params:
            split_at (float or str/date): If float, it will represent fraction
                of shuffled data to be used as training data. If string or
                date, it will separate training and validation set by
                start_dates before and after split_at.
        """

        self._data = data
        self._df = pd.DataFrame(
            index=data.index,
            columns=[
                "start_date",
                "end_date",
                "age",
                "alive",
                "subscription_initial",
                "subscription_last",
                "subscription_current",
            ]
        )
        self._dtr = self._df.copy()
        self._dva = self._df.copy()

        self._create_date_column(start_date_col, kind="start")
        self._create_date_column(end_date_col, kind="end")

        if age_col is not None:
            self._df["age"] = self._data[age_col].fillna(0).astype(int)
        else:
            self._df["age"] = self.df.apply(self._calculate_age, axis=1)

        if alive_col is not None:
            self._df["alive"] = self._data[alive_col].astype(int)
        else:
            self._df["alive"] = self.df["end_date"].apply(pd.isnull).astype(int)

        if subscription_initial is not None:
            self._df["subscription_initial"] = self._data[subscription_initial]

        if subscription_current is not None:
            self._df["subscription_last"] = self._data[subscription_current]

            self._df["subscription_current"] = self._data[subscription_current] * \
                self._df["alive"]

        self._subscription_trend = pd.Series(
            (
                (self.df["subscription_last"] - self.df["subscription_initial"]) /
                np.maximum(self.df["age"] - 1, 1)
            ),
            name="subscription_trend",
        )

        if additional_cols is not None:
            for col in additional_cols:
                if type(col) == str:
                    self._df[col] = self._data[col]
                else:
                    self._df = self._df.join(col)

        self._train_validation_split(split_at=split_at)

        self._cdf = self._cross_join(
            self.df[["start_date", "end_date"]].reset_index())

        if type(split_at) == str or type(split_at) == date:
            train_map = self._cdf.index.map(
                lambda ind: ind[1] < datetime.strptime(split_at, "%Y-%m-%d")
            ).values.astype(bool)

            self._cdtr = self._cdf[train_map]
            self._cdva = self._cdf[~train_map]
        else:
            self._cdtr = self.cdf.join(self.dtr[[]], how="inner")
            self._cdva = self.cdf.join(self.dva[[]], how="inner")

    @property
    def df(self):
        return self._df

    @property
    def dtr(self):
        return self._dtr

    @property
    def dva(self):
        return self._dva

    @property
    def cdf(self):
        return self._cdf

    @property
    def cdtr(self):
        return self._cdtr

    @property
    def cdva(self):
        return self._cdva

    @staticmethod
    def _calculate_age(row):
        if pd.isnull(row["end_date"]):
            end_date = date.today()
        else:
            end_date = row["end_date"]

        age = 12 * (end_date.year - row["start_date"].year)
        age += end_date.month - row["start_date"].month
        return max(age, 1)

    @staticmethod
    def _normalize_date_first_of_month(dt):
        if pd.isnull(dt):
            return dt
        return dt.strftime("%Y-%m-01")

    def _create_date_column(self, date_col, kind="start"):
        if date_col is not None:
            dates = self._data[date_col]
        else:
            try:
                dates = self._data["{kind}_date".format(kind=kind)]
            except KeyError:
                err_s = (
                    "Either a {kind} date column must be passed or a "
                    "column with name '{kind}_date' must exist in the "
                    "original dataframe."
                )
                raise KeyError(err_s.format(kind=kind))

        self._df["{kind}_date".format(kind=kind)] = pd.to_datetime(
            dates.apply(self._normalize_date_first_of_month)
        )

    def _train_validation_split(self, split_at):
        if type(split_at) == float:
            self._dtr = (
                self.df
                .sample(frac=1, random_state=7)
                .iloc[:int(split_at * len(self.df))]
            )
            self._dva = (
                self.df
                .sample(frac=1, random_state=7)
                .iloc[int(split_at * len(self.df)):]
            )
        elif type(split_at) == str or type(split_at) == date:
            self._dtr = self.df[self.df["start_date"] < split_at]
            self._dva = self.df[self.df["start_date"] >= split_at]
        else:
            raise TypeError(
                "split_at must be a float or a date represented either in "
                "date or string format (%Y-%m-%d)"
            )

    def _cross_join(self, df):
        def _dummy_col(df: pd.DataFrame):
            d = df.copy()
            d["dummy"] = 1
            return d

        def _is_alive(row):
            if ((row["start_date"].year == row["end_of_month"].year) and
                (row["start_date"].month == row["end_of_month"].month)):

                if ~pd.isnull(row["end_date"]):
                    if ((row["end_date"].year == row["end_of_month"].year) and
                        (row["end_date"].month == row["end_of_month"].month)):
                        return 0
                return 1

            if row["start_date"] > row["end_of_month"]:
                return 0

            if ~pd.isnull(row["end_date"]):
                if row["end_date"] < row["end_of_month"]:
                    return 0

            return 1

        def _age(row):
            if row["start_date"] > row["end_of_month"]:
                return np.nan

            age = 12 * (row["end_of_month"].year - row["start_date"].year)
            age += row["end_of_month"].month - row["start_date"].month
            return age

        def _starting_month(row):
            if ((row["start_date"].year == row["end_of_month"].year) and
                (row["start_date"].month == row["end_of_month"].month)):
                return 1
            return 0

        def _cancelation_month(row):
            if pd.isnull(row["end_date"]):
                return 0
            if ((row["end_date"].year == row["end_of_month"].year) and
                (row["end_date"].month == row["end_of_month"].month)):
                return 1
            return 0

        def _calculate_subscription_progression(row):
            if row["age"] == 0:
                return row["subscription_initial"]

            subscription = (
                row["subscription_initial"] +
                (row["subscription_trend"] * row["age"])
            )

            if row["subscription_last"] < row["subscription_initial"]:
                return (
                    row["alive"] *
                    max(row["subscription_last"], subscription)
                )
            elif row["subscription_last"] >= row["subscription_initial"]:
                return (
                    row["alive"] *
                    min(row["subscription_last"], subscription)
                )

        months = pd.DataFrame(
            data=pd.date_range(
                df["start_date"].min(),
                end=date.today(),
                freq="M",
            ),
            columns=['end_of_month']
        )

        df = (
            df
            .pipe(_dummy_col)
            .merge(
                right=(
                    months
                    .pipe(_dummy_col)
                ),
                on="dummy"
            )
            .drop("dummy", axis=1)
        )
        df = df[df["end_of_month"] >= df["start_date"]]

        df["age"] = df.apply(_age, axis=1)
        df["alive"] = df.apply(_is_alive, axis=1)
        df["is_starting_month"] = df.apply(_starting_month, axis=1)
        df["is_cancelation_month"] = df.apply(_cancelation_month, axis=1)

        df.set_index(
            keys=[df.columns[0], "end_of_month"],
            inplace=True,
        )

        df["months_alive"] = (
            df
            .groupby(df.index.get_level_values("account_id"))
            .agg({"alive": "cumsum"})
        )
        df["subscription_value_base"] = (
            df[["alive"]].join(
                self.df[["subscription_initial"]]
            )
            .apply(
                lambda row: row["subscription_initial"] * row["alive"],
                axis=1
            )
        )
        df["subscription_value_linear"] = (
            df
            .join(
                self.df[[
                    "subscription_initial",
                    "subscription_last",
                    "subscription_current",
                ]]
            )
            .join(self._subscription_trend)
            .apply(_calculate_subscription_progression, axis=1)
        )
        df["value_to_date_base"] = (
            df
            .groupby(df.index.get_level_values("account_id"))
            .agg({"subscription_value_base": "cumsum"})
        )
        df["value_to_date_linear"] = (
            df
            .groupby(df.index.get_level_values("account_id"))
            .agg({"subscription_value_linear": "cumsum"})
        )
        return df
