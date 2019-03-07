import numpy as np
import pandas as pd
from datetime import date, datetime


class SubscriptionData:
    def __init__(
        self,
        data,
        start_date_col=None,
        end_date_col=None,
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

        self._train_validation_split(split_at=split_at)

        for split_kind, df in zip([None, "train", "validation"],
                                  [self.df, self.dtr, self.dva]):
            df["age"] = df.apply(
                lambda row: self._calculate_age(row, split_at, split_kind),
                axis=1
            )
            df["alive"] = df["end_date"].apply(pd.isnull).astype(int)

            if subscription_initial is not None:
                df["subscription_initial"] = df.join(
                    other=self._data[subscription_initial],
                    lsuffix="_"
                )[subscription_initial]

            if subscription_current is not None:
                df["subscription_last"] = df.join(
                    other=self._data[subscription_current],
                    lsuffix="_"
                )[subscription_current]

                df["subscription_current"] = (
                    df["subscription_last"] * df["alive"]
                )

            if additional_cols is not None:
                for col in additional_cols:
                    if type(col) == str:
                        df[col] = df.join(self._data[col])[col]
                    elif type(col) == pd.Series:
                        df[col.name] = df.join(col)[col.name]
                    else:
                        raise ValueError(
                            "Additional columns must either be strings " +
                            "matching existing columns in the original " +
                            "data, or named pandas series. with matching " +
                            "indices."
                        )

        self._subscription_trend = pd.Series(
            data=(
                (self.df["subscription_last"] - self.df["subscription_initial"]) /
                np.maximum(self.df["age"] - 1, 1)
            ),
            name="subscription_trend",
        )

        self._periods = pd.DataFrame(
            data=pd.date_range(
                self.df["start_date"].min(),
                end=date.today(),
                freq="M",
            ),
            columns=['end_of_month']
        )

        self._cdf = self._cross_join(
            self.df[["start_date", "end_date"]].reset_index())

        self._cross_train_validation_split(split_at=split_at)

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

    def _calculate_age(self, row, split_at, split_kind):
        if pd.isnull(row["end_date"]):
            if split_kind == "train" and type(split_at) == datetime:
                end_date = split_at
            else:
                end_date = datetime.now()
        else:
            end_date = row["end_date"]

        age = 12 * (end_date.year - row["start_date"].year)
        age += end_date.month - row["start_date"].month

        assert age >= 0, (
            "Something went wrong, check start and end dates make sense"
        )
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
        elif type(split_at) == datetime:
            self._dtr = self.df[self.df["start_date"] < split_at].copy()
            self._dva = self.df[self.df["start_date"] >= split_at].copy()

            # Make sure future end dates are not evident in the training set
            self._dtr['end_date'] = self._dtr["end_date"].apply(
                lambda dt: dt if dt < split_at else None)
        else:
            raise TypeError(
                "split_at must be a float or a datetime object"
            )

    def _cross_train_validation_split(self, split_at):
        try:
            # Only rows for which the end of the month date (represented by
            # index 1 of multi-index) happens before the split date will be
            # included in the train set.
            train_map = self._cdf.index.map(
                lambda ind: ind[1] < split_at
            ).values.astype(bool)

            self._cdtr = self._cdf[train_map].copy()
            self._cdva = self._cdf[~train_map].copy()

            # Make sure future end dates are not evident in the training set
            self._cdtr['end_date'] = self._cdtr["end_date"].apply(
                lambda dt: dt if dt < split_at else None)
        except TypeError:
            self._cdtr = self.cdf.join(self.dtr[[]], how="inner")
            self._cdva = self.cdf.join(self.dva[[]], how="inner")

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

        df = (
            df
            .pipe(_dummy_col)
            .merge(
                right=(
                    self._periods
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

    def month_one_retention_trend(self, smoothing=3):
        """
        NOTE: Data is two months old in a sense, making sure both
            subscription and cancelation month are over.
        """
        retention = (
            self.cdtr
            .query("age == 1")
            .groupby("start_date")
            .agg({"alive": "mean"})
            .sort_index()
            .rolling(window=smoothing, min_periods=0)
            .agg({"alive": "mean"})
            .rename(columns={
                "alive": "month_one_retention_trend_s{s}".format(s=smoothing)
            })
        )

        start_date_index = pd.Series(
            data=self._periods.apply(
                lambda row: row["end_of_month"].replace(day=1),
                axis=1
            ),
            name="start_date"
        )

        # Retention trends must exist for all cohorts in the dataset.
        # In the case explicit data for a given cohort doesn't exist,
        # either due to it being a slow cohort with no data, or a future
        # cohort for which data doesn't exist yet, we make sure a trend
        # value for it exists by forward filling empty slots. I.e.: the
        # last trend value know will be used.
        retention = (
            pd.DataFrame(index=start_date_index)
            .join(retention)
            .fillna(method="ffill")
            .shift(2)
        )

        return retention

    def features(self):
        retention_trend_3 = self.month_one_retention_trend(smoothing=3)
        retention_trend_12 = self.month_one_retention_trend(smoothing=12)

        features = (
            self.df[["start_date"]]
            .merge(
                right=retention_trend_3,
                left_on="start_date",
                right_index=True,
            )
            .merge(
                right=retention_trend_12,
                left_on="start_date",
                right_index=True,
            )
            .drop("start_date", axis=1)
        )

        return features


class DataHandler:
    """
    DataHandler is an object to perform several manipulations to a pandas
    dataframe making it suitable to be fed to a ShiftedBeta object.

    Given a dataframe the user specifies an age and alive fields, and possible
    feature fields. These will be processed to comply with the input API of
    the lower lever ShiftedBeta object.

    Categorical features will be one-hot-encoded, a bias column can be added,
    and numerical features can also be normalized during the pre-processing
    stage.
    """

    def __init__(self, age, alive, features=None, bias=True, normalize=True):
        """
        The object is initialized with the dataset to be transformed, the name
        of the fields identified as cohort, individual age and optional
        category(ies) to be used as predictors.

        :param age: str
            The column name to identify the age of each individual. Age has to
            be an integer value, and will determine the time intervals the
            model with work with.

        :param alive: str
            The column name with the status of each individual. In the context
            of survival analysis, an individual may be dead or alive, and its
            contribution to the model will depend on it.

        :param features: str, list or None
            A string with the name of the column to be used as features, or a
            list of names of columns to be used as features or None, if no
            features are to be used.

        :param bias: bool
            Whether or not a bias term should be added to the feature matrix.

        :param normalize: bool
            Whether or not numerical fields should be normalized (centered and
            scaled to have std=1)
        """

        # The name of age and alive fields
        self.age = age
        self.alive = alive

        # Make sure we have something to be used as a feature matrix. I.e.:
        # features=None and bias=False should raise an error.
        if features is None and not bias:
            raise ValueError("bias must be True if no features are being used "
                             "(features=None).")

        # If the category name was passed as a single string, we turn it into
        # a list of one element (not list of characters, as you would get with
        # list('abc').
        if isinstance(features, str):
            features = [features]
        # Try to explicitly transform category to a list (perhaps it was passed
        # as a tuple or something. If it was None to begin with, we catch a
        # TypeError and move on.
        try:
            self.features = sorted(features)
        except TypeError:
            self.features = None

        # What features are categorical?
        self.categorical = []
        self.numerical = []

        # OHE feature map to be constructed
        self.feature_map = {}

        # should bias be added
        self.add_bias = bias

        # standarize features?
        self.normalize = normalize
        self.stats = {'mean': {}, 'std': {}}

        # fit before transform!
        self.fitted_model = False

    @staticmethod
    def _get_categoricals(df, features):
        """
        A method to sort features and divide them into categorical and
        numerical, and storing the names in sorted lists.

        Moreover this method also construct the dictionary feat_map, which
        takes categorical feature names as keys and a list with all the
        values they take as values.

        :param df: pandas DataFrame
            pandas dataframe from which features will be extracted

        :param features:
            list of columns names that will be used as features.

        :return: tuple (list, dict, list)
            A tuple with:
                A list of names of categorical features
                A dict with categorical feature name: all possible values
                A list of names of numerical features
        """

        # No features? No problem! Return empty object and let the code deal
        # with it further ahead.
        if features is None:
            return [], {}, []

        # Yes Features? Do stuff!!
        # Start by identifying which features are categorical by checking the
        # data types of columns in the dataframe. Those with data type
        # "category" are added to the list.
        cat_list = df.columns[df.dtypes == 'category']
        cat_list = [cat for cat in cat_list if cat in features]

        # Update them categorical features
        cat_list = sorted(cat_list)

        # Build feature maps!
        feat_map = {}

        # Loop over the list of categorical features creating a map that takes
        # each factor to an integer, in alphabetical order. So if feature
        # 'city' takes values 'NY', 'LA' and 'SF' the feature map will have the
        # entry: {'city': {'LA': 0, 'NY': 1, 'SF': 2}, ...}. Similarly for all
        # categorical features present.
        for feature in cat_list:
            feat_map[feature] = dict(zip(sorted(df[feature].cat.categories),
                                         range(len(df[feature].cat.categories))))

        # Update list of numerical features. Make it such any feature that was
        # given by the user and it is not a categorical feature, will be
        # considered a numerical feature.
        num = sorted([feat for feat in features if feat not in cat_list])

        # Returns both lists
        return cat_list, feat_map, num

    def _one_hot_encode(self, df, categoricals):
        """
        A method to one-hot-encode categorical features. It computes a dummy
        matrix for each categorical column and returns a concatenation of all
        such matrices.


        :param df: pandas DataFrame
            Pandas dataframe from which features are to be extracted.

        :param categoricals:
        :return:
        """
        # Make sure these are sorted so we don't get things mixed up later!
        categoricals = sorted(categoricals)

        # dict to hold matrices of OHE features
        ohed_map = {}

        # Loop over each categorical feature and create appropriate dummy
        # matrices. The size of the matrix is dictated by the feature_map
        # instance variable, that, for each categorical feature, hold a dict
        # with as many key: val pairs as there are factors in the training set
        for feature in categoricals:
            # categoricals: sorted list with names of categorical features
            #      feature: name of a categorical feature

            # Use the size of the dictionary to create a dummy matrix with the
            # appropriate size. ** Redundant since dummy matrix need
            # only to have n_factors - 1 columns, however this model only makes
            # sense with a bias variable added, which takes care of that.
            ohed_map[feature] = np.zeros((df.shape[0],
                                          len(self.feature_map[feature])),
                                         dtype=int)

        # Sometimes it is useful to know if new factors were found in the test
        # set, so we create a dict to store a list of new factors found in each
        # categorical feature.
        warning_new = {}

        # Mutable object to store index changes! No need for pandas index, phew!
        # using a mutable list seemed like the easiest answer, even if its a bit
        # ugly.
        currow_mutable = [0, ]

        # Internal function that is passed to pandas' apply method.
        def update_ohe_matrix(row, categorical_columns, currow):
            """
            Given a row of a pandas DataFrame, this function updates all
            dummy matrices corresponding to each categorical variables.

            Notice that the feature_map variable was generated in a such a way
            to immediately lends itself to populating an empty matrix in dummy
            fashion.

            So this function can simply use the value in the factor: value pair
            to set to one the correct column in the dummy matrix.

            As for rows, we use an external, mutable object to keep track of
            which row the pandas object is reading the data from and use that
            to update the correct row of the dummy matrices.

            :param row: row of a pandas DataFrame
                The row passed by the method apply of a pandas DataFrame.

            :param categorical_columns: list
                A list with the name of all categorical features that should be
                one hot encoded.

            :param currow: list (mutable object)
                An external, mutable object used to keep track of which row of
                the dataframe is being read (in case indexes are messed up).
            """

            # Loop over all categorical features
            for curr_feature in categorical_columns:
                # categorical_columns: list of names of categorical variables
                #        curr_feature: name of categorical variable

                # Read the index of current row from external mutable object
                row_index = currow[-1]

                # Value of current feature in current row
                row_feat_val = row[curr_feature]

                # Sometimes the test set contains factors that were not present
                # at training time, so we must take that into account. We do so
                # by catching a KeyError generated by trying to read an invalid
                # value off of a dictionary.
                try:
                    # Map between current categorical row-feature value and its
                    # numerical representation
                    mapped_val = self.feature_map[curr_feature][row_feat_val]

                    # Update OHE matrix by adding one to the appropriate column
                    # in the current row
                    ohed_map[curr_feature][row_index, mapped_val] += 1
                except KeyError:
                    try:
                        # Add newly seen value to warning dict
                        warning_new[curr_feature].add(row_feat_val)
                    except KeyError:
                        # If warning dict hasn't been populated yet,
                        # we do it here.
                        warning_new[curr_feature] = {row_feat_val}

            # Update the index so that the next row with be read the next time
            # this function is invoked.
            currow[-1] += 1

        # apply function with pandas apply method
        df.apply(lambda row: update_ohe_matrix(row,
                                               categoricals,
                                               currow_mutable),
                 axis=1)

        # Print some warking in case new factors were found.
        if len(warning_new) > 0:
            print('WARNING: NEW STUFF: {}'.format(warning_new))

        # Return a matrix complized of all dummy matrices for each categorical
        # variables concatenated along columns.
        return np.concatenate([xohe for key, xohe in ohed_map.items()], axis=1)

    def fit(self, df):
        """
        This method is responsible for scanning the dataframe -df- and storing
        the necessary parameters and statistics used to pre-process all
        features.

        It starts by splitting the full features list into a categorical and a
        numerical list of features, while also creating the feature_map for
        categorical features. This is done by invoking the _get_categoricals
        method.

        That being done, the next step is computing the mean and standard
        deviations of all numerical features in case the normalize keyword was
        set to True when the object was created. These statistics are stored
        in a dictionary to be used at transformation time.

        Finally the method sets the fitted_model flag to True.

        :param df: pandas DataFrame
            The dataframe on which categorical and numerical features will be
            based.
        """

        # Get types of features (in place updates!)
        cat, feat_map, num = self._get_categoricals(df=df,
                                                    features=self.features)

        # store features
        self.categorical.extend(cat)
        self.feature_map = feat_map

        # numerical
        self.numerical.extend(num)
        # should we center and standard?
        if self.normalize and len(self.numerical) > 0:
            # pandas is awesome!
            stats = df[self.numerical].describe().T.to_dict()
            # update mean and std at once =)
            self.stats['mean'].update(stats['mean'])
            self.stats['std'].update(stats['std'])

        # update fitted status
        self.fitted_model = True

    def transform(self, df):
        """
        A method to turn a pandas DataFrame into feature and target matrices
        suitable for the ShiftedBeta object.

        Armed with he one-hot-encoding and (possible) centering and re-scaling
        of the training data. This method will apply the learned
        transformations to the dataframe in question.

        It starts by constructing the feature matrix. It adds bias as needed
        followed by extracting the numerical features, which are then centered
        and re-scaled if self.normalize = True. Finally it contructs the one-
        hot-encoding of the categorical features.

        If no features were passed by the user, and the parameter bias is False
        the feature matrix will come out as None, to avoid it, this method
        adds a layer of protection by raising a ValueError.

        Target features, age and alive, may or may not be present in the
        dataframe. Since their presence is not strictly necessary, this method
        can handle both scenarios. It does so by trying to extract the target
        values, and, in case of failure, a value of None is return. This is
        done with the help of the "return" function below.

        :param df: pandas DataFrame
            The pandas dataframe we want to transform.

        :return: tuple (ndarray of shape(n_samples, n_features),
                        ndarray of shape(n_samples, ) or None,
                        ndarray of shape(n_samples, ) or None)
            The feature matrix and corresponding target values (when they exist
            otherwise None is returned).
        """
        # Make sure the object was trained first.
        if not self.fitted_model:
            raise RuntimeError("Fit to data before transforming it.")

        # Add a bias columns (a columns of ones) to the feature matrix. Note
        # that if not bias, then the feature matrix is given a temporary value
        # of None.
        if self.add_bias:
            xout = np.ones((df.shape[0], 1), dtype=int)
        else:
            xout = None

        # If the list self.numerical is not empty if means we have numerical
        # features. If that is the case, proceed to extracting and transforming
        # them as needed.
        if len(self.numerical) > 0:

            # Numerical variables extracted from dataframe in alphabetical
            # order.
            num_vals = df[sorted(self.numerical)].values

            # If self.normalize = True, proceed with centering and re-scaling.
            if self.normalize:
                for col, num_feat in enumerate(sorted(self.numerical)):
                    #    col: Position of column to be altered (recall that
                    #         they are arranged in alphabetical order)
                    # num_feat: the name of the numerical features (necessary
                    #           to load stats from self.stats dictionary)

                    # Center values by subtracting mean.
                    num_vals[:, col] -= self.stats['mean'][num_feat]
                    # Re-scale by dividing by standard-deviation (with some
                    # arbitrary clip on minimum STD lest things break)
                    num_vals[:, col] /= max(self.stats['std'][num_feat], 1e-4)

            # If bias is True, the feature matrix already exists and we simply
            # append to it. However, if bias is False the feature matrix is
            # overwritten from None to the numerical feature matrix created here.
            if self.add_bias:
                xout = np.concatenate((xout, num_vals),
                                      axis=1)
            else:
                xout = num_vals

        # If the list self.categorical is not empty if means we have
        # categorical features. If that is the case, proceed to extracting and
        # one-hot-encoding them as needed.
        if len(self.categorical) > 0:

            # Use method _one_hot_encode to ohe the features passed in the
            # sorted self.categorical list. While this list has been sorted
            # at the origin, we do so again here as an extra layer of
            # protection.
            xohe = self._one_hot_encode(df, sorted(self.categorical))

            # By now, either the feature matrix xout is still None (in case of
            # no bias and no numerical features) in which case it becomes the
            # one-hot-encoded matrix we just created, or it is something (bias,
            # numerical of both) in which case we append the OHE features to
            # it.
            if xout is not None:
                xout = np.concatenate((xout, xohe),
                                      axis=1)
            else:
                xout = xohe

        # If by this point the feature matrix is still None it means no data
        # will be used, which is obviously a problem. While this should not be
        # possible, and should be stopped at the object constructor, we add an
        # extra layer of protection here and raise a Value Error just in case.
        if xout is None:
            raise ValueError('No data!')

        def returner(data_frame, key):
            """
            When transforming the dataset, age and alive field must not be always
            present. If that's the case, we return None. To make life easier, and
            avoid an ugly chain of if statements, we create a nice little
            function to handle it.

            :param data_frame: pandas DataFrame
                dataframe from which target variables will be extracted.

            :param key: str
                name of the target variable in question

            :return: ndarray of shape(n_samples, ) or None
                np.array will target values or None, if the target variable is
                not present in dataframe.
            """
            # Try to extract target values, if the field is not present in the
            # dataframe, pandas will raise a KeyError, we catch it and return
            # None.
            try:
                return data_frame[key].values.astype(int)
            except KeyError:
                return None

        return xout, returner(df, self.age), returner(df, self.alive)

    def fit_transform(self, df):
        """
        A method to implement both fit and transform in one shot. While these
        two methods, when done together can be optimized, for simplicity (and
        some laziness) we don't optimize anything here.

        :param df: pandas DataFrame
            dataframe with features and target values

        :return: tuple (ndarray of shape(n_samples, n_features),
                        ndarray of shape(n_samples, ) or None,
                        ndarray of shape(n_samples, ) or None)
            The feature matrix and corresponding target values (when they exist
            otherwise None is returned).
        """

        self.fit(df)
        return self.transform(df)

    def get_names(self):
        """
        A handy function to return the names of all variables in the
        transformed version of the dataset in the correct order. Particularly
        useful for the ShiftedBetaSurvival wrapper.

        :return: list
            list of names in correct order
        """
        # Initialize an empty list to store the names
        names = []

        # If bias is true (meaning we added bias) we add bias as the fiest
        # name in the list.
        if self.add_bias:
            names.append('bias')

        # If numerical features are being used, we added them. As usual, we
        # make sure to sort them before doing so lest something weird happened.
        if len(self.numerical) > 0:
            names.extend(sorted(self.numerical))

        # If categorical features are being used we must add them too. Here we
        # use a two level naming convention: category_factor.
        if len(self.categorical) > 0:
            # Sort everything to avoid naming things incorrectly. Notice
            # that all names should be sorted in their origin. However,
            # it doesn't hurt to be extra safe..
            for cat_name in sorted(self.categorical):
                # Use the feature map to get the name of all factor for the
                # current category. Sort them and add to composite name to the
                # list.
                for category in sorted(self.feature_map[cat_name]):
                    names.append(cat_name + "_" + category)

        return names
