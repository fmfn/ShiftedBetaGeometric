import os
import pandas as pd


class _ClientDataLoader:
    def __init__(self, path):
        self._path = path
        self._transformations = {}

    @property
    def transformations(self):
        return self._transformations

    def get_data(self):
        return pd.DataFrame()


class JobberDataLoader(_ClientDataLoader):
    """"""

    def __init__(self, path):
        super().__init__(path=path)
        self._transformations = {
            "country": self.top_5_countries,
            "industry": self.top_20_crm_industries,
        }

    @staticmethod
    def top_5_countries(c):
        if c in ['United States', 'Canada', 'United Kingdom', 'Australia',
                 'New Zealand']:
            return c.lower().replace(" ", "_")
        return "other_country"

    @staticmethod
    def top_20_crm_industries(c):
        if c in ['Lawn Care & Lawn Maintenance', 'Other',
                 'Residential Cleaning', 'Construction & Contracting',
                 'HVAC', 'Landscaping Contractor', 'Plumbing',
                 'Electrical Contractor', 'Commercial Cleaning',
                 'Arborist / Tree Care', 'Pest Control', 'Window Washing',
                 'Handyman', 'Pressure Washing Service',
                 'Pool and Spa Service', 'Renovations', 'Painting',
                 'Computers & IT', 'Carpet Cleaning',
                 'Mechanical Service']:
            return c.lower().replace(" ", "_")
        return "other_industry"

    def get_data(self):
        ltv_file = "ltv_per_customer-20190215-jobber.csv"
        data_file = "customer-20190228-jobber.csv"

        ltv = pd.read_csv(
            filepath_or_buffer=os.path.join(self._path, ltv_file),
            index_col=["account_id"],
        )
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(self._path, data_file),
            parse_dates=["first_paying_date", "last_churn_date"],
            index_col=["account_id"],
        )

        df = df.drop_duplicates()
        df = df.dropna(
            how="all",
            subset=["initial_subscription_amt", "current_subscription_amt"]
        )

        irrelevant_cols = [
            'attr_channel_category_v1', 'attr_channel_v1', 'heard_about_us',
            'heard_about_us_category', 'heard_about_us_sub_category',
            'hdyhau_channel', 'hdyhau_channel_category', 'utmz_channel_category',
            'utmz_channel', 'lifecycle_state', 'account_created_month',
            'first_paying_month',
        ]
        df.drop(irrelevant_cols, axis=1, inplace=True)

        df = df.join(ltv.drop("lifecycle_state", axis=1))
        df = df.fillna({'paying_months': 0, 'tot_revenue': 0})
        df.rename(
            columns={
                "crm_industry": "industry",
                "initial_billing_frequency": "frequency",
            },
            inplace=True
        )

        df = df[df['frequency'] != "Unknown"]

        return df
