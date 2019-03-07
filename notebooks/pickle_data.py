import pickle

import sys
sys.path.append('../')
from axia import loader, util


def main():
    print("Loading raw data...")
    jobber_data = loader.JobberDataLoader(path="../data/")
    df = jobber_data.get_data()

    print("Preparing subscription data...")
    data = util.SubscriptionData(
        data=df,
        start_date_col="first_paying_date",
        end_date_col="last_churn_date",
        subscription_initial="initial_subscription_amt",
        subscription_current="current_subscription_amt",
        additional_cols=[df["frequency"].astype("category")] + [
            df[col].apply(func).astype("category")
            for col, func in jobber_data.transformations.items()
        ],
        split_at="2018-01-01"
    )

    print("Saving subscription data object...")
    with open("./jobber_working_data-20180101_split.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Done!")


if __name__ == "__main__":
    main()
