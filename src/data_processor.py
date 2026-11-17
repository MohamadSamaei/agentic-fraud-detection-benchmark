import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometers between coordinates using the Haversine formula.

    Args:
        lat1, lon1: Latitude and longitude of the first point(s) in decimal degrees.
        lat2, lon2: Latitude and longitude of the second point(s) in decimal degrees.

    Returns:
        Union[float, np.ndarray, pd.Series]: Distance(s) in kilometers.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def create_advanced_contextual_features(df):
    """Engineer causal temporal, 24h rolling velocity, historical spend, and geospatial features per card.

    Args:
        df (pd.DataFrame): Raw transaction DataFrame with card IDs, timestamps, amounts, and coordinates.

    Returns:
        pd.DataFrame: Chronologically sorted DataFrame enriched with contextual features.
    """
    print("Sorting data to prevent leakage")
    df = df.sort_values(by=['cc_num', "trans_date_trans_time"]).copy()

    # basic time calculations
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["time_since_last_trans"] = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()/3600
    df["time_since_last_trans"] = df["time_since_last_trans"].fillna(9999)
    df["is_night_time"] = df["trans_date_trans_time"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

    print("Calculating rolling windows")
    df = df.set_index("trans_date_trans_time")
    df["velocity_24h"] = df.groupby('cc_num')["trans_num"].transform(lambda x: x.rolling("24h").count() - 1)
    df["amount_24h"] = df.groupby("cc_num")['amt'].transform(lambda x: x.rolling("24h").sum() - x)
    df = df.reset_index()

    print("Calculating historical aggregations")
    grouped = df.groupby("cc_num")["amt"]
    df["card_mean_spend"] = grouped.transform(lambda x: x.shift().expanding().mean()).fillna(df["amt"])
    df["card_std_spend"] = grouped.transform(lambda x: x.shift().expanding().std()).fillna(1)
    df["card_median_spend"] = grouped.transform(lambda x: x.shift().expanding().median()).fillna(df["amt"])

    df["spend_ratio"] = df["amt"]/(df["card_median_spend"] + 1e-5)
    df["amount_z_score"] = (df["amt"] - df["card_mean_spend"])/(df["card_std_spend"] + 1e-5)

    df["merchant_familiarity"] = df.groupby(["cc_num", "merchant"])["trans_num"].transform(lambda x: x.expanding().count().shift()).fillna(0)
    df["category_familiarity"] = df.groupby(["cc_num", "category"])["trans_num"].transform(lambda x: x.expanding().count().shift()).fillna(0)

    print("Calculating geospatial features")
    df["dis_to_home"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["prev_merch_lat"] = df.groupby("cc_num")["merch_lat"].shift()
    df["prev_merch_long"] = df.groupby("cc_num")["merch_long"].shift()
    df["distance_from_prev_trans"] = haversine(df["merch_lat"], df["merch_long"], df["prev_merch_lat"], df["prev_merch_long"]).fillna(0)

    df["travel_speed"] = df["distance_from_prev_trans"]/(df["time_since_last_trans"] + 1e-5)
    
    df = df.drop(columns = ["prev_merch_lat", "prev_merch_long"])

    return df


def add_global_features(df):
    """Compute cardholder `age` in years at transaction time from `dob` and `trans_date_trans_time`.

    Args:
        df (pd.DataFrame): Transaction DataFrame containing `dob` and `trans_date_trans_time`.

    Returns:
        pd.DataFrame: DataFrame with the added integer `age` column.
    """
    dob_dt = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - dob_dt).dt.days // 365
    return df


def prepare_data_for_xgboost(train_df, test_df):
    """Extract calendar features, drop identifiers/PII, and cast categorical columns for XGBoost.

    Args:
        train_df (pd.DataFrame): Feature-engineered training DataFrame.
        test_df (pd.DataFrame): Feature-engineered test DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]: `(x_train, y_train, x_test, y_test, drop_columns)`.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for df in [train_encoded, test_encoded]:
        df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        df["trans_dayofweek"] = df["trans_date_trans_time"].dt.day_of_week
        df["trans_week"] = df["trans_date_trans_time"].dt.month


    drop_columns = ['Unnamed: 0', 'trans_num', 'cc_num', 'first', 'last', 'street', 
                    'city', 'zip', 'dob', 'trans_date_trans_time', 'unix_time', 'is_fraud']

    x_train = train_encoded.drop(columns = drop_columns)
    y_train = train_encoded["is_fraud"].astype(int)

    x_test = test_encoded.drop(columns = drop_columns)
    y_test = test_encoded["is_fraud"].astype(int)

    cat_cols = ["gender", "category", "state", "merchant", "job"]
    for col in cat_cols:
        x_train[col] = x_train[col].astype("category")
        x_test[col] = x_test[col].astype("category")

    return x_train, y_train, x_test, y_test, drop_columns

