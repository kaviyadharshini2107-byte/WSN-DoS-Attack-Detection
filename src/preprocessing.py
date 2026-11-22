
import pandas as pd

def clean_wsn_data(data):
    """
    Clean WSN dataset column names
    and verify required features.
    """

    data = data.copy()

    # Remove unwanted spaces from column names
    data.columns = data.columns.str.strip()

    # Remove duplicate records
    data = data.drop_duplicates()

    return data


def validate_features(data, feature_names):
    """
    Check whether all required model features exist.
    """

    missing_features = [
        feature for feature in feature_names
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    return True
