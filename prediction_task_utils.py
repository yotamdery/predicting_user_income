import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import Tuple


def get_outliers_top_n_features(df, target_label, top_n_features: int=20) -> pd.DataFrame:
    """ Returns a DF with description of how many outliers are the for each feature, from the top N correlated features (with the label)"""
    top_n_features = get_corr_top_n_features(df, target_label, top_n_features)
    top_n_features_names = list(top_n_features.index )
    # Compute IQR for each feature
    outlier_counts = {}

    for feature in top_n_features_names:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        # Define outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()

        outlier_counts[feature] = outliers

    # Convert outlier counts to DataFrame for better interpretation
    outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=["Feature", "Outlier Count"]).sort_values(by="Outlier Count", ascending=False)
    outlier_df = outlier_df.reset_index(drop=True)
    # Returns the number of outliers detected in each feature
    return outlier_df

def get_corr_top_n_features(df, target_label, top_n_features: int=20) -> pd.DataFrame:
    correlation_matrix = df.copy()
    correlation_matrix["target"] = target_label  # Add target for correlation check
    corr_with_target = np.round(correlation_matrix.corr()["target"].drop("target").sort_values(ascending=False),2).head(top_n_features)
    return corr_with_target

def identify_highly_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute the correlation matrix for feature selection
    feature_corr_matrix = df.corr()

    # Identify pairs of highly correlated features (absolute correlation > 0.9)
    high_corr_pairs = []
    threshold = 0.9  # Set correlation threshold

    for col in feature_corr_matrix.columns:
        for idx in feature_corr_matrix.index:
            if col != idx and abs(feature_corr_matrix.loc[idx, col]) > threshold:
                high_corr_pairs.append((col, idx, feature_corr_matrix.loc[idx, col]))

    # Convert to a DataFrame for easier interpretation
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"])

    # Output the highly correlated feature pairs
    high_corr_df = high_corr_df.sort_values(by='Correlation', ascending=False)
    high_corr_df = high_corr_df.reset_index(drop=True)
    high_corr_df = high_corr_df.drop_duplicates()
    return high_corr_df
    

def identify_skewed_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute skewness for each feature in X_train
    skewness = df.apply(lambda x: skew(x), axis=0)

    # Identify highly skewed features (e.g., absolute skewness > 1)
    highly_skewed_features = skewness[abs(skewness) > 1].sort_values(ascending=False)
    return highly_skewed_features

def apply_log_transformation(feature: pd.Series) -> pd.Series:
    # Apply log transformation
    feature_log = np.log1p(feature)  # log1p to handle zeros safely
    return feature_log

def check_skewness_score(feature: pd.Series) -> Tuple[float, float]:
    # Compute skewness of the feature variable
    feature_skewness = skew(feature)

    # Apply log transformation
    feature_log = apply_log_transformation(feature)

    # Compute skewness after transformation
    log_feature_skewness = skew(feature_log)

    # Return the skewness values before and after transformation
    return feature_skewness, log_feature_skewness

