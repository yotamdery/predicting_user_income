import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import Tuple

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
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"]).drop_duplicates()

    # Output the highly correlated feature pairs
    high_corr_df = high_corr_df.sort_values(by='Correlation', ascending=False)
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

