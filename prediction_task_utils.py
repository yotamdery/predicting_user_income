import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import Tuple

def apply_log_transformation(feature: pd.Series) -> pd.Series:
    # Apply log transformation
    feature_log = np.log1p(feature)  # log1p to handle zeros safely
    return feature_log

def check_skewness(feature: pd.Series) -> Tuple[float, float]:
    # Compute skewness of the feature variable
    feature_skewness = skew(feature)

    # Apply log transformation
    feature_log = apply_log_transformation(feature)

    # Compute skewness after transformation
    log_feature_skewness = skew(feature_log)

    # Return the skewness values before and after transformation
    return feature_skewness, log_feature_skewness

