from copy import copy
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def load_dataset(file_path):
    """
    Loads dataset from csv file into pandas dataframe

    Parameters
    ----------
    file_path: str

    Returns
    -------
    pandas dataframe
    """

    df = pd.read_csv(file_path)
    return df.drop(columns=["Unnamed: 0"])


def drop_low_var_features(df, threshold=0.0):
    """
    Drops features with a variance threshold that doesn't surpass <threshold>.

    Parameters
    ---------
    df: pandas dataframe
    threshold: float or int

    Returns
    -------
    pandas dataframe
    """
    # get mask for filtering cols
    var_filter = VarianceThreshold(threshold=threshold)
    var_filter.fit(df)
    keep_cols_mask = var_filter.get_support()

    # feat names to keep
    keep_cols = df.columns[keep_cols_mask]

    # print number of cols dropped
    n_feats_dropped = df.shape[1] - len(keep_cols)
    msg = "Number of features dropped {0}. Number of features remaining {1}".format(n_feats_dropped, len(keep_cols))
    logging.info(msg)

    # return dataframe with dropped cols
    return df[keep_cols]


def identify_categorical_features(df, threshold=5, target_feat_name=None):
    """
    Returns a list with the names of features that had less than <threshold> unique values.

    Parameters
    ----------
    df: pandas dataframe
    threshold: int
    target_feat_name: boolean or str
        name of target feature to remove before identifying categorical features

    Returns
    -------
    cate_feats: list of strings
        names of features believed to the categorical
    """

    x_cols = df.columns.values.tolist()

    if target_feat_name is not None:
        x_cols.remove(target_feat_name)

    cate_feats = []

    for col in x_cols:
        n_unique_vals = len(df[col].unique())

        if n_unique_vals <= threshold:
            cate_feats.append(col)

    return cate_feats

def get_continue_features(df, categorical_feat_names, y_col_name=None):
    """
    Returns list with names of features with continuous values

    Parameters
    ----------
    df: pandas dataframe
    categorical_feat_names: list of str
        Names of categorical features

    y_col_name: None or str
        Name of target variable
        Needs to be provided only if target variable isn't present in categorical_feat_names
        
    Returns
    -------
    continuous_feats: list of str
    """
    cate_feat_names = copy(categorical_feat_names)
    if y_col_name is not None:
        cate_feat_names.remove(y_col_name)
    continuous_feats_mask = ~df.columns.isin(cate_feat_names)
    continuous_feats = df.columns[continuous_feats_mask]
    return continuous_feats


def statistical_outlier_removal(df_data, bias="mean", n_std=3):
    """
    Removes points (on a column by column basis) that exist outside of acceptable upper and lower bound region
    defined by X number of standard deviations above and below the bias (mean or median).

    Note
    ----
    This approach applies a type of survival analysis in that the samples for column X that exist
    within the acceptable region have their indicies passed to the next column.

    Parameters
    ----------
    df_data: pandas dataframe

    bias: str
        Acceptable values are "mean" or "50%" for median

    n_std: int or float
        Typical values 1, 2, and 3

    Returns
    -------
    pandas dataframe
    """

    # get list with categorical feature names (including targets)
    categorical_feats = identify_categorical_features(df_data,
                                                      threshold=5,
                                                      target_feat_name=None)

    real_valued_feats = df_data.drop(columns=categorical_feats).columns

    df = df_data.copy()
    df_stats = df[real_valued_feats].describe()

    # apply outlier filter only to real valued features
    for col in real_valued_feats:
        # get dist statistics
        average = df_stats[col][bias]
        std = df_stats[col]["std"]

        # create bounding regions
        upper_bound = average + (std * n_std)
        lower_bound = average - (std * n_std)

        # create masks for bounding regions
        upper_bound_mask = df[col] < upper_bound
        lower_bound_mask = df[col] > lower_bound

        # filter points that exists outside of the bounding regions
        indicies = df[col][upper_bound_mask & lower_bound_mask].index.values

        # filter out outlier points
        ind_mask = df.index.isin(indicies)
        df = df[ind_mask]

    return df


def filter_features_using_correlation(df, corr_method, corr_thresh, y_col_name):
    """
    Identifies which features have a correlation value wrt <y_col_name> that is greater
    than the threshold provide. Features with corr values above the threshold will have
    their names returned in a list.

    Parameters
    ----------
    df: pandas datafarme

    corr_method: str
        Same values permited by df.corr(), i.e. "pearson", "spearman", "kendall"

    corr_thresh: int or float
        Features with a corr value less than this will will not have their name returned

    y_col_name: str
        Name of target label

    Return
    ------
    df_corr: pandas dataframe
        correlation matrix

    strong_corr_feats: list of str
        Names of features with a corr value greater than <corr_thresh>
    """

    # drop categorical features then calculate corr
    df_corr = df.corr(method=corr_method)
    df_corr = df_corr[[y_col_name]].sort_values(y_col_name, ascending=False)
    df_corr = df_corr.iloc[1:]  # remove the corr between "targets" and itself

    # by applying a threshold to the abs value of the correlations
    # we can filter out weak correlating features and keep the stronger ones
    corr_mask = df_corr.abs() > corr_thresh
    strong_corr_feats = df_corr.abs()[corr_mask].index

    return df_corr, strong_corr_feats

