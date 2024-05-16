# native tools
import sys
from collections import Counter
import logging
logging.basicConfig(level=logging.INFO)

# for imbalanced classes
from imblearn.over_sampling import SMOTENC

# modeling building tools
from sklearn.model_selection import train_test_split

# custom functions, classes, etc ...
from eda_helper_functions import (drop_low_var_features,
                                  load_dataset,
                                  identify_categorical_features,
                                  statistical_outlier_removal,
                                  filter_features_using_correlation,
                                  get_continue_features)


class PrepDataForModeling(object):

    def __init__(self, data_path, y_col="targets"):
        """
        This class is a ETL pipeline that cleans, filters, and transforms data
        in preparation for a ML classification model.


        Parameters
        ----------
        data_path: str
            Location of raw data in a csv file
        y_col: str
            Name of target variable

        Returns
        -------
        None
        """

        self.data_path = data_path
        self.y_col = y_col
        self.x_cols = None
        self.df = None
        self.df_outliers_removed = None
        self.X_train_filtered = None
        self.X_test_filtered = None
        self.y_train_filtered = None
        self.y_test_filtered = None
        self.X_train_balanced = None
        self.y_train_balanced = None

    def load_data(self):
        """
        Loads data from csv file into a pandas data frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        msg = "Loading data from {}".format(self.data_path)
        logging.info(msg)
        self.df = load_dataset(self.data_path)

    def feature_selection(self, drop_cols=None):
        """

        Parameters
        ----------
        drop_cols: None or list of str
            Names of input features to drop

        Returns
        -------
        None
        """

        # drop all zero variance features from dataset
        # we might need to comeback and increase the variance threshold later
        self.df = drop_low_var_features(self.df, threshold=0.1)

        # drop low variance 3 category input feature
        if drop_cols is not None:
            self.df.drop(columns=drop_cols, inplace=True)

        msg = "Using statistical_outlier_removal to remove outliers"
        logging.info(msg)
        self.df_outliers_removed = statistical_outlier_removal(self.df, bias="mean", n_std=2)

    def split_data(self, test_size=0.70):
        """
        Splits feature inside of self.df into a training and test set.

        Parameters
        ----------
        test_size: float
            Proportion of data set used for the test set

        Returns
        -------
        None
        """

        msg = "Splitting data into training and test sets"
        logging.info(msg)

        # move data into arrays
        y = self.df_outliers_removed.targets.values
        self.x_cols = self.df_outliers_removed.columns.drop([self.y_col])
        x = self.df_outliers_removed[self.x_cols].values

        # split data into train and test sets
        self.X_train_filtered, self.X_test_filtered, self.y_train_filtered, self.y_test_filtered =\
            train_test_split(x, y, test_size=test_size)

    def balance_labels(self):
        """

        Uses Synthetic Minority Over-sampling Technique algorithm SMOTENC to balance imbalanced binary labels.

        Note
        ----
        SMOTENC documentation:
        https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTEN.html

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        msg = "Using SMOTENC to balance out labels in train set only."
        logging.info(msg)

        # list with names of categorical features
        categorical_feat_names = identify_categorical_features(self.df_outliers_removed,
                                                               threshold=5,
                                                               target_feat_name=self.y_col)

        # boolean mask indicating which features care categorical
        cate_feat_mask = self.df.columns.isin(categorical_feat_names)

        sm = SMOTENC(random_state=42,
                     categorical_features=cate_feat_mask,
                     k_neighbors=5) # number of neighbors to use when creating a synthetic sample

        # when modeling, only fit on training
        self.X_train_balanced, self.y_train_balanced = sm.fit_resample(self.X_train_filtered, self.y_train_filtered)

        msg = "Original imbalanced label count for training data {}".format(Counter(self.y_train_filtered))
        logging.info(msg)

        msg = "New balanced label count for training data {}".format(Counter(self.y_train_balanced))
        logging.info(msg)

    def gen_model_ready_data(self, return_data=False):
        """

        Main method used to load, clean, split, and transform raw data into model ready data.

        Parameters
        ----------
        return_data: boolean
            Whether to return train and test data sets.

        Returns
        -------
        X_train_filtered: array
        y_train_filtered: array
        X_test_filtered: array
        y_test_filtered: array
        """

        # loads csv file
        self.load_data()

        # filters features using several methods
        self.feature_selection()

        # splits data into train and test sets
        self.split_data()

        # balances imbalanced binary labels
        self.balance_labels()

        if return_data:
            return self.X_train_filtered, self.y_train_filtered, self.X_test_filtered, self.y_test_filtered