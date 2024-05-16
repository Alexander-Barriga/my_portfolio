# native tools
from collections import defaultdict
import sys
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np

# feature engineering
from sklearn.preprocessing import StandardScaler

# linear models
from sklearn.linear_model import LogisticRegression

# non-linear models
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from deep_learning import BuildNeuralNetArchitecture

# modeling building tools
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score,fbeta_score,  precision_recall_curve, PrecisionRecallDisplay)

# custom functions, classes, etc ...
from eda_helper_functions import identify_categorical_features


def f_beta_score(pre, rec, beta):
    """
    Calculates the harmonic mean of Precision and Recall a.k.a the F-Score.

    Note
    ----
    where β is chosen such that recall is considered β times as important as precision.

        β = 0.5 -> precision is weighted twice as much as recall
        β = 1.5 -> precision is weighted half as much as recall
        β = 1.0 -> precision is weighted equally as recall
    """
    return (1 + beta ** 2) * (pre * rec) / ((beta ** 2 * pre) + rec)


class TrainModels(object):

    def __init__(self, x_train, y_train, x_test, y_test, scoring="accuracy"):
        """
        This class instantiates sklearn pipeline objects that transform categorical
        and numerical features independently and train a default or custom list of ML
        classification models.

        x_train: list or array
        y_train: list or array
        x_test: list or array
        y_test: list or array
        scoring: str
            Name of valid sklearn classification scoring metric
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_x_cols = self.x_train.shape[1]
        self.scoring = scoring

        self.modeling_results = defaultdict(dict)
        self.best_pipeline = {}

        # create lists for model object and parameter dicts
        lr = [LogisticRegression(max_iter=1000, solver="liblinear"),
              {"model__penalty": ["l1", "l2"]}]

        rf = [RandomForestClassifier(n_jobs=-1),
              {"model__n_estimators": [100, 1000],
               "model__max_depth": [None, 6, 12]}]

        lightbgt = [lgb.LGBMClassifier(n_jobs=-1),
                    {'model__num_leaves': [31],
                     "model__max_depth": [-1],
                     "model__learning_rate": [0.01, 0.1],
                     "model__reg_alpha": [0.0, 0.01],
                     "model__reg_lambda": [0.0, 0.01],
                     "model__n_estimators": [100, 500]}]

# TODO: add deep learning models 
#         dense_network = [BuildNeuralNetArchitecture(),
#                          {"model__n_dense_hidden_layers": [1, 2],
#                           "model__n_inputs":[self.n_x_cols],
#                           "model__first_dense_hidden_layer_nodes": [500],
#                           "model__last_dense_hidden_layer_nodes": [10],
#                           "model__output_layer_nodes": [1],
#                           "model__output_layer_activation":["sigmoid"]}]

        self.model_dict = {
            "LogisticRegression": lr,
            "RandomForestClassifier": rf,
            "LGBMClassifier": lightbgt
        }
        #     "dense_network": dense_network
        # }

        # set n_jobs for gridsearch arg to 1 for parallelizable models
        # that are already set to n_jobs=-1
        self.n_jobs_for_gridsearch = {"LogisticRegression": -1,
                                      "RandomForestClassifier": 1,
                                      "LGBMClassifier": 1,
                                      "dense_network": -1}

        # portion of pipeline with data transformers
        self.preprocessor = None

    def create_transform_portion_of_pipeline(self, df):
        """
        Creates feature engineering portion of pipeline with separate transformers
        for categorical and numerical features.

        Parameters
        ----------
        df: pandas dataframe
            Used to identify categorical and numerical features

        Returns
        -------
        None
        """

        msg = "Building data transformer portion of the pipeline..."
        logging.info(msg)

        # get names of categorical features
        categorical_feats = identify_categorical_features(df, threshold=2, target_feat_name=None)

        # create boolean mask for numerical features
        numerical_feats_mask = ~ df.columns.isin(categorical_feats)

        # remove boolean value for y_col which appears at the end
        numerical_feats_mask = numerical_feats_mask[:-1]

        # create pipeline for numerical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # create transformer pipeline used prior to modeling
        self.preprocessor = ColumnTransformer(
            # categorical features will pass through untransformed
            remainder="passthrough",
            # all numerical features will pass through the numeric_transformer pipeline
            transformers=[('num', numeric_transformer, numerical_feats_mask)]
        )

    def grid_search_model(self, model, model_name, params):
        """
        Instantiate pipeline and grid search objects, trains a single pipeline with model.

        model: object
            A sklearn grid search compatible ML model

        model_name: str
            Valid model name which can be in in the keys of self.model_dict

        params: dict
            keys: name of model hyper-parameters
            values: list of hyper-parameters values

        Returns
        --------
        None
        """
        pipe_args = [
            ("preprocessor", self.preprocessor),
            ("model", model)]

        pipe = Pipeline(pipe_args)

        # select value based on model that is being fitted
        n_jobs = self.n_jobs_for_gridsearch[model_name]

        # create grid search
        gs = GridSearchCV(pipe,
                          params,
                          verbose=1,
                          cv=3,
                          n_jobs=n_jobs,
                          scoring=self.scoring)

        # run grid search
        gs.fit(self.x_train, self.y_train)

        # get results
        test_score = gs.score(self.x_test, self.y_test)

        # store results
        self.modeling_results[model_name]["test_score"] = test_score
        self.modeling_results[model_name]["gs_obj"] = gs

        # log best performing pipeline
        self.log_best_scoring_pipeline(model_name, test_score)

        msg = "Best {0} has test score {1:.4}".format(model_name, test_score)
        logging.info(msg)

    def grid_search_models(self, model_list=None):
        """
        Convenience method provides a loop around self.grid_search_model()

        Parameters
        ----------
        model_list: None or list of strs
            Names of classifiers to train, options are:
                - LogisticRegression
                - RandomForestClassifier
                - LGBMClassifier

        Returns
        -------
        None
        """

        # if user doesn't provide names of models to train
        if model_list is None:
            model_list = self.model_dict.keys()

        for model_name in model_list:

            model_params = self.model_dict[model_name]
            model = model_params[0]
            param_dict = model_params[1]

            msg = "Training {0} ...".format(model_name)
            logging.info(msg)

            self.grid_search_model(model, model_name, param_dict)

    def log_best_scoring_pipeline(self, model_name, test_score):
        """
        Logs best scoring pipeline for easy reference.

        Parameters
        ----------
        model_name: str
            Name of model for which performance scores are being logged.

        test_score: float
            Performance score of model for test data

        Returns
        -------
        None
        """
        # log best model
        if len(self.best_pipeline) == 0:
            self.best_pipeline["model_name"] = model_name
            self.best_pipeline["best_test_score"] = test_score
        elif len(self.best_pipeline) != 0:
            if self.best_pipeline["best_test_score"] < test_score:
                self.best_pipeline["model_name"] = model_name
                self.best_pipeline["best_test_score"] = test_score

        msg = "Logging {} as best performing model.".format(model_name)
        logging.info(msg)

    def return_best_pipeline(self):
        """
        Returns Grid search object with best performing pipeline and best scoring model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        best_pipeline_name = self.best_pipeline["model_name"]
        # return grid search object with pipeline
        return self.modeling_results[best_pipeline_name]["gs_obj"]

    def create_metric_df(self, adj_threshold=False):
        """
        Returns pandas data farme with model names as rows and metric scores as columns

        Parameters
        ----------
        adj_threshold: boolean
            Whether to recalibrate probability threshold used for model classifications.
            Default for classification models is usually set at 0.5
        """

        data = []
        index = []
        columns = ["Accuracy", "Precision", "Recall", "F1_Score"]

        if adj_threshold:
            columns.append("New_Threshold")

        for model_name, pipeline in self.modeling_results.items():

            model = pipeline["gs_obj"]
            index.append(model_name)
            model_scores = []

            if adj_threshold:
                y_pred, threshold = self.adjust_model_predictions(model)
            else:
                # index for grid search object and get predictions
                y_pred = model.predict(self.x_test)

            # score predictions
            acc = accuracy_score(self.y_test, y_pred)
            pre = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            if adj_threshold:
                data.append([acc, pre, rec, f1, threshold])
            else:
                data.append([acc, pre, rec, f1])

        return pd.DataFrame(data=data, index=index, columns=columns).round(4)

    def get_adj_model_predictions(self, new_threshold, positive_class_prob):
        """
        Adjust model predictions using a new binary classification threshold.

        Parameters
        ----------
        new_threshold: float
            Adjusted probability threshold used to update model classifications

        positive_class_prob: list of floats
            Model probability scores for Positive class

        Returns
        -------
        List with adjusted model classifications using adjusted probability threshold
        """

        return [1 if p >= new_threshold else 0 for p in positive_class_prob]

    def adjust_model_predictions(self, model, beta=1.05):
        """
        Adjust model predictions by shifting the classification threshold away from the default of 0.5
        Take the weight average score between Precision and Recall using the F Beta Score

        model: object
            ML classification model

        beta: float
            Value to use in F Beta Scoring metric

        Note
        ----
        β is chosen such that recall is considered β times as important as precision.

            β = 0.5 -> precision is weighted twice as much as recall
            β = 1.5 -> precision is weighted half as much as recall
            β = 1.0 -> precision is weighted equally as recall
        """

        # Todo: a stretch goal
        #beta = self.search_for_best_beta(model)

        y_pred_prob = model.predict_proba(self.x_test)
        positive_class_prob = y_pred_prob.T[1]

        precision, recall, thresholds = precision_recall_curve(self.y_test, positive_class_prob, pos_label=1)

        # convert to f score
        f_scores = f_beta_score(precision, recall, beta)

        # locate index max f score
        max_f_score_index = np.argmax(f_scores)

        # new threshold maximizes both precision and recall
        new_threshold = thresholds[max_f_score_index]

        y_pred_adj = self.get_adj_model_predictions(new_threshold, positive_class_prob)

        return y_pred_adj, new_threshold

    def create_precision_recall_df(self, model, plot=False):
        """
        Creates a dataframe populated with precision recall values that vary wrt the
        classification threshold.

        Used for plotting.

        Parameters
        ----------
        model: object
            Trained classification model

        plot: Boolean
            Whether to plot precision recall curve

        Returns
        -------
        pandas dataframe
        """
        y_pred_prob = model.predict_proba(self.x_test)
        positive_class_prob = y_pred_prob.T[1]

        precision, recall, thresholds = precision_recall_curve(self.y_test, positive_class_prob, pos_label=1)

        precision = precision.reshape((precision.shape[0], 1))
        recall = recall.reshape((recall.shape[0], 1))

        data = np.concatenate([precision, recall], axis=1)

        columns = ["precision", "recall"]
        df_pre_rec = pd.DataFrame(data=data, columns=columns)

        if plot:
            title = "Precision vs. Recall"
            df_pre_rec.plot(x="recall", y="precision", grid=True, figsize=(10, 5), title=title)

        return df_pre_rec

    def create_feature_importance_df(self, model, x_cols,  normalize=True, plot=False):
        """

        Parameters
        ----------
        model: object
            Trained classification model

         x_cols: list of str
            Names of input features

         normalize: boolean
            Whether to normalize the feature importance values

         plot: boolean
            Whether to plot feature importance

        Returns
        -------
        df_feat_imp: pandas dataframe
        """
        feat_importance = model.feature_importances_
        if normalize:
            feat_importance = feat_importance / feat_importance.sum()

        df_feat_imp = pd.DataFrame(feat_importance, index=x_cols)
        df_feat_imp.rename(columns={0: "feat_imp"}, inplace=True)

        if plot:
            title = "Feature Importance in detecting Fraud"
            df_feat_imp.sort_values("feat_imp", ascending=False).plot(kind="bar", grid=True, legend=True,
                                                                      figsize=(25, 5), use_index=True, title=title);

        return df_feat_imp

    # def search_for_best_beta(self, model):
    #     """
    #     Selects beta value that balances precision and recall scores
    #     """
    #
    #     betas = np.arange(0.1, 3.0, 0.1)
    #     fb_scores = []
    #
    #     # convert to f score
    #     f_scores = f_beta_score(precision, recall, beta)
    #
    #     # get predictions
    #     y_pred = model.predict(self.x_test)
    #
    #     for b in betas:
    #         # get corresponding f_beta score
    #         fb_score = fbeta_score(self.y_test, y_pred, beta=b)
    #         # store
    #         fb_scores.append(fb_score)
    #
    #     # locate index max f score
    #     max_f_score_index = np.argmax(fb_scores)
    #
    #     # returns beta threshold that corresponds to max F score
    #     return betas[max_f_score_index]





