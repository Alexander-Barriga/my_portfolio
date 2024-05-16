import argparse
from joblib import dump, load
from train_models import TrainModels
from prep_data import PrepDataForModeling
import logging
logging.basicConfig(level=logging.INFO)
              
def main(data_path, model_list, pipeline_save_path):
    """
    Load and prep data for ML modeling pipeline, save best pipeline to file.

    Parameters
    ----------
    data_path: str
        Location of raw data in a csv file format
    model_list: list of str
        Names of models to train (see complete list in train_models.py doc string
    pipeline_save_path: str
        Location to save best performing pipeline
    """
    # instantiate data prep class
    prep_data = PrepDataForModeling(data_path, y_col='targets')

    # prep data for modeling
    X_train, y_train, X_test, y_test = prep_data.gen_model_ready_data(return_data=True)


    # instantiate model training class 
    train_models = TrainModels(X_train, 
                               y_train, 
                               X_test, 
                               y_test)

    # create pipelines for handeling categorical and numerical data 
    train_models.create_transform_portion_of_pipeline(prep_data.df)

    # gridsearch pipelines/models
    train_models.grid_search_models(model_list)

    # get best pipeline (contains model within)
    best_pipeline = train_models.return_best_pipeline()

    # save best pipeline to file 
    msg = "Saving best pipeline to {}".format(pipeline_save_path)
    logging.info(msg)
    dump(best_pipeline, pipeline_save_path)
              

if __name__ == "__main__":
    
    # default list of models to train
    model_list = ["LogisticRegression", 
                  "RandomForestClassifier",
                  "LGBMClassifier"]
    
    # default location of data file
    data_path = "../Data/fraud_detection_bank_dataset.csv"
    
    # default location to save best pipeline
    pipeline_save_path = "../Models/best_pipeline.joblib"
              
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", 
                        type=str, 
                        help="Enter path of data file location",
                        default=data_path)
    
    parser.add_argument("--model_list", 
                        type=str, 
                        help="Enter list of names of models to train (see TrainModels doc string for list.)",
                        default=model_list)
    
    parser.add_argument("--pipeline_save_path", 
                        type=str, 
                        help="Enter path to save best performing pipeline",
                        default=pipeline_save_path)

    args = parser.parse_args()

    main(args.data_path,
         args.model_list,
         args.pipeline_save_path)
