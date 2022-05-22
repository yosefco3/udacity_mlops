"""
This is the Python Test for the churn_library.py module.

This module will be used to test
    1. import_data
    2. peform_eda
    3. encode_data
    4. perform_feature_engineering
    5. train_test_model

Author: Yosef Cohen
Date: 22/5/2022
"""
import logging
import os
import shutil
from churn_library import (PLOTS_TO_EDA, import_data, perform_eda,
                          encoder_helper, perform_feature_engineering, 
                          train_models)

DATA_PATH = "./data/bank_data.csv"
TESTING_IMAGES_PATH = "images/testing"
TESTING_MODELS_PATH = "models/testing"
LOGGING_PATH = './logs/churn_library.log'


if os.path.exists(LOGGING_PATH):
    os.remove(LOGGING_PATH)

logging.basicConfig(
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=LOGGING_PATH,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def path_preprocessing(path):
    '''
    1. creates path for testing and delete all files in it.
    2. delete old log file
    '''
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as err:
            logging.error("Failed to create testing path")
            raise err
    for root, dirs, files in os.walk(path):
        try:
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        except OSError as err:
            logging.error("Failed to empty testin path")
            raise err


def test_create_paths():
    path_preprocessing(TESTING_IMAGES_PATH)
    path_preprocessing(TESTING_MODELS_PATH)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    '''
    test perform eda function
    '''
    perform_eda(df, path=TESTING_IMAGES_PATH)
    for _, cols in PLOTS_TO_EDA.items():
        for col in cols:
            try:
                assert os.path.exists(
                    f"{TESTING_IMAGES_PATH}/{col}_plot.png"
                )
                logging.info(f"Testing Eda: SUCCESS {col} plot")
            except AssertionError as err:
                logging.error(f"{col} eda plot file doesnt exist!")
                raise err
    try:
        assert os.path.exists(
            f"{TESTING_IMAGES_PATH}/df_heatmap.png"
        )
        logging.info("Testing Eda: SUCCESS heatmap plot")
    except AssertionError as err:
        logging.error(f"df heatmap plot file doesnt exist!")
        raise err


def test_encoder_helper(df):
    '''
    test encoder helper
    '''
    encoded = encoder_helper(df)
    cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Churn',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    try:
        assert df['Churn'].isin([0, 1]).all()
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: target column contains more than 1 & 0.")
        raise err
    try:
        assert sorted(encoded.columns) == sorted(cols)
        logging.info("Testing encoder_helper: SUCCESS, the columns are OK.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: the columns are wrong.")
        raise err
    try:
        assert encoded.shape[0] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoded_helper: The encoded df doesn't appear to have rows")
        raise err
    return encoded


def test_perform_feature_engineering(df):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    try:
        assert X_train.shape[1], len(y_train.shape) == (19, 1)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The shapes of X and y aren't OK.")
        raise err
    try:
        assert X_train.shape[0] > X_test.shape[0] and y_train.shape[0] > y_test.shape[0]
    except AssertionError as err:
        logging.error('''
        Testing perform_feature_engineering:
        the number of train rows isnt bigger than the test rows!
        ''')
        raise err
    logging.info(
        "Testing perform_feature_engineering: SUCCESS, train and test dfs seems OK.")


def test_train_models(df):
    '''
    test train_models
    '''
    train_models(
        df,
        load_models=False,
        path=TESTING_MODELS_PATH,
        images_path=TESTING_IMAGES_PATH)
    try:
        assert os.path.exists(
            f"{TESTING_MODELS_PATH}/rfc_model.pkl"
        ) and os.path.exists(
            f"{TESTING_MODELS_PATH}/logistic_model.pkl"
        )
        logging.info(f"Testing train_models: SUCCESS models are created")
    except AssertionError as err:
        logging.error(f"models didin't created!")
        raise err
    '''
    now we will check the version when we loaded ready to use models from models directory.
    first we will delete the images we created, it will create them again.
    '''
    models_plots = ['features_importance.png',
                    'Loistic_regression_classification_report.png',
                    'LR&RF_roc_curves.png',
                    'Random_forest_classification_report.png']
    for plot in models_plots:
        try:
            os.remove(f"{TESTING_IMAGES_PATH}/{plot}")
        except OSError as err:
            logging.error(f"{plot} didnt created")
            raise err
    try:
        train_models(
            df,
            load_models=True,
            path=TESTING_MODELS_PATH,
            images_path=TESTING_IMAGES_PATH)
        logging.info(
            "Testing train_models: SUCCESS loaded ready models successfuly")
    except Exception as err:
        raise err
    for plot in models_plots:
        try:
            os.path.exists(f"{TESTING_IMAGES_PATH}/{plot}")
            logging.info(f"Testing train_models: SUCCESS {plot} has created")
        except OSError as err:
            logging.error(f"{plot} didnt created")
            raise err

# def cleanup():
#     shutil.rmtree(TESTING_IMAGES_PATH)
#     shutil.rmtree(TESTING_MODELS_PATH)


if __name__ == "__main__":
    test_create_paths()
    test_import(import_data)
    df = import_data(DATA_PATH)
    test_eda(df)
    encoded_df = test_encoder_helper(df)
    test_perform_feature_engineering(encoded_df)
    test_train_models(encoded_df)
    print("All tests passed! see churn_library.log file for details.")
#     cleanup()
