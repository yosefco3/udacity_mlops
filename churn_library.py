"""
The churn_library.py is a library of functions to find customers who are likely to churn.
This module will be used to:
    1. import_data
    2. peform_eda
    3. encode_data
    4. perform_feature_engineering
    5. train_test_model

Author: Yosef Cohen
Date: 22/5/2022
"""
import warnings
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
sns.set()

QT_FONTS = "/opt/conda/lib/fonts"
PATH = r"./data/bank_data.csv"
IMAGES_PATH = 'images/results'
MODELS_PATH = './models'
PLOTS_TO_EDA = {
        'categorical': ('Attrition_Flag', 'Marital_Status'),
        'quantitative': ('Customer_Age', 'Total_Trans_Ct'),
    }

# we added this line because there is an error otherwise
if not os.path.exists(QT_FONTS):
    os.mkdir(QT_FONTS)

# ignoring pandas warnings
warnings.filterwarnings("ignore")

# import libraries
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(path = PATH):
    '''
    1. returns dataframe for the csv found at pth.
    2. replaces the target categorical column ('Attrition_Flag') with numerical (Churn).
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(path)
    return df

def perform_eda(df, path = IMAGES_PATH):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    for col_type,cols in PLOTS_TO_EDA.items():
        for col in cols:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=200)
            if col_type == 'categorical':
                ax.set_title(f"{col} count plot")
                sns.countplot(df[col])
            elif col_type == 'quantitative':
                ax.set_title(f"{col} distribution")
                sns.distplot(df[col])
            fig.savefig(f"{path}/{col}_plot.png")
#             plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig(f"{path}/df_heatmap.png")


def encoder_helper(df, category_lst=None, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name.
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
#     first we will make 'Attrition_Flag' to a numeric variable "Churn".
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df.drop(labels=['Attrition_Flag'], axis=1, inplace=True)
    if category_lst is None:
        category_lst = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

    response = [f"{cat}_{response}" for cat in category_lst]
    proportions = {}
    for idx, cat in enumerate(category_lst):
        proportions[cat] = df.groupby(cat).mean()['Churn']
        df[response[idx]] = df[cat].apply(lambda x: proportions[cat].loc[x])
    return keep_cols(df)


def keep_cols(df):
    '''helper function to filter undesirable columns.
    we want numeriical columns with relationship to the target column.
    input:
        df - our dataframe.
    output:
        df - with the desireable columns.
    '''
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    df = df[numerical_cols]
    df.drop(labels=['Unnamed: 0', 'CLIENTNUM'], axis=1, inplace=True)
    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe.
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    X = df.drop(labels=['Churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def train_models(df, load_models = False, path = MODELS_PATH, images_path = IMAGES_PATH):
    '''
    train, store model results: images + scores, and store models
    images: ROC curves, classification reports.
    input:
              df
              load_models - if false, create models and train them.
                            if true, load models from models directory(faster).
    output:
              None
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    if load_models:
        try:
            rf = joblib.load(f'{path}/rfc_model.pkl')
            lrc = joblib.load(f'{path}/logistic_model.pkl')
        except FileNotFoundError as err:
            raise Exception('''
            Seems like the models are not exist. 
            change the 'load_models" parameter in the "train_models" function, to "False"
            ''') from err
    else:
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    #     training:
        cv_rfc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)

    #     saving models:
        rf = cv_rfc.best_estimator_
        joblib.dump(rf, f'{path}/rfc_model.pkl')
        joblib.dump(lrc, f'{path}/logistic_model.pkl')

#     ploting roc curves:
    plot_roc_models(lrc, rf, X_test, y_test, images_path)
#     creating predictions
    y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = create_preds(
        X_train, X_test, lrc, rf)
#     create classification report images:
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                images_path)
#     feature importance plot for rf model
    x_cols = X_train.columns
    feature_importance_plot(rf, x_cols, images_path)


def pred_from_model(model, X_train, X_test):
    '''
    helper function to create train and test predictions from a model.
    input:
        X_train
        X_test
    output:
        y_train_preds - train set predictions
        y_test_preds - test set predictions
    '''
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)
    return y_train_preds, y_test_preds


def create_preds(X_train, X_test, lr, rf):
    '''
    this function loads the models and makes predictions
    input:
        X_train,X_test - the features of the train and the test
        lr,rf - models.
    output:
        y_train_preds_lr = prdictions of lr training data,
        y_train_preds_rf = predictions of rf training data,
        y_test_preds_lr = predictions of lr testing data,
        y_test_preds_rf= predictions of rf testing data.

    '''
    y_train_preds_lr, y_test_preds_lr = pred_from_model(lr, X_train, X_test)
    y_train_preds_rf, y_test_preds_rf = pred_from_model(rf, X_train, X_test)
    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def clasification_report_model(
        model_name,
        y_train,
        y_test,
        y_preds_train,
        y_preds_test,
        path = IMAGES_PATH):
    '''
    helper function to create image classification report
    input:
        model_name: string
        y_train - real values train set.
        y_test - real values test set.
        y_preds_train - predictions train set
        y_preds_test - predictions test set
        path - path so save image

    '''
    fig = plt.figure(figsize=(10, 5))
    plt.axis([0, 10, 0, 10])
    plt.text(0, 10, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0, 5, str(classification_report(y_test, y_preds_test)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0, 4.5, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0, 0, str(classification_report(y_train, y_preds_train)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    fig.savefig(f"{path}/{model_name}_classification_report.png")


def plot_roc_models(lr, rf, X_test, y_test, path = IMAGES_PATH):
    '''
    plots ROC to both of our models
    input:
        lr, rf - models
        X_test - test feature set.
        Y_test - test target.
    output:
        None
    '''
    fig, axes = plt.subplots(1, 1, figsize=(15, 8))
    plot_roc_curve(lr, X_test, y_test, ax=axes, alpha=0.8)
    plot_roc_curve(rf, X_test, y_test, ax=axes, alpha=0.8)
    fig.savefig(f"{path}/LR&RF_roc_curves.png")


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                path = IMAGES_PATH):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    clasification_report_model(
        "Random_forest",
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        path)
    clasification_report_model(
        "Loistic_regression",
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        path)


def feature_importance_plot(
        model,
        x_cols,
        path = IMAGES_PATH):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_cols: columns of x set.
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_cols[i] for i in indices]

    # Create plot
    fig, _ = plt.subplots(1, 1, figsize=(20, 8))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(len(x_cols)), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(len(x_cols)), names, rotation=60)
    fig.tight_layout()
    fig.savefig(f"{path}/features_importance.png")


if __name__ == "__main__":
    churn_df = import_data(PATH)
    perform_eda(churn_df)
    encoded_df = encoder_helper(churn_df)
    train_models(encoded_df)
    print('nice job! all done :)')
