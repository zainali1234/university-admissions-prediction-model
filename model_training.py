import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

#function that creates a dataframe using the .csv Kaggle dataset
def create_df():
    #creates dataframe
    file_path = 'Admission_Predict.csv'
    df = pd.read_csv(file_path)
    
    #renames columns within df
    df.rename(columns={'Serial No.': 'SerialNum', 'GRE Score': 'GREScore', 'TOEFL Score':'TOEFLScore',
                       'LOR ': 'LOR','University Rating': 'UniRating', 'Chance of Admit ': 'AdmissionChance'}, inplace=True)
    df.set_index('SerialNum', inplace=True)
    return df

#function that uses dataframe to create and train a pipeline model for predictions
def create_model(df):
    #seperates feature and target prediction dataframes
    X = df.drop("AdmissionChance", axis=1)
    y = df.AdmissionChance
    
    #creates a pipeline that defines pre-processing steps and the XGBRegressor model 
    model = XGBRegressor(n_estimators=500, random_state=1)

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='constant')),
                                  ('model', model)])
    
    #uses cross-validation to test pipeline model's pred. accuracy (variable will not be used since dataset and model are already validated)
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=8, scoring='neg_mean_absolute_error')
    my_pipeline.fit(X, y)
    return my_pipeline
