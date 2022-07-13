import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

def create_df():
    file_path = 'Admission_Predict.csv'
    df = pd.read_csv(file_path)
    df.rename(columns={'Serial No.': 'SerialNum', 'GRE Score': 'GREScore', 'TOEFL Score':'TOEFLScore',
                       'LOR ': 'LOR','University Rating': 'UniRating', 'Chance of Admit ': 'AdmissionChance'}, inplace=True)
    df.set_index('SerialNum', inplace=True)
    return df

def create_model(df):
    X = df.drop("AdmissionChance", axis=1)
    y = df.AdmissionChance
    model = XGBRegressor(n_estimators=500, random_state=1)

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='constant')),
                                  ('model', model)])

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=8, scoring='neg_mean_absolute_error')
    my_pipeline.fit(X, y)
    return my_pipeline
