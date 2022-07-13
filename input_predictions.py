from model_training import *
import pandas as pd

#creates a dataframe using the .csv dataset
original_df = create_df()

#creates an empty copy of the dataframe for user input
input_df = original_df[0:0]

#Creates and fits the model using the dataset
my_pipeline = create_model(original_df)

#Prompts user for input
new_row = {'GREScore': [float(input("Enter GREScore: "))],
           'TOEFLScore': [float(input("Enter TOEFLScore: "))],
           'UniRating': [float(input("Enter UniRating: "))],
           'SOP': [float(input("Enter SOP: "))],
           'LOR': [float(input("Enter LOR: "))],
           'CGPA': [float(input("Enter CGPA: "))],
           'Research': [float(input("Enter Research: "))]}

#adds user-input row of data to the input df for predictions
row_df = pd.DataFrame.from_dict(new_row)
input_df = pd.concat([input_df, row_df])

dataCols = ['GREScore', 'TOEFLScore', 'UniRating', 'SOP',
                              'LOR', 'CGPA', 'Research']

#removes AdmissionChance column since it is the prediction target
original_df = original_df.drop("AdmissionChance", axis=1)
input_df = input_df.drop("AdmissionChance", axis=1)

#creates and stores prediction using user-input values
preds = round(((my_pipeline.predict(input_df))[0] * 100), 3)

#prints predictions to user
print("You have a " + str(preds) + "% chance of being admitted to a university with a rating of "
      + str(new_row['UniRating'][0]) + ".")

#finds and outputs the mean difference between the input data and imported dataset
original_means = original_df[dataCols].mean()
input_df_means = input_df[dataCols].mean()

mean_diff = ((abs(input_df_means - original_means) / ((input_df_means + original_means) / 2)) * 100).mean()
print("Your selected values differ from the mean by exactly: " + str(round(mean_diff, 3)) + "%")

