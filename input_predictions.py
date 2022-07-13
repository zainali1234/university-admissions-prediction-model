from model_training import *
import pandas as pd

original_df = create_df()
input_df = original_df[0:0]
my_pipeline = create_model(original_df)

new_row = {'GREScore': [float(input("Enter GREScore: "))],
           'TOEFLScore': [float(input("Enter TOEFLScore: "))],
           'UniRating': [float(input("Enter UniRating: "))],
           'SOP': [float(input("Enter SOP: "))],
           'LOR': [float(input("Enter LOR: "))],
           'CGPA': [float(input("Enter CGPA: "))],
           'Research': [float(input("Enter Research: "))]}

row_df = pd.DataFrame.from_dict(new_row)
input_df = pd.concat([input_df, row_df])

dataCols = ['GREScore', 'TOEFLScore', 'UniRating', 'SOP',
                              'LOR', 'CGPA', 'Research']

original_df = original_df.drop("AdmissionChance", axis=1)
input_df = input_df.drop("AdmissionChance", axis=1)

preds = round(((my_pipeline.predict(input_df))[0] * 100), 3)

print("You have a " + str(preds) + "% chance of being admitted to a university with a rating of "
      + str(new_row['UniRating'][0]) + ".")

original_means = original_df[dataCols].mean()
input_df_means = input_df[dataCols].mean()

mean_diff = ((abs(input_df_means - original_means) / ((input_df_means + original_means) / 2)) * 100).mean()
print("Your selected values differ from the mean by exactly: " + str(round(mean_diff, 3)) + "%")

