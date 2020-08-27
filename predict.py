#This function loads the model dump and predicts the hours based on input parameters

import joblib

def predict(df_test):
    #Loading the pipeline dump to transform and predict
    model=joblib.load('pipe.ml')
    predicted_hours=model.predict(df_test)
    return predicted_hours