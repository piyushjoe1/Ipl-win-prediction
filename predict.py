import joblib
import numpy as np

def predict_winner(home_team, away_team, venue, toss_decision, toss_winner):
    model = joblib.load('model/ipl_model.pkl')
    le = joblib.load('model/label_encoder.pkl')

    input_data = np.array([
        le.transform([home_team])[0],
        le.transform([away_team])[0],
        le.transform([venue])[0],
        le.transform([toss_decision])[0],
        le.transform([toss_winner])[0]
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    return le.inverse_transform(prediction)[0]
