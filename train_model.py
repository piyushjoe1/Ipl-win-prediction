import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess

def train_model():
    df, le = load_and_preprocess("data/Cricket_data.csv")

    X = df[['home_team_enc', 'away_team_enc', 'venue_enc', 'toss_decision_enc', 'toss_winner_enc']]
    y = df['winner_enc']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/ipl_model.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')

    print("Model trained and saved to /model")

if __name__ == "__main__":
    train_model()
