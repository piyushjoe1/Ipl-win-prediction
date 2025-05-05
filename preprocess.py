import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Drop rows with missing winner
    df = df[df['winner'].notna()]

    # Replace short forms with full team names
    team_name_map = {
        "CSK": "Chennai Super Kings",
        "MI": "Mumbai Indians",
        "RCB": "Royal Challengers Bangalore",
        "KKR": "Kolkata Knight Riders",
        "DC": "Delhi Capitals",
        "SRH": "Sunrisers Hyderabad",
        "RR": "Rajasthan Royals",
        "PBKS": "Punjab Kings",
        "KXIP": "Punjab Kings",  # Legacy name
        "GT": "Gujarat Titans",
        "LSG": "Lucknow Super Giants"
        # We intentionally exclude RPS, GL, KTK, etc.
    }

    columns_to_fix = ['home_team', 'away_team', 'winner', 'toss_won']
    for col in columns_to_fix:
        df[col] = df[col].replace(team_name_map)

    # Remove rows with unknown/defunct teams
    known_teams = [
        "Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders",
        "Royal Challengers Bangalore", "Delhi Capitals", "Sunrisers Hyderabad",
        "Rajasthan Royals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"
    ]
    for col in columns_to_fix:
        df = df[df[col].isin(known_teams)]

    # Clean venue names → extract city from stadium name
    df['venue_name'] = df['venue_name'].str.extract(r',\s*(\w+)$')[0]

    # Extract runs from inning scores
    df['1st_inning_runs'] = df['1st_inning_score'].str.extract(r'(\d+)/\d+').astype(float)
    df['2nd_inning_runs'] = df['2nd_inning_score'].str.extract(r'(\d+)/\d+').astype(float)

    # All known labels
    all_teams = known_teams
    all_venues = [
        "Mumbai", "Chennai", "Kolkata", "Delhi", "Bangalore", "Bengaluru", "Ahmedabad",
        "Lucknow", "Hyderabad", "Jaipur", "Pune", "Raipur", "Visakhapatnam",
        "Dharamsala", "Nagpur", "Indore", "Ranchi", "Cuttack", "Kanpur", "Mohali"
    ]
    all_decisions = ['BAT FIRST', 'BOWL FIRST']
    df = df[df['venue_name'].isin(all_venues)]
    le = LabelEncoder()
    le.fit(all_teams + all_venues + all_decisions)

    # Encode categorical features
    df['home_team_enc'] = le.transform(df['home_team'].astype(str))
    df['away_team_enc'] = le.transform(df['away_team'].astype(str))
    df['venue_enc'] = le.transform(df['venue_name'].astype(str))
    df['toss_decision_enc'] = le.transform(df['decision'].astype(str))
    df['toss_winner_enc'] = le.transform(df['toss_won'].astype(str))
    df['winner_enc'] = le.transform(df['winner'].astype(str))

    return df, le
