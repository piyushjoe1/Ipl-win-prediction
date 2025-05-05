import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_team_strength(data_path='data/Cricket_data.csv'):
    df = pd.read_csv(data_path)
    df['1st_inning_runs'] = df['1st_inning_score'].str.extract(r'(\d+)/\d+').astype(float)

    team_strength = df.groupby('home_team')['1st_inning_runs'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=team_strength.index, y=team_strength.values)
    plt.xticks(rotation=45)
    plt.title("🏏 Team Batting Strength (Avg 1st Innings Score)")
    plt.ylabel("Avg Score")
    plt.xlabel("Team")
    plt.tight_layout()
    plt.show()

