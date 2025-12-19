import sys
import pandas as pd
import numpy as np
from src.fighter_states import initialize_fighter, update_fighter_states
from src.feature_engineering import build_features
from pathlib import Path

def main():
    fighter_states = {}

    # Load data
    project_root = Path().resolve().parents[0]
    sys.path.append(str(project_root))

    df_fighter = pd.read_csv(Path(project_root) / "UFC-Fight-Outcome-Predictor" / "data" / "raw" / "fighter_details.csv")
    df_fight = pd.read_csv(Path(project_root) / "UFC-Fight-Outcome-Predictor"  / "data" / "raw" / "fight_details.csv")
    df_event = pd.read_csv(Path(project_root) / "UFC-Fight-Outcome-Predictor" / "data" / "raw" / "event_details.csv")

    # Basic data inspection
    df_fighter.info()
    df_fight.info()
    df_event.info()

    # Format DateTime
    df_event["date"] = pd.to_datetime(df_event["date"], format="%B %d, %Y")
    df_fighter["dob"] = pd.to_datetime(df_fighter["dob"], format="%b %d, %Y")
    # Drop fights with no winner 
    df_event = df_event.dropna(subset=["winner", "winner_id"])
    # Drop fights before Unified Rules of MMA (UFC 28, November 17, 2000)
    cutoff_date = pd.Timestamp("2000-11-17")
    df_event = df_event[df_event["date"] >= cutoff_date]
    # Sort the fights from oldest to newest for dataset construction
    df_event = df_event.sort_values(by="date")

    # Join events with fights
    df_event_fights = pd.merge(df_event, df_fight, on="fight_id")
    print(len(df_event_fights))
    
    # Construct the default blank fighter states for every fighter
    # Fighter states are used for incremental fight stat increases to avoid data leakage
    for fighter in df_fighter.itertuples(index=False):
        fighter_states[fighter.id] = initialize_fighter(fighter)

    # Construct the dataset and build up fighter states using fighter_id and stats
    dataset_list = [] # List of dictionaries with each representing an entry

    for fight in df_event_fights.itertuples(index=False):
        red_id = fight.r_id
        blue_id = fight.b_id

        # Bulid the features and add label
        features = build_features(fighter_states[red_id], fighter_states[blue_id], fight)
        if fight.winner_id == red_id:
            features["winner"] = 1
        else:
            features["winner"] = 0

        dataset_list.append(features)

        # Update the fighter states based on the fight
        update_fighter_states(fighter_states[red_id], fighter_states[blue_id], fight)

    # Make dataset and save
    # dataset = pd.DataFrame(dataset_list)
    print(dataset_list[0])

    # Encode and then save again for model training




if __name__ == "__main__":
    main()