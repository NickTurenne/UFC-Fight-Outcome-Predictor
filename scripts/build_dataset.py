import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from src.clean import clean_fight_event_data, clean_fighter_data
from src.fighter_states import initialize_fighter, update_fighter_states, get_red_stats_for_rolling_avg, get_blue_stats_for_rolling_avg
from src.feature_engineering import build_features, add_rolling_averages_to_row

def main():
    fighter_states = {}
    fighter_last_3 = {}

    df_event_fights = clean_fight_event_data()
    df_fighter = clean_fighter_data()
    
    # Construct the default blank fighter states for every fighter
    # Fighter states are used for incremental fight stat increases to avoid data leakage
    for fighter in df_fighter.itertuples(index=False):
        fighter_states[fighter.id] = initialize_fighter(fighter)

    # Construct the dataset and build up fighter states using fighter_id and stats
    dataset_list = [] # List of dictionaries with each representing an entry

    for fight in df_event_fights.itertuples(index=False):
        red_id = fight.r_id
        blue_id = fight.b_id

        # Check deuque
        if red_id not in fighter_last_3:
            fighter_last_3[red_id] = deque(maxlen=3)
        if blue_id not in fighter_last_3:
            fighter_last_3[blue_id] = deque(maxlen=3)

        # Bulid the features and add label
        features = build_features(fighter_states[red_id], fighter_states[blue_id], fight, fighter_last_3, red_id=red_id, blue_id=blue_id)
        if fight.winner_id == red_id:
            features["winner"] = 1
        else:
            features["winner"] = 0

        dataset_list.append(features)

        # Build the mirrored features and add label to add more data and remove corner bias
        features = build_features(fighter_states[blue_id], fighter_states[red_id], fight, fighter_last_3, red_id=blue_id, blue_id=red_id)
        if fight.winner_id == blue_id:
            features["winner"] = 1
        else:
            features["winner"] = 0

        dataset_list.append(features)

        # Update the fighter states based on the fight
        update_fighter_states(fighter_states[red_id], fighter_states[blue_id], fight)
        fighter_last_3[red_id].append(get_red_stats_for_rolling_avg(fight))
        fighter_last_3[blue_id].append(get_blue_stats_for_rolling_avg(fight))

    # Make dataset and save
    dataset = pd.DataFrame(dataset_list)
    project_root = Path().resolve().parents[0]
    sys.path.append(str(project_root))
    dataset.to_csv(Path(project_root) / "UFC-Fight-Outcome-Predictor"  / "data" / "processed" / "dataset.csv", index=False)
    print("Dataset saved!")


if __name__ == "__main__":
    main()