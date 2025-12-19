import pandas as pd
import sys
from src.fighter_states import initialize_fighter
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
    # Drop fights with no winner 
    df_event = df_event.dropna(subset=["winner", "winner_id"])
    # Drop fights before Unified Rules of MMA (UFC 28, November 17, 2000)
    cutoff_date = pd.Timestamp("2000-11-17")
    df_event = df_event[df_event["date"] >= cutoff_date]
    # Sort the fights from oldest to newest for dataset construction
    df_event = df_event.sort_values(by="date")
    print(df_event)
    
    # Construct the default blank fighter states for every fighter
    for fighter in df_fighter.itertuples(index=False):
        fighter_states[fighter.id] = initialize_fighter(fighter)

    print(len(fighter_states))

    # Construct the dataset and build up fighter states using fighter_id and stats
    for row in df_event.itertuples(index=False):
        print("")

if __name__ == "__main__":
    main()