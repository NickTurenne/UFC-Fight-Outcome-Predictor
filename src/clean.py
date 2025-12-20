import sys
import pandas as pd
import numpy as np
from pathlib import Path

def clean_fight_event_data() -> pd.DataFrame:
    # Load data
    project_root = Path().resolve().parents[0]
    sys.path.append(str(project_root))

    df_fight = pd.read_csv(Path(project_root) / "UFC-Fight-Outcome-Predictor"  / "data" / "raw" / "fight_details.csv")
    df_event = pd.read_csv(Path(project_root) / "UFC-Fight-Outcome-Predictor" / "data" / "raw" / "event_details.csv")

    # Format DateTime
    df_event["date"] = pd.to_datetime(df_event["date"], format="%B %d, %Y")
    # Drop fights with no winner 
    df_event = df_event.dropna(subset=["winner", "winner_id"])
    # Drop fights before Unified Rules of MMA (UFC 28, November 17, 2000)
    cutoff_date = pd.Timestamp("2000-11-17")
    df_event = df_event[df_event["date"] >= cutoff_date]
    # Sort the fights from oldest to newest for dataset construction
    df_event = df_event.sort_values(by="date")

    df_event_fights = pd.merge(df_event, df_fight, on="fight_id")
    return df_event_fights


def clean_fighter_data() -> pd.DataFrame:
    # Load data
    project_root = Path().resolve().parents[0]
    sys.path.append(str(project_root))

    df_fighter = pd.read_csv(Path(project_root) / "UFC-Fight-Outcome-Predictor" / "data" / "raw" / "fighter_details.csv")

    # Format DateTime
    df_fighter["dob"] = pd.to_datetime(df_fighter["dob"], format="%b %d, %Y")
    return df_fighter