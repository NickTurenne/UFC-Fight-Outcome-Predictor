import pandas as pd
import numpy as np

def build_features(red_state, blue_state, fight):
    fight_data_dict = {
        "date" : fight.date,
        "division" : normalize_division(fight.division),
        "r_height" : red_state["height"],
        "r_reach" : red_state["reach"],
        "r_stance" : red_state["stance"],
        "r_age" : get_age(fight.date, red_state["dob"]),
        "r_fights" : red_state["num_fights"],



    }
    return fight_data_dict

def calc_average(total, num_fights):
    if num_fights == 0:
        return 0
    else:
        return total / num_fights
    
def calc_accuracy(landed, total):
    if total == 0:
        return 0
    else:
        return landed / total

def get_age(fight_date, dob):
    if pd.isna(dob):
        return 30 * 365.25 # average UFC fighter age in days for fighters with no age (This is usually only the case for old fighters)
    else:
        return (fight_date - dob).days

def normalize_division(division):
    division = division.lower().strip()
    
    if "strawweight" in division:
        return "womens_strawweight"
    elif "flyweight" in division:
        return "flyweight" if "women" not in division else "womens_flyweight"
    elif "bantamweight" in division:
        return "bantamweight" if "women" not in division else "womens_bantamweight"
    elif "featherweight" in division:
        return "featherweight" if "women" not in division else "womens_featherweight"
    elif "lightweight" in division:
        return "lightweight"
    elif "welterweight" in division:
        return "welterweight"
    elif "middleweight" in division:
        return "middleweight"
    elif "light heavyweight" in division:
        return "light_heavyweight"
    elif "heavyweight" in division:
        return "heavyweight"
    elif "super heavyweight" in division:
        return "super_heavyweight"
    else:
        return "other"
