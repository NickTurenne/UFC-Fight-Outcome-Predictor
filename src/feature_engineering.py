import pandas as pd
import numpy as np

AVERAGE_UFC_FIGHTER_AGE = 30

DIFF_FEATURES = [
    # Physical / fighter info
    "height",
    "reach",
    "age",
    "days_since_last_fight",
    "fights",
    "win_streak",
    "loss_streak",

    # Offensive averages
    "sig_str_landed_avg",
    "sig_str_attempt_avg",
    "sig_str_acc",
    "tot_str_landed_avg",
    "tot_str_attempt_avg",
    "tot_str_acc",
    "td_landed_avg",
    "td_attempt_avg",
    "td_acc",
    "sub_attempt_avg",
    "ctrl_time_avg",
    "head_landed_avg",
    "head_attempt_avg",
    "head_acc",
    "body_landed_avg",
    "body_attempt_avg",
    "body_acc",
    "leg_landed_avg",
    "leg_attempt_avg",
    "leg_acc",
    "distance_landed_avg",
    "distance_attempt_avg",
    "distance_acc",
    "clinch_landed_avg",
    "clinch_attempt_avg",
    "clinch_acc",
    "ground_landed_avg",
    "ground_attempt_avg",
    "ground_acc",
    "knockdown_avg",

    # Defensive averages / percentages
    "sig_str_def",
    "tot_str_def",
    "td_def",
    "head_def",
    "body_def",
    "leg_def",
    "distance_def",
    "clinch_def",
    "ground_def",
    "knockdown_against_avg",

    # Advantages
    "sig_str_adv",
    "head_adv",
    "distance_adv",
    "td_adv",

    # Rolling average
    "sig_str_attempt_last_3_avg"
]

ROLLING_STATS = [
    "sig_str_landed_avg",
    "sig_str_attempt_avg",
    "sig_str_acc",
    "td_landed_avg",
    "td_attempt_avg",
    "td_acc",
    "sub_attempt_avg",
    "ctrl_time_avg",
    "sig_str_def",
    "td_def",
    "head_def"
]

def build_features(red_state, blue_state, fight, fighter_last_3, red_id, blue_id):
    fight_data_dict = {
        "date" : fight.date,
        "division" : normalize_division(fight.division),
        # Red fighter info
        "r_height" : red_state["height"],
        "r_height_missing" : int(red_state["height_missing"]),
        "r_reach" : red_state["reach"],
        "r_reach_missing" : int(red_state["reach_missing"]),
        "r_stance" : red_state["stance"],
        "r_age" : get_age(fight.date, red_state["dob"]),
        "r_dob_missing" : int(red_state["dob_missing"]),
        "r_fights" : red_state["num_fights"],
        "r_debut" : int(red_state["num_fights"] == 0),
        "r_win_streak" : red_state["win_streak"],
        "r_loss_streak" : red_state["loss_streak"],
        "r_days_since_last_fight" : get_days_since_last_fight(fight.date, red_state["date_of_last_fight"]),
        # Offensive stats
        "r_knockdown_avg" : safe_division(red_state["knockdowns"], red_state["num_fights"]),
        "r_sig_str_landed_avg" : safe_division(red_state["sig_str_landed"], red_state["num_fights"]),
        "r_sig_str_attempt_avg" : safe_division(red_state["sig_str_attempt"], red_state["num_fights"]),
        "r_sig_str_acc" : safe_division(red_state["sig_str_landed"], red_state["sig_str_attempt"]),
        "r_tot_str_landed_avg" : safe_division(red_state["total_str_landed"], red_state["num_fights"]), 
        "r_tot_str_attempt_avg" : safe_division(red_state["total_str_attempt"], red_state["num_fights"]),
        "r_tot_str_acc" : safe_division(red_state["total_str_landed"], red_state["total_str_attempt"]),
        "r_td_landed_avg" : safe_division(red_state["td_landed"], red_state["num_fights"]),
        "r_td_attempt_avg" : safe_division(red_state["td_attempt"], red_state["num_fights"]),
        "r_td_acc" : safe_division(red_state["td_landed"], red_state["td_attempt"]),
        "r_sub_attempt_avg" : safe_division(red_state["sub_attempt"], red_state["num_fights"]),
        "r_ctrl_time_avg" : safe_division(red_state["ctrl_time"], red_state["num_fights"]),
        "r_head_landed_avg" : safe_division(red_state["head_landed"], red_state["num_fights"]),
        "r_head_attempt_avg" : safe_division(red_state["head_attempt"], red_state["num_fights"]),
        "r_head_acc" : safe_division(red_state["head_landed"], red_state["head_attempt"]),
        "r_body_landed_avg" : safe_division(red_state["body_landed"], red_state["num_fights"]),
        "r_body_attempt_avg" : safe_division(red_state["body_attempt"], red_state["num_fights"]),
        "r_body_acc" : safe_division(red_state["body_landed"], red_state["body_attempt"]),
        "r_leg_landed_avg" : safe_division(red_state["leg_landed"], red_state["num_fights"]),
        "r_leg_attempt_avg" : safe_division(red_state["leg_attempt"], red_state["num_fights"]),
        "r_leg_acc" : safe_division(red_state["leg_landed"], red_state["leg_attempt"]),
        "r_distance_landed_avg" : safe_division(red_state["distance_landed"], red_state["num_fights"]),
        "r_distance_attempt_avg" : safe_division(red_state["distance_attempt"], red_state["num_fights"]),
        "r_distance_acc" : safe_division(red_state["distance_landed"], red_state["distance_attempt"]), 
        "r_clinch_landed_avg" : safe_division(red_state["clinch_landed"], red_state["num_fights"]),
        "r_clinch_attempt_avg" : safe_division(red_state["clinch_attempt"], red_state["num_fights"]),
        "r_clinch_acc" : safe_division(red_state["clinch_landed"], red_state["clinch_attempt"]),
        "r_ground_landed_avg" : safe_division(red_state["ground_landed"], red_state["num_fights"]),
        "r_ground_attempt_avg" : safe_division(red_state["ground_attempt"], red_state["num_fights"]),
        "r_ground_acc" : safe_division(red_state["ground_landed"], red_state["ground_attempt"]),
        # Defensive stats
        "r_knockdown_against_avg" : safe_division(red_state["knockdowns_against"], red_state["num_fights"]),
        "r_sig_str_landed_against_avg" : safe_division(red_state["sig_str_landed_against"], red_state["num_fights"]),
        "r_sig_str_attempt_against_avg" : safe_division(red_state["sig_str_attempt_against"], red_state["num_fights"]),
        "r_sig_str_def" : safe_defence(red_state["sig_str_landed_against"], red_state["sig_str_attempt_against"]),
        "r_tot_str_landed_against_avg" : safe_division(red_state["total_str_landed_against"], red_state["num_fights"]), 
        "r_tot_str_attempt_against_avg" : safe_division(red_state["total_str_attempt_against"], red_state["num_fights"]),
        "r_tot_str_def" : safe_defence(red_state["total_str_landed_against"], red_state["total_str_attempt_against"]),
        "r_td_landed_against_avg" : safe_division(red_state["td_landed_against"], red_state["num_fights"]),
        "r_td_attempt_against_avg" : safe_division(red_state["td_attempt_against"], red_state["num_fights"]),
        "r_td_def" : safe_defence(red_state["td_landed_against"], red_state["td_attempt_against"]),
        "r_sub_attempt_against_avg" : safe_division(red_state["sub_attempt_against"], red_state["num_fights"]),
        "r_ctrl_time_against_avg" : safe_division(red_state["ctrl_time_against"], red_state["num_fights"]),
        "r_head_landed_against_avg" : safe_division(red_state["head_landed_against"], red_state["num_fights"]),
        "r_head_attempt_against_avg" : safe_division(red_state["head_attempt_against"], red_state["num_fights"]),
        "r_head_def" : safe_defence(red_state["head_landed_against"], red_state["head_attempt_against"]),
        "r_body_landed_against_avg" : safe_division(red_state["body_landed_against"], red_state["num_fights"]),
        "r_body_attempt_against_avg" : safe_division(red_state["body_attempt_against"], red_state["num_fights"]),
        "r_body_def" : safe_defence(red_state["body_landed_against"], red_state["body_attempt_against"]),
        "r_leg_landed_against_avg" : safe_division(red_state["leg_landed_against"], red_state["num_fights"]),
        "r_leg_attempt_against_avg" : safe_division(red_state["leg_attempt_against"], red_state["num_fights"]),
        "r_leg_def" : safe_defence(red_state["leg_landed_against"], red_state["leg_attempt_against"]),
        "r_distance_against_landed_avg" : safe_division(red_state["distance_landed_against"], red_state["num_fights"]),
        "r_distance_against_attempt_avg" : safe_division(red_state["distance_attempt_against"], red_state["num_fights"]),
        "r_distance_def" : safe_defence(red_state["distance_landed_against"], red_state["distance_attempt_against"]), 
        "r_clinch_landed_against_avg" : safe_division(red_state["clinch_landed_against"], red_state["num_fights"]),
        "r_clinch_attempt_against_avg" : safe_division(red_state["clinch_attempt_against"], red_state["num_fights"]),
        "r_clinch_def" : safe_defence(red_state["clinch_landed_against"], red_state["clinch_attempt_against"]),
        "r_ground_landed_against_avg" : safe_division(red_state["ground_landed_against"], red_state["num_fights"]),
        "r_ground_attempt_against_avg" : safe_division(red_state["ground_attempt_against"], red_state["num_fights"]),
        "r_ground_def" : safe_defence(red_state["ground_landed_against"], red_state["ground_attempt_against"]),
        # Blue fighter info
        "b_height" : blue_state["height"],
        "b_height_missing" : int(blue_state["height_missing"]),
        "b_reach" : blue_state["reach"],
        "b_reach_missing" : int(blue_state["reach_missing"]),
        "b_stance" : blue_state["stance"],
        "b_age" : get_age(fight.date, blue_state["dob"]),
        "b_dob_missing" : int(blue_state["dob_missing"]),
        "b_fights" : blue_state["num_fights"],
        "b_debut" : int(blue_state["num_fights"] == 0),
        "b_win_streak" : blue_state["win_streak"],
        "b_loss_streak" : blue_state["loss_streak"],
        "b_days_since_last_fight" : get_days_since_last_fight(fight.date, blue_state["date_of_last_fight"]),
        # Offensive stats
        "b_knockdown_avg" : safe_division(blue_state["knockdowns"], blue_state["num_fights"]),
        "b_sig_str_landed_avg" : safe_division(blue_state["sig_str_landed"], blue_state["num_fights"]),
        "b_sig_str_attempt_avg" : safe_division(blue_state["sig_str_attempt"], blue_state["num_fights"]),
        "b_sig_str_acc" : safe_division(blue_state["sig_str_landed"], blue_state["sig_str_attempt"]),
        "b_tot_str_landed_avg" : safe_division(blue_state["total_str_landed"], blue_state["num_fights"]),
        "b_tot_str_attempt_avg" : safe_division(blue_state["total_str_attempt"], blue_state["num_fights"]),
        "b_tot_str_acc" : safe_division(blue_state["total_str_landed"], blue_state["total_str_attempt"]),
        "b_td_landed_avg" : safe_division(blue_state["td_landed"], blue_state["num_fights"]),
        "b_td_attempt_avg" : safe_division(blue_state["td_attempt"], blue_state["num_fights"]),
        "b_td_acc" : safe_division(blue_state["td_landed"], blue_state["td_attempt"]),
        "b_sub_attempt_avg" : safe_division(blue_state["sub_attempt"], blue_state["num_fights"]),
        "b_ctrl_time_avg" : safe_division(blue_state["ctrl_time"], blue_state["num_fights"]),
        "b_head_landed_avg" : safe_division(blue_state["head_landed"], blue_state["num_fights"]),
        "b_head_attempt_avg" : safe_division(blue_state["head_attempt"], blue_state["num_fights"]),
        "b_head_acc" : safe_division(blue_state["head_landed"], blue_state["head_attempt"]),
        "b_body_landed_avg" : safe_division(blue_state["body_landed"], blue_state["num_fights"]),
        "b_body_attempt_avg" : safe_division(blue_state["body_attempt"], blue_state["num_fights"]),
        "b_body_acc" : safe_division(blue_state["body_landed"], blue_state["body_attempt"]),
        "b_leg_landed_avg" : safe_division(blue_state["leg_landed"], blue_state["num_fights"]),
        "b_leg_attempt_avg" : safe_division(blue_state["leg_attempt"], blue_state["num_fights"]),
        "b_leg_acc" : safe_division(blue_state["leg_landed"], blue_state["leg_attempt"]),
        "b_distance_landed_avg" : safe_division(blue_state["distance_landed"], blue_state["num_fights"]),
        "b_distance_attempt_avg" : safe_division(blue_state["distance_attempt"], blue_state["num_fights"]),
        "b_distance_acc" : safe_division(blue_state["distance_landed"], blue_state["distance_attempt"]),
        "b_clinch_landed_avg" : safe_division(blue_state["clinch_landed"], blue_state["num_fights"]),
        "b_clinch_attempt_avg" : safe_division(blue_state["clinch_attempt"], blue_state["num_fights"]),
        "b_clinch_acc" : safe_division(blue_state["clinch_landed"], blue_state["clinch_attempt"]),
        "b_ground_landed_avg" : safe_division(blue_state["ground_landed"], blue_state["num_fights"]),
        "b_ground_attempt_avg" : safe_division(blue_state["ground_attempt"], blue_state["num_fights"]),
        "b_ground_acc" : safe_division(blue_state["ground_landed"], blue_state["ground_attempt"]),
        # Defensive stats
        "b_knockdown_against_avg" : safe_division(blue_state["knockdowns_against"], blue_state["num_fights"]),
        "b_sig_str_landed_against_avg" : safe_division(blue_state["sig_str_landed_against"], blue_state["num_fights"]),
        "b_sig_str_attempt_against_avg" : safe_division(blue_state["sig_str_attempt_against"], blue_state["num_fights"]),
        "b_sig_str_def" : safe_defence(blue_state["sig_str_landed_against"], blue_state["sig_str_attempt_against"]),
        "b_tot_str_landed_against_avg" : safe_division(blue_state["total_str_landed_against"], blue_state["num_fights"]),
        "b_tot_str_attempt_against_avg" : safe_division(blue_state["total_str_attempt_against"], blue_state["num_fights"]),
        "b_tot_str_def" : safe_defence(blue_state["total_str_landed_against"], blue_state["total_str_attempt_against"]),
        "b_td_landed_against_avg" : safe_division(blue_state["td_landed_against"], blue_state["num_fights"]),
        "b_td_attempt_against_avg" : safe_division(blue_state["td_attempt_against"], blue_state["num_fights"]),
        "b_td_def" : safe_defence(blue_state["td_landed_against"], blue_state["td_attempt_against"]),
        "b_sub_attempt_against_avg" : safe_division(blue_state["sub_attempt_against"], blue_state["num_fights"]),
        "b_ctrl_time_against_avg" : safe_division(blue_state["ctrl_time_against"], blue_state["num_fights"]),
        "b_head_landed_against_avg" : safe_division(blue_state["head_landed_against"], blue_state["num_fights"]),
        "b_head_attempt_against_avg" : safe_division(blue_state["head_attempt_against"], blue_state["num_fights"]),
        "b_head_def" : safe_defence(blue_state["head_landed_against"], blue_state["head_attempt_against"]),
        "b_body_landed_against_avg" : safe_division(blue_state["body_landed_against"], blue_state["num_fights"]),
        "b_body_attempt_against_avg" : safe_division(blue_state["body_attempt_against"], blue_state["num_fights"]),
        "b_body_def" : safe_defence(blue_state["body_landed_against"], blue_state["body_attempt_against"]),
        "b_leg_landed_against_avg" : safe_division(blue_state["leg_landed_against"], blue_state["num_fights"]),
        "b_leg_attempt_against_avg" : safe_division(blue_state["leg_attempt_against"], blue_state["num_fights"]),
        "b_leg_def" : safe_defence(blue_state["leg_landed_against"], blue_state["leg_attempt_against"]),
        "b_distance_against_landed_avg" : safe_division(blue_state["distance_landed_against"], blue_state["num_fights"]),
        "b_distance_against_attempt_avg" : safe_division(blue_state["distance_attempt_against"], blue_state["num_fights"]),
        "b_distance_def" : safe_defence(blue_state["distance_landed_against"], blue_state["distance_attempt_against"]),
        "b_clinch_landed_against_avg" : safe_division(blue_state["clinch_landed_against"], blue_state["num_fights"]),
        "b_clinch_attempt_against_avg" : safe_division(blue_state["clinch_attempt_against"], blue_state["num_fights"]),
        "b_clinch_def" : safe_defence(blue_state["clinch_landed_against"], blue_state["clinch_attempt_against"]),
        "b_ground_landed_against_avg" : safe_division(blue_state["ground_landed_against"], blue_state["num_fights"]),
        "b_ground_attempt_against_avg" : safe_division(blue_state["ground_attempt_against"], blue_state["num_fights"]),
        "b_ground_def" : safe_defence(blue_state["ground_landed_against"], blue_state["ground_attempt_against"]),
    }
    add_adv_features(fight_data_dict)
    add_rolling_averages_to_row(fight_data_dict, fighter_last_3, red_id, blue_id)
    add_diff_features(fight_data_dict, DIFF_FEATURES)
    return fight_data_dict

def safe_division(dividend, divisor) -> float:
    if divisor == 0:
        return 0
    else:
        return dividend / divisor

def safe_defence(landed, attempted):
    if attempted == 0:
        return 0.5
    else:
        return 1.0 - (landed / attempted)

def get_days_since_last_fight(fight_date, date_last_fight):
    if (date_last_fight is not None):
        return (fight_date - date_last_fight).days
    else:
        return 0

def get_age(fight_date, dob):
    if pd.isna(dob):
        return AVERAGE_UFC_FIGHTER_AGE * 365.25 # average UFC fighter age in days for fighters with no age (This is usually only the case for old fighters)
    else:
        return (fight_date - dob).days

def add_diff_features(row, diff_features):
    for feat in diff_features:
        row[f"{feat}_diff"] = row[f"r_{feat}"] - row[f"b_{feat}"]
    return row

def add_adv_features(row):
    r_b = ["r", "b"]
    for letter in r_b:
        row[f"{letter}_sig_str_adv"] = row[f"{letter}_sig_str_acc"] - row[f"{letter}_sig_str_def"]
        row[f"{letter}_head_adv"] = row[f"{letter}_head_acc"] - row[f"{letter}_head_def"]
        row[f"{letter}_distance_adv"] = row[f"{letter}_distance_acc"] - row[f"{letter}_distance_def"]
        row[f"{letter}_td_adv"] = row[f"{letter}_td_acc"] - row[f"{letter}_td_def"]
    return row

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

def add_rolling_averages_to_row(row, fighter_last_3, red_id, blue_id):
    for fighter_id, prefix in [(red_id, "r"), (blue_id, "b")]:
        history = fighter_last_3[fighter_id] 

        if len(history) == 0:
            row[f"{prefix}_sig_str_landed_last_3_avg"] = 0.0
            row[f"{prefix}_sig_str_attempt_last_3_avg"] = 0.0
            row[f"{prefix}_sig_str_acc_last_3_avg"] = 0.0
            row[f"{prefix}_sig_str_def_last_3_avg"] = 0.0

            row[f"{prefix}_td_landed_last_3_avg"] = 0.0
            row[f"{prefix}_td_attempt_last_3_avg"] = 0.0
            row[f"{prefix}_td_acc_last_3_avg"] = 0.0
            row[f"{prefix}_td_def_last_3_avg"] = 0.0

            row[f"{prefix}_sub_attempt_last_3_avg"] = 0.0
            row[f"{prefix}_ctrl_time_last_3_avg"] = 0.0
            continue

        sig_landed = []
        sig_attempt = []
        td_landed = []
        td_attempt = []
        sub_attempt = []
        ctrl_time = []
        sig_landed_against = []
        sig_attempt_against = []
        td_landed_against = []
        td_attempt_against = []

        for fight in history:
            # Offensive
            sig_landed.append(fight.get("sig_str_landed", 0))
            sig_attempt.append(fight.get("sig_str_attempt", 0))
            td_landed.append(fight.get("td_landed", 0))
            td_attempt.append(fight.get("td_attempt", 0))
            sub_attempt.append(fight.get("sub_attempt", 0))
            ctrl_time.append(fight.get("ctrl_time", 0))

            # Defensive
            sig_landed_against.append(fight.get("sig_str_landed_against", 0))
            sig_attempt_against.append(fight.get("sig_str_attempt_against", 0))
            td_landed_against.append(fight.get("td_landed_against", 0))
            td_attempt_against.append(fight.get("td_attempt_against", 0))

        sig_landed = np.array(sig_landed, np.float32)
        sig_attempt = np.array(sig_attempt, np.float32)
        td_landed = np.array(td_landed, np.float32)
        td_attempt = np.array(td_attempt, np.float32)
        sub_attempt = np.array(sub_attempt, np.float32)
        ctrl_time = np.array(ctrl_time, np.float32)
        sig_landed_against = np.array(sig_landed_against, np.float32)
        sig_attempt_against = np.array(sig_attempt_against, np.float32)
        td_landed_against = np.array(td_landed_against, np.float32)
        td_attempt_against = np.array(td_attempt_against, np.float32)

        sig_acc = np.divide(sig_landed, sig_attempt, out=np.zeros_like(sig_attempt), where=sig_attempt>0)
        sig_def = np.divide(sig_landed_against, sig_attempt_against, out=np.zeros_like(sig_attempt_against), where=sig_attempt_against>0)
        td_acc = np.divide(td_landed, td_attempt, out=np.zeros_like(td_attempt), where=td_attempt>0)
        td_def = np.divide(td_landed_against, td_attempt_against, out=np.zeros_like(td_attempt_against), where=td_attempt_against>0)

        row[f"{prefix}_sig_str_landed_last_3_avg"] = float(sig_landed.mean())
        row[f"{prefix}_sig_str_attempt_last_3_avg"] = float(sig_attempt.mean())
        row[f"{prefix}_sig_str_acc_last_3_avg"] = float(sig_acc.mean())
        row[f"{prefix}_sig_str_def_last_3_avg"] = float(sig_def.mean())

        row[f"{prefix}_td_landed_last_3_avg"] = float(td_landed.mean())
        row[f"{prefix}_td_attempt_last_3_avg"] = float(td_attempt.mean())
        row[f"{prefix}_td_acc_last_3_avg"] = float(td_acc.mean())
        row[f"{prefix}_td_def_last_3_avg"] = float(td_def.mean())

        row[f"{prefix}_sub_attempt_last_3_avg"] = float(sub_attempt.mean())
        row[f"{prefix}_ctrl_time_last_3_avg"] = float(ctrl_time.mean())

    return row


