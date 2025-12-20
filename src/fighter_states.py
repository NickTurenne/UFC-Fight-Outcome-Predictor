import pandas as pd
import numpy as np

DEFAULT_HEIGHT_REACH = 170.0
DEFAULT_STANCE = "other"

def initialize_fighter(fighter):
    stats_dict = {
        # Static
        "height" : check_height(fighter.height),
        "height_missing" : pd.isna(fighter.height),
        "reach" : check_reach(fighter.reach, fighter.height),
        "reach_missing" : pd.isna(fighter.reach),
        "stance" : check_stance(fighter.stance),
        "dob" : fighter.dob,
        "dob_missing" : pd.isna(fighter.dob),

        # Dynamic (all start at 0)
        # Fight history
        "num_fights" : 0,
        "wins" : 0,
        "losses" : 0,
        "win_streak" : 0,
        "loss_streak" : 0,
        "date_of_last_fight" : None,

        # Offencsive stats
        "knockdowns" : 0,
        "sig_str_landed" : 0,
        "sig_str_attempt" : 0,
        "total_str_landed" : 0,
        "total_str_attempt" : 0,
        "td_landed" : 0,
        "td_attempt" : 0,
        "sub_attempt" : 0,
        "ctrl_time" : 0,
        "head_landed" : 0,
        "head_attempt" : 0,
        "body_landed" : 0,
        "body_attempt" : 0,
        "leg_landed" : 0,
        "leg_attempt" : 0,
        "distance_landed" : 0,
        "distance_attempt" : 0,
        "clinch_landed" : 0,
        "clinch_attempt" : 0,
        "ground_landed" : 0,
        "ground_attempt" : 0,

        # Defensice stats
        "knockdowns_against" : 0,
        "sig_str_landed_against" : 0,
        "sig_str_attempt_against" : 0,
        "total_str_landed_against" : 0,
        "total_str_attempt_against" : 0,
        "td_landed_against" : 0,
        "td_attempt_against" : 0,
        "sub_attempt_against" : 0,
        "ctrl_time_against" : 0,
        "head_landed_against" : 0,
        "head_attempt_against" : 0,
        "body_landed_against" : 0,
        "body_attempt_against" : 0,
        "leg_landed_against" : 0,
        "leg_attempt_against" : 0,
        "distance_landed_against" : 0,
        "distance_attempt_against" : 0,
        "clinch_landed_against" : 0,
        "clinch_attempt_against" : 0,
        "ground_landed_against" : 0,
        "ground_attempt_against" : 0,
    }
    return stats_dict

def check_height(height):
    if pd.notna(height):
        return height
    else:
        return DEFAULT_HEIGHT_REACH

def check_reach(reach, height):
    if pd.notna(reach):
        return reach
    elif pd.notna(height):
        return height
    else:
        return DEFAULT_HEIGHT_REACH
    
def check_stance(stance):
    if pd.notna(stance):
        return stance
    else:
        return DEFAULT_STANCE

def update_fighter_states(red_state, blue_state, fight):
    update_red_state(state=red_state, fight=fight)
    update_blue_state(state=blue_state, fight=fight)
    return

def update_red_state(state, fight):
    # Update fight history
    state["num_fights"] += 1
    if fight.winner_id == fight.r_id:
        state["wins"] += 1
        state["win_streak"] += 1
        if state["loss_streak"] > 0:
            state["loss_streak"] = 0
    else:
        state["losses"] += 1
        state["loss_streak"] += 1
        if state["win_streak"] > 0:
            state["win_streak"] = 0
    state["date_of_last_fight"] = fight.date

    # Offensive stats
    state["knockdowns"] += fight.r_kd
    state["sig_str_landed"] += fight.r_sig_str_landed
    state["sig_str_attempt"] += fight.r_sig_str_atmpted
    state["total_str_landed"] += fight.r_total_str_landed
    state["total_str_attempt"] += fight.r_total_str_atmpted
    state["td_landed"] += fight.r_td_landed
    state["td_attempt"] += fight.r_td_atmpted
    return

def update_blue_state(state, fight):
    return