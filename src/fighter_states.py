import pandas as pd
import numpy as np

def initialize_fighter(fighter):
    stats_dict = {
        # Static
        "height" : fighter.height,
        "reach" : fighter.reach,
        "stance" : fighter.stance,
        "dob" : fighter.dob,

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

def update_fighter_states(red_state, blue_state, fight):
    return