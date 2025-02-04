variable_dict = {
    "WCalib": "weight",  # WCalib appears with no __ suffix in the .sav file.
    "Y16_Q4__": "life_satisfaction",
    "Y16_Q5__": "happiness",
    "Y16_Q14__": "hrs_worked_main_job",
    "married": "is_married",
    "Y16_Q48__": "general_health",
    "Y16_Q58a__": "quality_health_services",
    "Y16_Q58b__": "quality_education_services",
    "Y16_Q58c__": "quality_public_transport",
    "Y16_Q58d__": "quality_childcare_services",
    "Y16_Q58e__": "quality_longterm_care_services",
    "Y16_Q58f__": "quality_social_municipal_housing",
    "Y16_Q58g__": "quality_state_pension",
    "Y16_Children_Total__": "y16_num_children",
    "Y16_Education_3categories__": "y16_education",
    "Y16_Income_Equivalised__": "y16_income",
    "index_wellbeing2__": "well_being_2",
    "Y16_Q51a_reversed": "cheerful",
    "Y16_Q51b_reversed": "calm_relaxed",
    "Y16_Q51c_reversed": "active_vigorous",
    "Y16_Q51d_reversed": "woke_up_fresh",
    "Y16_Q51e_reversed": "interest_filled_life",
    "Well_being13__": "well_being_13",
    "age_square": "age_square",
    "Welfare_Type_Cat1": "welfare_type_category_1",
    "Welfare_Type_Cat2": "welfare_type_category_2",
    "Index_quality_of_services": "mean_q58_a_f",  # mean of Q58a to Q58f
}

# welfare regime enum
# 1. Social Democratic
# 2. Liberal
# 3. Continental
# 4. Mediterranean
# 5. Eastern European
# 6. Other
welfare_regime_dict = {
    1: "Social Democratic",
    2: "Liberal",
    3: "Continental",
    4: "Mediterranean",
    5: "Eastern European",
    6: "Other",
}
