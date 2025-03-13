variable_dict = {
    "wcalib": "weight",
    "hours_worked_in_main_job_new": "hrs_worked_main_job",
    "married": "is_married",
    "health_reversed_new": "general_health",
    # quality of services
    "y16_q58a__": "quality_health_services",
    "y16_q58b__": "quality_education_services",
    "y16_q58c__": "quality_public_transport",
    "y16_q58d_new": "quality_childcare_services",
    "y16_q58e_new": "quality_longterm_care_services",
    "y16_q58f_new": "quality_social_municipal_housing",
    "y16_q58g__": "quality_state_pension",
    "index_quality_services_new": "mean_q58_a_f",
    "children_total_new": "y16_num_children",
    "education_new": "y16_education",
    "income_equivalised_new": "y16_income",
    "y16_hh2b": "age",
    "age_square": "age_squared",
    "gender_new": "gender",
    # wellbeing
    "index_wellbeing2_new": "wellbeing_2",
    "wb13_z": "wellbeing_13",
    "welfare_type_cat1": "welfare_type_category_1",
    "welfare_type_cat2": "welfare_type_category_2",
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
