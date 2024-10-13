from pathlib import Path
from typing import List

import pandas as pd
from sacred import Ingredient

from dataset_variables import variable_dict
from utils import remove_all_zero_rows, rescale

dataset_ingredient = Ingredient("dataset")

_SAV_FILEPATH = Path().cwd().joinpath("data", "eqls_integrated_trend_2003-2016.sav")
MAIN_INPUT_VARIABLES: list[str] = [
    "weight",
    "hrs_worked_main_job",
    "is_married",
    "general_health",
    "y16_num_children",
    "y16_education",
    "y16_income",
    "age_square",
    "quality_health_services",
    "quality_education_services",
    "quality_public_transport",
    "quality_childcare_services",
    "quality_longterm_care_services",
    "quality_social_municipal_housing",
    "quality_state_pension",
]
WELFARE_CATEGORY_LIST: list[str] = [
    "welfare_type_category_1",
    "welfare_type_category_2",
]


# main configuration (does not have the named_config decorator) - all subsequent configurations are named configurations
# and are defined as the change to this (main) configuration.
@dataset_ingredient.config
def forward_main():
    input_variable_list = MAIN_INPUT_VARIABLES + [WELFARE_CATEGORY_LIST[0]]
    output_variable = "well_being_2"
    num_rebalancing_rows = 30  # number of highest values rows to remove from the dataset after reweighing, around 20-50 rows are outliers. relevant for reweighing only.


@dataset_ingredient.named_config
def forward_main_welfare_type_2():
    input_variable_list = MAIN_INPUT_VARIABLES + [WELFARE_CATEGORY_LIST[1]]


@dataset_ingredient.named_config
def all_welfare_regimes():
    input_variable_list = (
        MAIN_INPUT_VARIABLES + WELFARE_CATEGORY_LIST
    )  # include both categories


@dataset_ingredient.named_config
def mean_q_services_forward():
    input_variable_list = [
        "weight",
        "hrs_worked_main_job",
        "is_married",
        "general_health",
        "y16_num_children",
        "y16_education",
        "y16_income",
        "age_square",
        "mean_q58_a_f",  # replace with mean of Q58a to Q58f - quality of services
    ]


@dataset_ingredient.named_config
def q_services_only_forward():
    input_variable_list = [
        "weight",
        "quality_health_services",
        "quality_education_services",
        "quality_public_transport",
        "quality_childcare_services",
        "quality_longterm_care_services",
        "quality_social_municipal_housing",
        "quality_state_pension",
    ]


@dataset_ingredient.named_config
def reverse():
    # input_variables stay the same, swapping the well being with the mean quality of services.
    input_variable_list = [
        "weight",
        "hrs_worked_main_job",
        "is_married",
        "general_health",
        "y16_num_children",
        "y16_education",
        "y16_income",
        "age_square",
        # swappable
        "well_being_2",
    ]
    output_variable = "mean_q58_a_f"


class Dataset:

    @dataset_ingredient.capture
    def __init__(self, _log, n_points: int | None = None) -> None:
        """__init__

        Args:
            _log (_type_): logger. sacred.
            n_points (int | None, optional): number of data points to use. Defaults to None.
        """
        _log.debug("loading data")

        # load the data from .sav file
        self._df = pd.DataFrame()
        self._load_sav_file()  # type: ignore
        self._process(n_points=n_points)  # type: ignore

        _log.info("dataset initialized successfully")

    @dataset_ingredient.capture
    def _load_sav_file(
        self,
        _run,
        _log,
        input_variable_list: List[str],
        output_variable: str,
    ):
        """_load_sav_file

        load .sav file, keep only input and output variables, remove missing values and all zero rows.

        Args:
            _run (_type_): scared run.
            _log (_type_): logger. sacred.
            input_variable_list (List[str]): parameters to use as input. sacred.
            output_variable (str): output variable [well-being]. sacred.
        """
        _log.debug("reading .sav file")

        _ = _run.open_resource(_SAV_FILEPATH)

        # load sav file
        self._df = pd.read_spss(_SAV_FILEPATH, usecols=variable_dict.keys(), convert_categoricals=False)  # type: ignore
        _log.info(
            f"loaded sav file with {len(self._df)} rows and {len(self._df.columns)} columns"
        )
        # rename columns
        self._df.rename(columns=variable_dict, inplace=True)

        # retain only the input and output variables - not all the variable_dict.
        self._df = self._df[input_variable_list + [output_variable]]
        _log.info(
            f"retained only input and output variables. {len(self._df.columns)} columns remaining"
        )

        # drop rows with missing values
        self._df.dropna(how="any", inplace=True, axis="index")
        _log.info(f"removed rows with missing values. {len(self._df)} rows remaining")

        # remove rows where all columns are zero by retaining all rows which have at least one non-zero value
        self._df = remove_all_zero_rows(self._df)
        _log.info(f"removed all zero rows. {len(self._df)} rows remaining")

    @dataset_ingredient.capture
    def _process(self, _log, n_points: int | None):
        """_process

        process

        Args:
            _log (_type_): logger. sacred.
            n_points (int | None): number of data-points to use.
        """

        def _rescale_wrapper():
            self._df = rescale(self._df)
            _log.debug("rescaled data")

        # rescale first
        _rescale_wrapper()

        # reweigh and drop the weight column. Do not reweigh the welfare category columns.
        weight_series = self._df["weight"]
        non_welfare_columns = [
            col for col in self._df.columns if col not in WELFARE_CATEGORY_LIST
        ]
        self._df[non_welfare_columns] = self._df[non_welfare_columns].multiply(
            weight_series, axis="index"
        )
        self._df.drop(columns=["weight"], inplace=True)
        _log.debug("reweighed data")

        # rescale again
        _rescale_wrapper()

        # rebalance
        self._rebalance()  # type: ignore

        # final rescale
        _rescale_wrapper()

        # remove all zero rows
        self._df = remove_all_zero_rows(self._df)

        # retain only n_points if needed
        if n_points is not None:
            self._df = self._df.sample(n=n_points)

        _log.info(f"final row count: {len(self._df)}")

    @dataset_ingredient.capture
    def _rebalance(self, _log, num_rebalancing_rows: int | None):
        """_rebalance

        rebalance the left skewed dataset by removing the num_rebalancing_rows highest values.
        occurs after reweighing.
        Do not rebalance the welfare category columns.

        Args:
            _log (_type_): _description_
            num_rebalancing_rows (int | None): _description_
        """
        if num_rebalancing_rows is None:
            return

        for col in self._df.columns:
            if col in WELFARE_CATEGORY_LIST:
                continue

            # sort from lowest to highest.
            self._df = self._df.sort_values(by=col, ascending=True)
            # remove the highest values
            self._df = self._df.iloc[:-num_rebalancing_rows]

    @property  # this makes price a protected class - accessing House.price is a wrapped, modified of House._price
    def df(self):
        return self._df
