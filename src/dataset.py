from pathlib import Path
from typing import List

import pandas as pd
from sacred import Ingredient

from dataset_variables import variable_dict
from utils import remove_all_zero_rows, rescale

dataset_ingredient = Ingredient("dataset")

DATA_FILEPATH = Path().cwd().joinpath("data", "dataset.csv")

MAIN_INPUT_VARIABLES = [
    "weight",
    "hrs_worked_main_job",
    "is_married",
    "general_health",
    "quality_health_services",
    "quality_education_services",
    "quality_public_transport",
    "quality_childcare_services",
    "quality_longterm_care_services",
    "quality_social_municipal_housing",
    "quality_state_pension",
    "mean_q58_a_f",
    "y16_num_children",
    "y16_education",
    "y16_income",
    "age",
    "age_squared",
    "gender",
]

WELFARE_CATEGORY_LIST: list[str] = [
    "welfare_type_category_1",
    "welfare_type_category_2",
]


# main configuration (does not have the named_config decorator) - all subsequent configurations are named configurations
# and are defined as the change to this (main) configuration.
@dataset_ingredient.config
def all_wb2():
    input_variable_list = MAIN_INPUT_VARIABLES  # noqa: F841
    output_variable = "wellbeing_2"  # noqa: F841
    num_rebalancing_rows = 30  # number of highest values rows to remove from the dataset after reweighing, around 20-50 rows are outliers. relevant for reweighing only.  # noqa: F841


@dataset_ingredient.named_config
def all_wb13():
    input_variable_list = MAIN_INPUT_VARIABLES  # noqa: F841
    output_variable = "wellbeing_13"  # noqa: F841


@dataset_ingredient.named_config
def welfare_regimes_wb2():
    input_variable_list = MAIN_INPUT_VARIABLES + WELFARE_CATEGORY_LIST  # noqa: F841
    output_variable = "wellbeing_2"  # noqa: F841


@dataset_ingredient.named_config
def welfare_regimes_wb13():
    input_variable_list = MAIN_INPUT_VARIABLES + WELFARE_CATEGORY_LIST  # noqa: F841
    output_variable = "wellbeing_13"  # noqa: F841


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
        self._load_data_file()  # type: ignore
        self._process(n_points=n_points)  # type: ignore

        _log.info("dataset initialized successfully")

    @dataset_ingredient.capture
    def _load_data_file(
        self,
        _run,
        _log,
        input_variable_list: List[str],
        output_variable: str,
    ):
        """_load_data_file

        load data file - csv or sav, keep only input and output variables, remove missing values and all zero rows.

        Args:
            _run (_type_): scared run.
            _log (_type_): logger. sacred.
            input_variable_list (List[str]): parameters to use as input. sacred.
            output_variable (str): output variable [well-being]. sacred.
        """
        _log.debug("reading datafile file")

        _ = _run.open_resource(DATA_FILEPATH)

        # load sav file
        use_cols = list(variable_dict.keys())

        if DATA_FILEPATH.suffix == ".csv":
            self._df = pd.read_csv(
                DATA_FILEPATH, usecols=use_cols, na_values=[".a", ".b"]
            )
        else:
            self._df = pd.read_spss(
                DATA_FILEPATH,
                usecols=use_cols,
                convert_categoricals=False,
            )

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
            # categorical variables are not reweighed.
            if col in WELFARE_CATEGORY_LIST:
                continue

            # sort from lowest to highest.
            self._df = self._df.sort_values(by=col, ascending=True)
            # remove the highest values
            self._df = self._df.iloc[:-num_rebalancing_rows]

    @property  # this makes price a protected class - accessing House.price is a wrapped, modified of House._price
    def df(self):
        return self._df
