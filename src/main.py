from pathlib import Path

import benedict
import numpy as np
import pandas as pd
from benedict import benedict
from sacred import Experiment
from sacred.observers import FileStorageObserver

from dataset import (
    WELFARE_CATEGORY_LIST,
    Dataset,
    dataset_ingredient,
)
from dataset_variables import welfare_regime_dict
from density_conjecture import density_conjecture_ingredient, run_density_conjecture
from diffusion_maps import diffusion_maps_ingredient, get_diffusion_coordinates
from utils import plot_3d_scatter, plot_3d_scatter_matplotlib

# * output folderpath
out_folderpath = Path().cwd().joinpath("out")

# * FILENAMES
DIFFUSION_COORDINATES_FILENAME = "diffusion_coordinates.csv"

# * create an experiment
ex = Experiment(
    "well-being-diffusion-maps",
    ingredients=[
        dataset_ingredient,
        diffusion_maps_ingredient,
        density_conjecture_ingredient,
    ],
)

# * add a file storage observer and "path management"
fp_observer = FileStorageObserver(out_folderpath)
ex.observers.append(fp_observer)


# join name with the results folder of the current run
def makepath(name: str) -> Path:
    return Path(fp_observer.dir).joinpath(name)


def makedir(*args) -> Path:
    folderpath = Path(fp_observer.dir).joinpath(*args)  # type: ignore
    folderpath.mkdir(parents=True, exist_ok=True)
    return folderpath


# main config
@ex.config
def main():
    d = 3  # embedding dimensions # noqa: F841


# @ex.config_hook
# def hook(config, command_name, logger):
#     if command_name == "run_welfare_regimes":
#         # config["dataset"]["input_variable_list"] = (
#         #     MAIN_INPUT_VARIABLES + WELFARE_CATEGORY_LIST
#         # )
#         # config.set
#         config["dataset"]["input_variable_list"] = {}
#         logger.debug("changed input variable list")
#     return config


# @dataset_ingredient.named_config
# def all_welfare_regimes():
#     dataset = {
#         "input_variable_list": MAIN_INPUT_VARIABLES + WELFARE_CATEGORY_LIST
#     }


# @ex.named_config
# def all_welfare_regimes():
#     dataset_ingredient = all_welfare_regimes_dataset
#     run_all_welfare_regimes = True


@ex.capture
def _is_run_exists(_run, _log) -> Path | None:
    """_is_run_exists

    Args:
        _run (_type_): sacred Run object.
        _log (_type_): logger. sacred.

    Returns:
        Path | None: existing run folderpath if exists, None otherwise.
    """
    # get the current config and remove the seed. can also get _config as an argument.
    current_config = _run.config.copy()
    del current_config["seed"]

    # look for all config.json files in the out folder recursively
    for file in out_folderpath.rglob("config.json"):
        # get the run id from the folder name
        temp_run_id = file.parent.name
        # read the config.json file as a dictionary
        temp_config = benedict(file)
        temp_config.remove("seed")

        # compare the current config with the config in the file, return the folderpath if they are the same
        if temp_config == current_config and temp_run_id != _run._id:
            _log.info(f"run exists in {file.parent}")

            return file.parent

    _log.debug("run does not exist")

    return None


@ex.capture
def _load_embeddings(_log, existing_filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    # read the results from the file
    _log.info(f"loading the results from existing filepath {existing_filepath}")
    diffusion_coordinates = pd.read_csv(
        existing_filepath.joinpath(DIFFUSION_COORDINATES_FILENAME)
    ).values
    # separate the diffusion coordinates from the output variable
    diffusion_coordinates, output_variable_series = (
        diffusion_coordinates[:, :-1],
        diffusion_coordinates[:, -1],
    )

    # use the same random seed - read from config
    existing_config = benedict.from_json(existing_filepath.joinpath("config.json"))
    np.random.seed(existing_config.seed)

    return diffusion_coordinates, output_variable_series


@ex.capture
def _save_embeddings(
    d: int,
    diffusion_coordinates: np.ndarray,
    output_variable_series: np.ndarray,
    output_variable: str,
    filepath: Path,
):
    # add the output variable series to the diffusion coordinates as a new column
    data = np.concatenate(
        [
            diffusion_coordinates,
            output_variable_series.reshape(-1, 1),
        ],
        axis=1,
    )
    columns = [f"x{i + 1}" for i in range(d)] + [output_variable]
    pd.DataFrame(data, columns=columns).to_csv(filepath, index=False)


@ex.capture
def _plot(
    diffusion_maps,  # diffusion maps ingredient
    diffusion_coordinates: np.ndarray,
    output_variable_series: np.ndarray,
    output_variable: str,
    is_show: bool = False,
    parent_folderpath: Path | None = None,
    additional_id: str = "",
):
    parameters_string = f"{diffusion_coordinates.shape[0]} points | t={diffusion_maps['t']} | $\\epsilon$={diffusion_maps['epsilon']} | $\\alpha$={diffusion_maps['alpha']}"
    additional_id = "\n" + additional_id if additional_id != "" else ""
    title = f"Diffusion Embedding on {parameters_string}{additional_id}"

    def plot_filepath(name: str) -> Path:
        return (
            parent_folderpath.joinpath(name)
            if parent_folderpath is not None
            else makepath(name)
        )

    # plot results with matplotlib
    plot_3d_scatter_matplotlib(
        data_3d=diffusion_coordinates,
        filepath=plot_filepath("embedding-matplotlib"),
        title=title,
        color_list=output_variable_series,
        color_bar_label=output_variable.replace("_", " "),
        is_show=is_show,
    )

    # plot the results
    plot_3d_scatter(
        data_3d=diffusion_coordinates,
        # remove latex - does not work with plotly
        title=title.replace("\\", "").replace("$", ""),
        filepath=plot_filepath("embedding.html"),
        color_list=output_variable_series,
        # color_bar_label=output_variable,
        is_show=is_show,
    )


@ex.capture
def run_diffusion_maps(
    _log,
    dataset,
    diffusion_maps,
    d: int,
    is_force_process: bool = False,
    n_points: int | None = None,
    is_show: bool = False,
):
    """_run_diffusion_maps

    run diffusion maps embedding. check for existing runs and load the existing embedding if exists.
    checking for existing files is done by comparing the config.json files in the out folder.

    Args:
        _log (_type_): logger. sacred.
        dataset (_type_): sacred dataset ingredient.
        diffusion_maps (_type_): sacred diffusion maps ingredient.
        d (int): diffusion maps embedding dimensions.
        is_force_process (bool, optional): force process flag - recompute even if run exists. Defaults to False.
        n_points (int | None, optional): number of points to use. Defaults to None.
    """
    # misc - get the output variable string from the dataset ingredient.
    output_variable = dataset["output_variable"]

    # check if the run exists
    existing_filepath = _is_run_exists()  # type: ignore
    if existing_filepath is not None and is_force_process is False:
        _log.info(f"experiment exists at {existing_filepath}")
    else:  # process from scratch.
        ds = Dataset(n_points=n_points)  # type: ignore

        # run the diffusion maps
        diffusion_coordinates = get_diffusion_coordinates(
            X=ds.df.drop(columns=[output_variable]).values, d=d
        )  # type: ignore
        output_variable_series = ds.df[output_variable].to_numpy()
        # save the results
        _log.debug("saving the results")
        _save_embeddings(
            d,
            diffusion_coordinates,
            output_variable_series,
            output_variable,
            makepath(DIFFUSION_COORDINATES_FILENAME),
        )

        _plot(
            diffusion_maps,
            diffusion_coordinates,
            output_variable_series,
            output_variable,
            is_show=is_show,
        )

        # run the density conjecture
        run_density_conjecture(
            _log,
            df=ds.df,
            output_variable=output_variable,
            folderpath=makedir("density-conjecture"),
        )  # type: ignore


@ex.command
def run_welfare_regimes(
    _log,
    dataset,
    diffusion_maps,
    d: int,
    is_force_process: bool = False,
    n_points: int | None = None,
):
    """_run_diffusion_maps

    run diffusion maps embedding. check for existing runs and load the existing embedding if exists.
    checking for existing files is done by comparing the config.json files in the out folder.

    Args:
        _log (_type_): logger. sacred.
        dataset (_type_): sacred dataset ingredient. - dictionary.
        diffusion_maps (_type_): sacred diffusion maps ingredient.
        d (int): diffusion maps embedding dimensions. part of sacred config.
        is_force_process (bool, optional): force process flag - recompute even if run exists. Defaults to False.
        n_points (int | None, optional): number of points to use. Defaults to None.
    """

    # misc - get the output variable string from the dataset ingredient.
    output_variable = dataset["output_variable"]

    # check if the run exists
    existing_filepath = _is_run_exists()  # type: ignore
    if existing_filepath is not None and is_force_process is False:
        _log.info(f"experiment exists at {existing_filepath}")
    else:  # process from scratch.
        ds = Dataset(n_points=n_points)  # type: ignore
        # run the diffusion maps for each welfare regime
        for category in WELFARE_CATEGORY_LIST:
            # drop the other categories

            other_categories = [c for c in WELFARE_CATEGORY_LIST if c != category]

            category_df = ds.df.drop(columns=other_categories)
            # group by the welfare regime
            for regime_index, (_, regime_df) in enumerate(
                category_df.groupby(category)
            ):
                # regime index and folderpath
                regime_index += 1  # make 1-based
                regime_folderpath = makedir(f"{category}-regime-{regime_index}")

                # get the output variable series for the current regime.
                output_variable_series = regime_df[
                    output_variable
                ].to_numpy()  # output variable is same for all regimes.

                # run diffusion maps
                diffusion_coordinates = get_diffusion_coordinates(
                    X=regime_df.drop(columns=[output_variable]).values, d=d
                )  # type: ignore

                # save the results
                _log.info(
                    f"saving results for {category} - welfare regime {regime_index}"
                )
                # save the diffusion coordinates
                _save_embeddings(
                    d,
                    diffusion_coordinates,
                    output_variable_series,
                    output_variable,
                    regime_folderpath.joinpath(DIFFUSION_COORDINATES_FILENAME),
                )
                # plot the results
                _plot(
                    diffusion_maps,
                    diffusion_coordinates,
                    output_variable_series,
                    output_variable,
                    is_show=False,
                    parent_folderpath=regime_folderpath,
                    additional_id=f" | {category} | welfare regime - {welfare_regime_dict[regime_index]}",
                )

                # run the density conjecture
                run_density_conjecture(
                    _log,
                    df=regime_df,
                    output_variable=output_variable,
                    folderpath=regime_folderpath,
                    additional_title=f"{category} | welfare regime - {welfare_regime_dict[regime_index]}",
                )  # type: ignore


@ex.automain
def run():
    # run separated by different welfare regimes - e.g. python main.py dataset.welfare_regimes_type_1_wb2
    run_welfare_regimes(
        is_force_process=True,
    )  # type: ignore

    # run without separation by welfare regimes.
    # run_diffusion_maps(is_force_process=True, is_show=True)  # type: ignore
