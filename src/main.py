import shutil
from pathlib import Path
from pprint import pprint
from typing import Callable

import benedict
import numpy as np
import pandas as pd
from benedict import benedict
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver

from dataset import Dataset, dataset_ingredient
from diffusion_maps import diffusion_maps_ingredient, get_diffusion_coordinates
from utils import plot_3d_scatter, plot_3d_scatter_matplotlib

# * output folderpath
out_folderpath = Path().cwd().joinpath("out", "diff-maps-embeddings")

# * create an experiment
ex = Experiment(
    "well-being-diffusion-maps",
    ingredients=[dataset_ingredient, diffusion_maps_ingredient],
)

# * add a file storage observer and "path management"
fp_observer = FileStorageObserver(out_folderpath)
ex.observers.append(fp_observer)
# join name with the results folder of the current run
makepath = lambda name: Path(fp_observer.dir).joinpath(name)  # type: ignore


# main config
@ex.config
def main():
    d = 3  # embedding dimensions


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
def _run_diffusion_maps(
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
        dataset (_type_): sacred dataset ingredient.
        d (int): diffusion maps embedding dimensions.
        is_force_process (bool, optional): force process flag - recompute even if run exists. Defaults to False.
    """
    # misc
    output_variable = dataset["output_variable"]
    diffusion_coordinates_filename = "diffusion_coordinates.csv"

    # check if the run exists
    existing_filepath = _is_run_exists()  # type: ignore
    if existing_filepath is not None and is_force_process is False:
        # read the results from the file
        _log.info(f"loading the results from existing filepath {existing_filepath}")
        diffusion_coordinates = pd.read_csv(
            existing_filepath.joinpath(diffusion_coordinates_filename)
        ).values
        # separate the diffusion coordinates from the output variable
        diffusion_coordinates, output_variable_series = (
            diffusion_coordinates[:, :-1],
            diffusion_coordinates[:, -1],
        )

        # use the same random seed - read from config
        existing_config = benedict.from_json(existing_filepath.joinpath("config.json"))
        np.random.seed(existing_config.seed)

    else:

        ds = Dataset(n_points=n_points)  # type: ignore

        # run the diffusion maps
        diffusion_coordinates = get_diffusion_coordinates(X=ds.df.drop(columns=[output_variable]).values, d=d)  # type: ignore
        output_variable_series = ds.df[output_variable].to_numpy()
        # save the results
        _log.debug("saving the results")
        pd.DataFrame(
            np.concatenate(
                [
                    diffusion_coordinates,
                    output_variable_series.reshape(-1, 1),
                ],
                axis=1,
            ),
            columns=[f"x{i+1}" for i in range(d)] + [output_variable],
        ).to_csv(makepath(diffusion_coordinates_filename), index=False)

    parameters_string = f"{diffusion_coordinates.shape[0]} points | t={diffusion_maps['t']} | $\\epsilon$={diffusion_maps['epsilon']} | $\\alpha$={diffusion_maps['alpha']}"
    title = f"Diffusion Embedding on {parameters_string}"

    # plot results with matplotlib
    plot_3d_scatter_matplotlib(
        data_3d=diffusion_coordinates,
        filepath=makepath(f"embedding-matplotlib"),
        title=title,
        color_list=output_variable_series,
        color_bar_label=output_variable.replace("_", " "),
        is_show=True,
    )

    # plot the results
    plot_3d_scatter(
        data_3d=diffusion_coordinates,
        # remove latex - does not work with plotly
        title=title.replace("\\", "").replace("$", ""),
        filepath=makepath("embedding.html"),
        color_list=output_variable_series,
        # color_bar_label=output_variable,
        is_show=True,
    )


@ex.automain
def run():
    _run_diffusion_maps(is_force_process=True)  # type: ignore
