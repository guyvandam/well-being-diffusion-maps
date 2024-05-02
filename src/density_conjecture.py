from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benedict import benedict
from sacred import Experiment
from sacred.observers import FileStorageObserver


from dataset import Dataset, dataset_ingredient
from utils import bootstrap

out_folderpath = Path().cwd().joinpath("out", "density-conjecture")

# * create an experiment
ex = Experiment("density-conjecture", ingredients=[dataset_ingredient])

# * add a file storage observer and "path management"
fp_observer = FileStorageObserver(out_folderpath)
ex.observers.append(fp_observer)
# join name with the results folder
makepath = lambda name: Path(fp_observer.dir).joinpath(name)  # type: ignore


@ex.config
def cfg():
    n_bins = 5
    diam_percentile = 95  # percentile of the diameter
    n_bootstraps = 1000
    is_diffusion_distance = False
    diffusion_distance_run_id = "main"
    # deprecated
    is_normalized_distance = False


@ex.named_config
def diffusion_dists():
    is_diffusion_distance = True


@ex.capture
def _bin_diam(
    arr: np.ndarray,  # deviating from convention so arr will be the first argument for the bootstrap function call.
    diam_percentile: int,
    is_normalized_distance: bool,
) -> float:
    """
    Compute the diameter of a bin.

    Args:
        arr (np.ndarray): The array of points in the bin.
        diam_percentile (int): percentile of the distances from the center of the bin to the other points.
        is_normalized_distance (bool): Flag indicating whether to normalize distances by point value.

    Returns:
        float: The diameter of the bin.

    """
    # compute the center of the bin - mean of all points - shape (n_features,)
    bin_center = np.mean(arr, axis=0)
    # distance of each point to the center of the bin
    distances = np.linalg.norm(arr - bin_center, axis=1)

    # normalize distances by point value to compute density
    if is_normalized_distance:
        point_norm = np.linalg.norm(x=arr, axis=1)
        distances /= point_norm

    diameter = np.percentile(distances, diam_percentile)
    return diameter


@ex.capture
def _plot(results_df: pd.DataFrame, filename: str, title: str | None = None):
    """_plot

    bar plot of the results.

    Args:
        results_df (pd.DataFrame): DataFrame with results - bin_size, bin_value, diameter, ci_low, ci_high.
        filename (str): filename to save the plot.
        title (str | None, optional): plot title. Defaults to None.
    """
    # control figure size
    plt.figure(figsize=(11, 7))

    # Error bars with capsize
    yerr = (results_df["ci_high"] - results_df["diameter"]).to_numpy()
    plt.bar(
        results_df.index,
        results_df["diameter"],
        yerr=yerr,
        capsize=7,
    )

    # Add labels to bars - number of points in each bin and the diameter of the bin.
    # for i in range(len(results_df)):

    #     bin_diam, bin_size = results_df.loc[i, ["diameter", "bin_size"]]  # type: ignore
    #     bin_diam = round(bin_diam, 3)  # ! .round(3) for the line above doesn't work

    #     if bin_diam == 0:
    #         continue

    #     plt.text(
    #         i,
    #         bin_diam // 2,
    #         f"diam: {bin_diam} \n bin-size: {bin_size}",
    #         ha="center",
    #         va="bottom",
    #         color="darkorange",
    #         weight="bold",
    #     )

    if title is not None:
        plt.title(title)
    plt.xlabel("Bin index")
    plt.ylabel("Diameter")

    plt.savefig(makepath(filename))


@ex.automain
def main(
    _log,
    dataset,  # config of the dataset...
    is_diffusion_distance: bool,
    diffusion_distance_run_id: str,
    n_bins: int,
    n_bootstraps: int,
    diam_percentile: int,
):
    """main

    main logic - load dataset / diffusion embedding, bin by the output variable, compute the diameter of each bin and plot the results.

    Args:
        _log (_type_): logger. sacred.
        dataset (_type_): dataset ingredient. sacred.
        diffusion_distance_run_id (str): diffusion embedding run id. sacred.
        n_bins (int): number of bins to split the output variable into.
        n_bootstraps (int): number of bootstraps to compute the confidence intervals.
        diam_percentile (int): percentile of distance values for the diameter.
    """
    if not is_diffusion_distance:
        ds = Dataset()  # actually initialize the dataset # type: ignore
        df = ds.df
        output_variable = dataset["output_variable"]
    else:
        # TODO: remove hardcoding of output folderpath and "runs" foldername
        run_folderpath = out_folderpath.parent.joinpath(
            "runs", diffusion_distance_run_id
        )
        embedding_filepath = run_folderpath.joinpath("diffusion_coordinates.csv")
        config_filepath = run_folderpath.joinpath("config.json")

        # add the resources # TODO: figure out why is not adding to the resources folder...
        ex.add_resource(embedding_filepath)
        ex.add_resource(config_filepath)

        # load the config and embedding files, read the embedding file and remove the output variable
        run_config = benedict.from_json(config_filepath)
        df = pd.read_csv(embedding_filepath)

        # we want to take the output variable from the run, not the dataset as it might be different.
        output_variable = run_config.dataset.output_variable

    results_df = pd.DataFrame(
        columns=["bin_size", "bin_value", "diameter", "ci_low", "ci_high"],
        index=range(n_bins),
    )

    # bin the output variable into 5 bins and create a new column
    df["output_variable_bins"], bins = pd.cut(
        df[output_variable], n_bins, labels=False, retbins=True
    )
    # log bin sizes in the results DataFrame
    results_df["bin_size"] = df["output_variable_bins"].value_counts()
    results_df["bin_range"] = [
        f"{bins[i]}->{bins[i+1]}" for i in range(len(bins) - 1)
    ]  # no need for the first one

    # * calculate the diameter of each bin
    for bin_idx in range(n_bins):
        # get the points in the bin
        bin_df = df[df["output_variable_bins"] == bin_idx]
        bin_df = bin_df.drop(columns=[output_variable, "output_variable_bins"])

        # bootstrap bin diameter
        bin_diam, ci = bootstrap(
            bin_df.values,
            _bin_diam,
            n_resamples=n_bootstraps,
            confidence_level=diam_percentile / 100,
        )

        results_df.loc[bin_idx, ("diameter", "ci_low", "ci_high")] = bin_diam, *ci  # type: ignore

    # plot the results
    title = f"Bin diameters | # points: {len(df)} | # bootstraps : {n_bootstraps} \n binned by: {output_variable} | distance type: {'diffusion' if is_diffusion_distance else 'Euclidean'} | diameter percentile: {diam_percentile}"
    _plot(
        results_df,
        filename=f"diameters-{output_variable}-bins-{'diffusion' if is_diffusion_distance else 'Euclidean'}-dists.png",
    )

    # save results DataFrame to csv
    results_df.to_csv(makepath("results.csv"), index=False)
