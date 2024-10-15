from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sacred import Ingredient

from dataset import dataset_ingredient
from utils import bootstrap, tick_font_size


# * create an experiment
density_conjecture_ingredient = Ingredient(
    "density-conjecture2", ingredients=[dataset_ingredient]
)


@density_conjecture_ingredient.config
def cfg():
    n_bins = 5
    diam_percentile = 95  # percentile of the diameter
    n_bootstraps = 1000


@density_conjecture_ingredient.capture
def _bin_diam(
    arr: np.ndarray,  # deviating from convention so arr will be the first argument for the bootstrap function call.
    diam_percentile: int,
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

    diameter = np.percentile(distances, diam_percentile)
    return diameter


@density_conjecture_ingredient.capture
def _plot(results_df: pd.DataFrame, filepath: Path, title: str | None = None):
    """_plot

    bar plot of the results.

    Args:
        results_df (pd.DataFrame): DataFrame with results - bin_size, diameter, ci_low, ci_high, bin_range.
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
    plt.xlabel("Bin index", fontsize=tick_font_size)
    plt.ylabel("Diameter", fontsize=tick_font_size)
    plt.tight_layout()
    plt.savefig(filepath)


@density_conjecture_ingredient.capture
def run_density_conjecture(
    _log,
    n_bins: int,  # sacred config setting
    n_bootstraps: int,  # sacred config setting
    diam_percentile: int,  # sacred config setting
    df: pd.DataFrame,  # dataset df
    output_variable: str,  # output variable of the dataset
    folderpath: Path,  # experiment folderpath
    additional_title: str = "",
):
    """run_density_conjecture

    main logic - load dataset / diffusion embedding, bin by the output variable, compute the diameter of each bin and plot the results.

    Args:
        _log (_type_): logger. sacred.
        df (pd.DataFrame): dataset df.
        output_variable (str): output variable to bin by.
        n_bins (int): number of bins to split the output variable into.
        n_bootstraps (int): number of bootstraps to compute the confidence intervals.
        diam_percentile (int): percentile of distance values for the diameter.
        folderpath (Path): folderpath to save the results.
        additional_title (str): additional title for the plot.
    """
    results_df = pd.DataFrame(
        columns=["bin_size", "diameter", "ci_low", "ci_high"],
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
    title = f"Bin diameters | # points: {len(df)} | # bootstraps : {n_bootstraps} | binned by: {output_variable} | diameter percentile: {diam_percentile}"
    title += f"\n{additional_title}" if additional_title else ""  # add additional info

    _plot(
        results_df,
        title=title,
        filepath=folderpath.joinpath(f"{output_variable}-bin-diameters.png"),
    )

    # save results DataFrame to csv
    results_df.to_csv(
        folderpath.joinpath(f"{output_variable}-bin-diameters.csv"), index=False
    )
