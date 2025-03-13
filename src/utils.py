from pathlib import Path
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import sem, t
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

### * helper functions

_plotly_tickformat = dict(tickformat=".2e")
TICK_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 16


def set_tick_and_axis_label_fontsizes(
    ax,
    tick_font_size: int = TICK_FONT_SIZE,
    axis_label_font_size: int = AXIS_LABEL_FONT_SIZE,
):
    """
    Set the font sizes for tick labels and axis labels.

    Args:
        ax: The axis object (e.g., matplotlib Axes).
        tick_font_size (int): Font size for tick labels.
        axis_label_font_size (int): Font size for axis labels.
    """
    ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
    ax.xaxis.label.set_size(axis_label_font_size)
    ax.yaxis.label.set_size(axis_label_font_size)
    ax.zaxis.label.set_size(axis_label_font_size)


"""remove all zero rows from a DataFrame"""
remove_all_zero_rows = lambda df: df.loc[(df != 0).any(axis=1)]


def rescale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rescales the values in the given DataFrame using Min-Max scaling.

    Parameters:
        df (pd.DataFrame): The DataFrame to be rescaled.

    Returns:
        pd.DataFrame: The rescaled DataFrame with values between 0 and 1.
    """
    min_max_scaler = MinMaxScaler()
    return pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)


def confidence_interval(
    sample_statistic_list: List[float], confidence_level: float
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate the confidence interval for a given sample statistic list.

    Args:
        sample_statistic_list (List[float]): A list of sample statistics.
        confidence_level (float): The desired confidence level (between 0 and 1).

    Returns:
        Tuple[float, Tuple[float, float]]: A tuple containing the mean of the sample statistic list
        and a tuple representing the confidence interval (lower bound, upper bound).
    """
    n = len(sample_statistic_list)
    # mean and standard error
    m, se = np.mean(sample_statistic_list), sem(
        sample_statistic_list, nan_policy="omit", ddof=n - 1
    )
    # h = se * t_{n-1, 1 - alpha/2}
    h = se * t.ppf((1 + confidence_level) / 2.0, n - 1)

    return float(m), (m - h, m + h)


def bootstrap(
    data: np.ndarray, statistic: Callable, n_resamples: int, confidence_level: float
) -> Tuple[float, Tuple[float, float]]:
    """
    Perform bootstrap resampling on the given data and calculate the confidence interval.

    Args:
        data (np.ndarray): The input data for resampling. can be multidimensional.
        statistic (Callable): The function to calculate the statistic of interest on each resampled data.
        n_resamples (int): The number of resamples to perform. # of bootstraps.
        confidence_level (float): The desired confidence level for the confidence interval.

    Returns:
        Tuple[float, Tuple[float, float]]: The estimated statistic and the confidence interval.

    """
    # sample data
    sample_statistics = []
    for _ in range(n_resamples):
        sample = data[np.random.choice(data.shape[0], len(data), replace=True), :]
        sample_statistics.append(statistic(sample))

    return confidence_interval(sample_statistics, confidence_level)


def plot_3d_scatter_matplotlib(
    data_3d: np.ndarray,
    filepath: Path | None,
    title: str,
    color_list: np.ndarray | None = None,
    color_bar_label: str | None = None,
    is_show=False,
    # axis_labels: dict[str, str] = {"x": "x1", "y": "x2", "z": "x3"},
):
    """plot_3d_scatter_matplotlib

    Args:
        data_3d (np.ndarray): data to plot
        filepath (Path): fig filepath to save
        title (str): figure title
        is_show (bool, optional): whether display it while running. Defaults to False.
    """
    # 3d scatter plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=color_list, s=50)  # type: ignore
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    set_tick_and_axis_label_fontsizes(plt.gca())
    plt.colorbar(scatter, label=color_bar_label)  # Add color bar

    fig.suptitle(title)

    # show, save and close
    if is_show:
        plt.show()
    if filepath is not None:
        plt.savefig(filepath.with_suffix(".png"))
    plt.tight_layout()
    plt.close()


def plot_3d_scatter(
    data_3d: np.ndarray,
    title: str,
    filepath: Union[Path, None] = None,
    color_list: Union[np.ndarray, None] = None,
    is_show=False,
    axis_labels: dict[str, str] = {"x": "x1", "y": "x2", "z": "x3"},
):
    """
    Create a 3D scatter plot using Plotly.

    Args:
        data_3d (np.ndarray): The 3D data points to be plotted.
        title (str): The title of the plot.
        filepath (Union[Path, None], optional): The file path to save the plot as an HTML file. Defaults to None.
        color_list (Union[np.ndarray, None], optional): The color list for the data points. Defaults to None.
        is_show (bool, optional): Whether to display the plot. Defaults to False.
    """

    # 3D scatter plot using Plotly
    fig = px.scatter_3d(
        data_3d,
        x=data_3d[:, 0],
        y=data_3d[:, 1],
        z=data_3d[:, 2],
        color=color_list,
        title=title,
        labels=axis_labels,
    )

    # show, save and close
    if is_show:
        fig.show()

    if filepath is not None:
        fig.write_html(str(filepath.with_suffix(".html")))
        # fig.write_image(str(filepath.with_suffix(".png")))


def bar_plot(y_array: np.ndarray, title: str, filepath: Path | None = None, **kwargs):
    """bar_plot bar plot

    Args:
        y_array (np.ndarray): data
        title (str): plot title
        filepath (Path, optional): filepath to save fig. Defaults to None.
    """
    plt.bar(range(1, len(y_array) + 1), y_array, color=kwargs.get("color", None))
    # plot values on top of the bars
    for i, v in enumerate(y_array):
        plt.text(i + 1, v, str(round(v, 2)), ha="center", va="bottom")

    # x, y labels and title
    plt.xlabel(kwargs.get("xlabel", None), fontsize=TICK_FONT_SIZE)
    plt.ylabel(kwargs.get("ylabel", None), fontsize=TICK_FONT_SIZE)
    set_tick_and_axis_label_fontsizes(plt.gca())
    plt.title(title, fontsize=TICK_FONT_SIZE)

    # save fig and clear all.
    plt.show() if filepath is None else plt.savefig(filepath)

    plt.cla()
    plt.clf()
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    # Generate Swiss roll data
    n_samples = 1000
    sr_points, sr_color = datasets.make_swiss_roll(n_samples, random_state=0)

    # Plot 3D scatter using matplotlib
    title = "Swiss Roll 3D Scatter Plot"
    # plot_3d_scatter(
    #     sr_points, filepath=None, title=title, color_list=sr_color, is_show=True
    # )

    plot_3d_scatter_matplotlib(
        sr_points, filepath=None, title=title, color_list=sr_color, is_show=True
    )
