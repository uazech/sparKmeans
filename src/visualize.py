"""Visualisation module
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

from constants import (CLASSES, PETAL_LENGTH_INDEX, PETAL_WIDTH_INDEX,
                       SEPAL_LENGHT_INDEX, SEPAL_WIDTH_INDEX)
from kmeans import closest_center


def visualize(centers_hist: list, df_without_labels: DataFrame):
    """KMeans visualisation over time

        centers_hist (list): The history of the centers identified by KMEans
        df_without_labels (DataFrame): the spark dataframe without labels
    """
    logging.info("Starting visualisation")
    # Init visualization colors
    colors = ["red", "blue", "green"]
    df_with_labels = pd.read_csv("data/iris.csv", header=0, delimiter=",")

    i = 0
    for centers in centers_hist:
        plot_one_iteration(centers, colors, df_with_labels,
                           df_without_labels, i)
        i += 1


def plot_one_iteration(centers: list, colors: list, df_with_labels: pd.DataFrame, df_without_labels: DataFrame, i: int):
    """Plot one KMeans iteration. One fig per iteration, with 6 subplots 
    - 5 for multidimensional plot
    - 1 to visualize correctly labeled data
    Args:
        centers (list): The centers for this iteration
        colors (list): The closest points, grouped by center for this iteration
        df_with_labels (pd.DataFrame): the dataframe, including labels
        df_without_labels (DataFrame): the dataframe without labels
        i (int): iteration number
    """
    logging.info(f"Visualizing iteration {i}")
    # Recalculate closest points by center, in order to conduct plots
    closest_points_by_center = df_without_labels.rdd\
        .map(lambda point:
             (closest_center(point, centers), point))\
        .collect()
    axs, fig = init_figure()

    plot_iteration(closest_points_by_center, axs, colors)
    plot_correctly_labeled(df_with_labels, axs, centers)

    # Customize plot
    fig.suptitle(f"Clustering visualization - iteration {i}")
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save figure
    plt.savefig(f"output/iteration-{i}")


def plot_correctly_labeled(df_with_labels, axs, centers):
    for (_, row) in df_with_labels.iterrows():
        axs[2, 1].scatter(row["sepal_length"],
                          row["sepal_width"],
                          color="green" if row['species'] == CLASSES[
            closest_center(
                np.array(row[0:4]),
                centers)]
            else "red", alpha=0.5)


def plot_iteration(closest_points_by_center, axs, colors):
    for (index, data) in closest_points_by_center:
        axs[0, 0].scatter(data[SEPAL_LENGHT_INDEX],
                          data[SEPAL_WIDTH_INDEX], color=colors[index], alpha=0.5)
        axs[0, 1].scatter(data[SEPAL_LENGHT_INDEX],
                          data[PETAL_LENGTH_INDEX], color=colors[index], alpha=0.5)
        axs[1, 0].scatter(data[SEPAL_LENGHT_INDEX],
                          data[PETAL_WIDTH_INDEX], color=colors[index], alpha=0.5)
        axs[1, 1].scatter(data[SEPAL_WIDTH_INDEX],
                          data[PETAL_LENGTH_INDEX], color=colors[index], alpha=0.5)
        axs[2, 0].scatter(data[PETAL_LENGTH_INDEX],
                          data[PETAL_WIDTH_INDEX], color=colors[index], alpha=0.5)
        axs[2, 1].set(xlabel='Sepal lenght', ylabel='Sepal width',
                      title="Correctly classified data")


def init_figure():
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].set(xlabel='Sepal lenght', ylabel='Sepal width',
                  title="Sepal lenght, Sepal width")
    axs[0, 1].set(xlabel='Sepal lenght', ylabel='Petal length',
                  title="Sepal lenght, Petal length")
    axs[1, 0].set(xlabel='Sepal lenght', ylabel='Petal width',
                  title="Sepal lenght, Petal width")
    axs[1, 1].set(xlabel='Sepal width', ylabel='Petal length',
                  title="Sepal width, Petal length")
    axs[2, 0].set(xlabel='Petal lenght', ylabel='Petal width',
                  title="Petal lenght, Petal width")
    axs[2, 1].set(xlabel='Sepal lenght', ylabel='Sepal width',
                  title="Correctly classified data")
    return axs, fig
