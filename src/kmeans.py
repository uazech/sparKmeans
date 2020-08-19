""" Implementation of KMeans
"""
import logging

import numpy as np
from pyspark import Row
from pyspark.sql import DataFrame, SparkSession
from scipy.spatial.distance import euclidean

from constants import CONVERGE_DIST, NB_CLASSES


def closest_center(point: Row, centers: list):
    """Calculate the closest center given a single point

    Args:
        point (Row): the point
        centers (list): the centers
    Returns:
        [int] -- the index of the closest center
    """
    closest_center_index = 0
    centerDistance = float("+inf")
    for i in range(len(centers)):
        # Get Euclydian dist
        temp_dist = euclidean(point, centers[i])
        if temp_dist < centerDistance:
            centerDistance = temp_dist
            closest_center_index = i
    return closest_center_index


def get_centers(df: DataFrame):
    """ Get KMeans centroids from the dataframe. DF must only contain numerical value

    Args:
        df (SparkSession): The dataframe

    Returns:
        [list]: the final centers
        [list]: the centers history (used for visualization)
    """
    logging.info("Starting KMeans")

    # Init the centers at a random state, by taking a K subsample
    centers = df.rdd.takeSample(False, NB_CLASSES, 1)
    # Keep center history
    centers_hist = [centers.copy()]
    dist = float("+inf")
    i = 1
    while dist > CONVERGE_DIST:
        logging.info(f"Starting iteration {i}")
        # Iterate until the distance is lower than the converging distance
        dist = iterate(df, centers, centers_hist)
        i +=1
    logging.info(f"Final centers: {centers}")
    logging.info(f"Nb iteration: {len(centers_hist) - 1} ")
    return (centers, centers_hist)


def iterate(df: DataFrame, centers: list, centers_hist: list):
    """One Kmean iteration. 
     - Updates the centers list
     - Updates the centers hist list

    Args:
        df (DataFrame): the dataframe containing the data
        centers (list): the list of centers
        centers_hist (list): list containing the history of the centers

    Returns:
        float: the converging distance
    """
    # Define the closest points by centers
    closest_points_by_center = df.rdd.map(lambda point:
                                          (closest_center(point, centers), (point, 1)))

    # Define aggregated points by centers, which is the sum of the closest points by centers
    aggregated_points_by_center = closest_points_by_center\
        .reduceByKey(lambda a, b:
                     (np.array(a[0])+np.array(b[0]), a[1]+b[1]))

    # Debug by printing centroids if needed
    aggregated_points_by_center.foreach(lambda x: logging.debug(x))

    # Define new centers based on the agregated points by center
    new_centers = aggregated_points_by_center.\
        map(lambda x: (x[0], x[1][0]/x[1][1])).\
        collect()

    # Define new distance between the old centers and the new ones
    dist = sum(np.sum((centers[index]-newCenter)**2)
               for (index, newCenter) in new_centers)

    # Define new centers for next iteration
    for (index, center) in new_centers:
        centers[index] = center
    centers_hist.append(centers.copy())
    return dist
