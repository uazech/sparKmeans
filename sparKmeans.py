

# %% Init pyspark & import libs
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName('sparKmeans').getOrCreate()
# %% Load and read CSV
df = spark.read.csv("iris.csv", header=True)
df.head(5)

# %% Set col types and drop columns
df = df.drop("species")
df = df.withColumn("sepal_length", df["sepal_length"].cast(DoubleType()))
df = df.withColumn("sepal_width", df["sepal_width"].cast(DoubleType()))
df = df.withColumn("petal_length", df["petal_length"].cast(DoubleType()))
df = df.withColumn("petal_width", df["petal_width"].cast(DoubleType()))
df.head(5)

# %% Kmeans settings
K = 3
CLASSES = ["setosa", "versicolor", "virginica"]
CONVERGE_DIST = 0.01
centers = df.rdd.takeSample(False, K, 1)
centers
# %% Constants
SEPAL_LENGHT_INDEX = 0
SEPAL_WIDTH_INDEX = 1
PETAL_LENGTH_INDEX = 2
PETAL_WIDTH_INDEX = 3
# %% Def functions


def closestCenter(point, centers: list):
    """ Calculate the closest center for a single point
    Arguments:
        point {spark.Row} -- the point
        centers {list(spark.Row)} -- the centers
    Returns:
        [int] -- the index of the closest center
    """
    closestCenterIndex = 0
    centerDistance = float("+inf")
    for i in range(len(centers)):
        # Get Euclydian dist
        tempDist = np.sum((np.array(point) - np.array(centers[i])) ** 2)
        if tempDist < centerDistance:
            centerDistance = tempDist
            closestCenterIndex = i
    return closestCenterIndex


# %% Calculate the clusters centers (iterative)
centersHist = [centers[:]]
dist = float("+inf")
while dist > CONVERGE_DIST:
    # Define the closest points by centers
    closestPointsByCenter = df.rdd.map(lambda point:
                                       (closestCenter(point, centers), (point, 1)))

    # Define aggregated points by centers, which is the sum of the closest points by centers
    aggregatedPointsByCenter = closestPointsByCenter\
        .reduceByKey(lambda a, b:
                     (np.array(a[0])+np.array(b[0]), a[1]+b[1]))
    aggregatedPointsByCenter.foreach(lambda x: print(x))
    # Define new centers based on the agregated points by center
    newCenters = aggregatedPointsByCenter.\
        map(lambda x: (x[0], x[1][0]/x[1][1])).\
        collect()

    # Define new distance between the old centers and the new ones
    dist = sum(np.sum((centers[index]-newCenter)**2)
               for (index, newCenter) in newCenters)

    # Define new centers for next iteration
    for (index, center) in newCenters:
        centers[index] = center
    centersHist.append(centers[:])

print("Final centers", str(centers))
print("Nb iteration : ", len(centersHist) - 1)

# %% Visualize the clustering step by step
colors = ["red", "blue", "green"]
dfWithLabels = pd.read_csv("iris.csv", header=0, delimiter=",")

i = 0
for centers in centersHist:
    # get the closest points mapped for each center
    closestPointsByCenter = df.rdd\
        .map(lambda point:
             (closestCenter(point, centers), point))\
        .collect()

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

    # plot
    for (index, data) in closestPointsByCenter:
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

    # plot correctly labbeled data / incorrectly
    for (index, row) in dfWithLabels.iterrows():
        axs[2, 1].scatter(row["sepal_length"],
                          row["sepal_width"],
                          color="green" if row['species'] == CLASSES[
                              closestCenter(
                                  np.array(row[0:4]),
                                  centers)]
                          else "red", alpha=0.5)
    fig.suptitle(f"Clustering visualization - iteration {i}")
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig("iteration"+str(i))
    i += 1


# %% Stop Spark
spark.stop()
