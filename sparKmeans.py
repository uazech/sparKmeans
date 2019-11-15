

# %% Init pyspark & import libs
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import numpy as np
import matplotlib.pyplot as plt

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
CONVERGE_DIST = 0.01
centers = df.rdd.takeSample(False, K, 1)
centers
# %% Constants

# %% Def functions


def closestCenter(point, centers: list):
    # TODO : optimiser ?
    """ Calculate the closest center for a single point
    Arguments:
        point {spark.Row} -- the point
        centers {list(spark.Row)} -- the centers
    Returns:
        [int] -- the index of the closest center
    """
    print(type(point), type(centers))
    closestCenterIndex = 0
    centerDistance = float("+inf")
    for i in range(len(centers)):
        # Get Euclydian dist
        tempDist = np.sum((np.array(point) - np.array(centers[i])) ** 2)
        if tempDist < centerDistance:
            centerDistance = tempDist
            closestCenterIndex = i
    return closestCenterIndex


# %% pseudo code
centersHist = [centers]
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
for centers in centersHist:
    closestPointsByCenter = df.rdd\
        .map(lambda point:
             (closestCenter(point, centers), point))\
        .collect()
    color = "red"
    fig, axs = plt.subplots(3, 2)
    for (index, data) in closestPointsByCenter:
        axs[0, 0].scatter(data[0], data[1], color=colors[index])
        axs[0, 1].scatter(data[0], data[2], color=colors[index])
        axs[1, 0].scatter(data[0], data[3], color=colors[index])
        axs[1, 1].scatter(data[1], data[2], color=colors[index])
        axs[2, 0].scatter(data[2], data[3], color=colors[index])
        axs[2, 0].scatter(data[2], data[3], color=colors[index])
        # plt.scatter(data[0], data[1], color=colors[index])
    # axs[0, 0].scatter(centers[0], centers[1], color="orange")

    plt.show()
