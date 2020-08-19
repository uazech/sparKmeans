"""
Spark configuration module
"""
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType

from constants import SPARK_LOGGING_LEVEL, INCLUDE_FILES


def read_input_data(spark_session: SparkSession):
    """Reads input data, which will be used by Kmeans

    Args:
        spark_session (SparkSession): the Spark session

    Returns:
        [pyspark.sql.DataFrame]: the dataframe
    """
    logging.info("Reading input data")

    df = spark_session.read.csv("data/iris.csv", header=True)
    df = df.drop("species")  # Drop species column (would be cheating...)
    # Cast columns to numerical types
    df = df.withColumn("sepal_length", df["sepal_length"].cast(DoubleType()))
    df = df.withColumn("sepal_width", df["sepal_width"].cast(DoubleType()))
    df = df.withColumn("petal_length", df["petal_length"].cast(DoubleType()))
    df = df.withColumn("petal_width", df["petal_width"].cast(DoubleType()))

    return df


def init_spark():
    """Create new spark session

    Returns:
        [SparkSession]: the spark session
    """
    logging.info("Creating new spark session")
    spark_session = SparkSession.builder\
        .appName('sparKmeans').getOrCreate()
    spark_context = spark_session.sparkContext
    
    for file_name in INCLUDE_FILES:
        spark_context.addPyFile(f"src/{file_name}")

    spark_context.setLogLevel(SPARK_LOGGING_LEVEL)
    return spark_session


def stop_spark(spark_session: SparkSession):
    """Stops the spark session

    Args:
        spark_session (SparkSession): The spark_session to stop
    """
    logging.info("Stopping Spark session")
    spark_session.stop()
