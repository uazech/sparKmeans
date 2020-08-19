""" Main SparkKeans module
"""
import logging
from logging import config as logging_config

from constants import APP_LOGGING_LEVEL
from kmeans import get_centers
from spark import init_spark, read_input_data, stop_spark
from visualize import visualize

# Init logging config
logging.basicConfig(level=APP_LOGGING_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def launch():
    """Launches the app
    """
    logging.info("Starting execution")
    spark_session = init_spark()

    df = read_input_data(spark_session)
    centers, centers_history = get_centers(df)
    visualize(centers_history, df)

    stop_spark(spark_session)


if __name__ == "__main__":
    launch()
