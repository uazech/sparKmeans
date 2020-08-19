import enum
import yaml

with open("config.yml") as file:
    _CONFIG = yaml.load(file, Loader=yaml.FullLoader)


CONVERGE_DIST = _CONFIG["converge_dist"]
NB_CLASSES = _CONFIG["nb_classes"]

APP_LOGGING_LEVEL = _CONFIG["app_logging_level"]
SPARK_LOGGING_LEVEL = _CONFIG["spark_logging_level"]

INCLUDE_FILES = _CONFIG["include_files"]

SEPAL_LENGHT_INDEX = 0
SEPAL_WIDTH_INDEX = 1
PETAL_LENGTH_INDEX = 2
PETAL_WIDTH_INDEX = 3
CLASSES = ["setosa", "versicolor", "virginica"]

