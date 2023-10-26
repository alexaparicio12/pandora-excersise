from pyspark.sql import SparkSession


def initialise_spark() -> SparkSession:
    """
    Initialises spark for the pipelines to run. With more time would write the parameters into a .yaml config file.
    """
    spark = SparkSession.builder \
        .master('local') \
        .appName('myAppName') \
        .config('spark.executor.memory', '5gb') \
        .config("spark.cores.max", "6") \
        .getOrCreate()
    return spark
