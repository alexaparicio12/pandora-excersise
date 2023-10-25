from pyspark.sql import SparkSession


def initialise_spark() -> SparkSession:
    spark = SparkSession.builder \
        .master('local') \
        .appName('myAppName') \
        .config('spark.executor.memory', '5gb') \
        .config("spark.cores.max", "6") \
        .getOrCreate()
    return spark
