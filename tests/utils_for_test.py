import pytest
from pyspark.sql.types import StructField, StringType, FloatType, StructType, IntegerType, ArrayType, DoubleType, LongType
from pyspark.sql import SparkSession
from pyspark.testing import assertDataFrameEqual


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def are_dfs_equal(df1, df2):
    return assertDataFrameEqual(df1, df2, checkRowOrder=True)