from pyspark.sql.types import StructField, StringType, FloatType, StructType, IntegerType, ArrayType, DoubleType, LongType
from pyspark.sql import SparkSession
from tests.utils_for_test import spark, are_dfs_equal
from preprocessing.distance_matching_processing import *

import pytest


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



@pytest.fixture
def dummy_sales(spark):
    data = [
        (12345,123,10.0),
        (23456,123,12.0),
        (23456,234,40.0),
        (12345,345,2.0),
        (34567,234,38.0)
    ]
    schema = StructType([
        StructField("customer_id",IntegerType(),True),
        StructField("product_id_h",IntegerType(),True),
        StructField("gross_value_dkk",FloatType(),True)
    ])

    return spark.createDataFrame(data=data,schema=schema)


@pytest.fixture
def dummy_product(spark):
    data = [
        (123,'A','A','B','C','A','B', 'one',50.0),
        (234,'A','B','B','C','A','B', 'two',20.0),
        (345,'B','C','B','C','A','C', 'two',10.0),
        (456,'A','A','B','C','A','B', 'one',100.0)
    ]
    schema = StructType([
        StructField("product_id_h",IntegerType(),True),
        StructField("category",StringType(),True),
        StructField("collection", StringType(), True),
        StructField("color", StringType(), True),
        StructField("material", StringType(), True),
        StructField("metal_type", StringType(), True),
        StructField("stone", StringType(), True),
        StructField("theme", StringType(), True),
        StructField("size", DoubleType(), True)
    ])
    return spark.createDataFrame(data=data,schema=schema)


@pytest.fixture
def dummy_product_data(spark):
    data = [
        (123,'A','A','B','C','A','B', 'one',50.0,11.0,2),
        (234,'A','B','B','C','A','B', 'two',20.0,39.0,2),
        (345,'B','C','B','C','A','C', 'two',10.0,2.0,1),
        (456,'A','A','B','C','A','B', 'one',100.0, None, None)
    ]
    schema = StructType([
        StructField("product_id_h",IntegerType(),True),
        StructField("category",StringType(),True),
        StructField("collection", StringType(), True),
        StructField("color", StringType(), True),
        StructField("material", StringType(), True),
        StructField("metal_type", StringType(), True),
        StructField("stone", StringType(), True),
        StructField("theme", StringType(), True),
        StructField("size", DoubleType(), True),
        StructField("avg_value", DoubleType(), True),
        StructField("sales_amt", LongType(), True)
    ])
    return spark.createDataFrame(data=data,schema=schema)


@pytest.fixture
def dummy_product_data_theme(spark):
    data = [
        (123,'A','A','B','C','A','B','one',50.0,11.0,2,'0.0'),
        (234,'A','B','B','C','A','B','two',20.0,39.0,2,'1.0'),
        (345,'B','C','B','C','A','C','two',10.0,2.0,1,'1.0'),
        (456,'A','A','B','C','A','B','one',100.0, None, None,'0.0')
    ]
    schema = StructType([
        StructField("product_id_h",IntegerType(),True),
        StructField("category",StringType(),True),
        StructField("collection", StringType(), True),
        StructField("color", StringType(), True),
        StructField("material", StringType(), True),
        StructField("metal_type", StringType(), True),
        StructField("stone", StringType(), True),
        StructField("theme", StringType(), True),
        StructField("size", DoubleType(), True),
        StructField("avg_value", DoubleType(), True),
        StructField("sales_amt", LongType(), True),
        StructField("theme_id", StringType(), True)
    ])
    return spark.createDataFrame(data=data,schema=schema)


@pytest.fixture
def numerical_features():
    return ['size', 'avg_value', 'sales_amt']


@pytest.fixture
def categorical_features():
    return ['category','collection','color','material','metal_type', 'stone', 'theme_id']


@pytest.fixture
def dummy_product_data_string(spark):
    data = [(123,'A','A','B','C','A','B','one',50.0,11.0,2,'0.0','AABCAB0.0'),
            (234,'A','B','B','C','A','B','two',20.0,39.0,2,'1.0','ABBCAB1.0'),
            (345,'B','C','B','C','A','C','two',10.0,2.0,1,'1.0','BCBCAC1.0'),
            (456,'A','A','B','C','A','B','one',100.0, None, None,'0.0','AABCAB0.0')
            ]
    schema = StructType([
        StructField("product_id_h",IntegerType(),True),
        StructField("category",StringType(),True),
        StructField("collection", StringType(), True),
        StructField("color", StringType(), True),
        StructField("material", StringType(), True),
        StructField("metal_type", StringType(), True),
        StructField("stone", StringType(), True),
        StructField("theme", StringType(), True),
        StructField("size", DoubleType(), True),
        StructField("avg_value", DoubleType(), True),
        StructField("sales_amt", LongType(), True),
        StructField("theme_id", StringType(), True),
        StructField("category_string", StringType(), True)
    ])
    return spark.createDataFrame(data=data, schema=schema)


@pytest.fixture
def dummy_processed_numerical_features(spark):
    data = [(123, 0.12371791482634835, -0.4019870074553858, 0.7833494518006403, [0.12371791482634835, -0.4019870074553858, 0.7833494518006403]),
            (234, -0.6185895741317418, 1.375218709715794, 0.7833494518006403, [-0.6185895741317418, 1.375218709715794, 0.7833494518006403]),
            (345, -0.8660254037844385, -0.9732317022604078, -0.26111648393354675, [-0.8660254037844385, -0.9732317022604078, -0.26111648393354675]),
            (456, 1.3608970630898318, 0.0, -1.3055824196677337, [1.3608970630898318, 0.0, -1.3055824196677337])
            ]
    schema = StructType([
        StructField("product_id_h", IntegerType(), True),
        StructField("size", DoubleType(), True),
        StructField("avg_value", DoubleType(), True),
        StructField("sales_amt", DoubleType(), True),
        StructField("vector", ArrayType(DoubleType(), containsNull=False))
    ])
    return spark.createDataFrame(data=data, schema=schema)


@pytest.fixture
def dummy_full_data(spark):
    data = [(123, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 50.0, 11.0, 2, '0.0', 'AABCAB0.0', [0.12371791482634835, -0.4019870074553858, 0.7833494518006403]),
            (234, 'A', 'B', 'B', 'C', 'A', 'B', 'two', 20.0, 39.0, 2, '1.0', 'ABBCAB1.0', [-0.6185895741317418, 1.375218709715794, 0.7833494518006403]),
            (345, 'B', 'C', 'B', 'C', 'A', 'C', 'two', 10.0, 2.0, 1, '1.0', 'BCBCAC1.0', [-0.8660254037844385, -0.9732317022604078, -0.26111648393354675]),
            (456, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 100.0, None, None, '0.0', 'AABCAB0.0', [1.3608970630898318, 0.0, -1.3055824196677337])
            ]
    schema = StructType([
        StructField("product_id_h", IntegerType(), True),
        StructField("category", StringType(), True),
        StructField("collection", StringType(), True),
        StructField("color", StringType(), True),
        StructField("material", StringType(), True),
        StructField("metal_type", StringType(), True),
        StructField("stone", StringType(), True),
        StructField("theme", StringType(), True),
        StructField("size", DoubleType(), True),
        StructField("avg_value", DoubleType(), True),
        StructField("sales_amt", LongType(), True),
        StructField("theme_id", StringType(), True),
        StructField("category_string", StringType(), True),
        StructField("vector", ArrayType(DoubleType(), containsNull=False))
    ])
    return spark.createDataFrame(data=data, schema=schema)



def test_create_product_features(dummy_sales, dummy_product, dummy_product_data):
    expected = dummy_product_data
    actual = create_product_features(dummy_sales, dummy_product)
    return are_dfs_equal(actual, expected)


def test_create_category_string(dummy_product_data_theme, dummy_product_data_string, categorical_features):
    expected = dummy_product_data_string
    actual = create_category_string(dummy_product_data_theme, categorical_features)
    return are_dfs_equal(actual, expected)


def test_process_numerical_features(dummy_product_data_string, dummy_processed_numerical_features, numerical_features):
    expected = dummy_processed_numerical_features
    actual = process_numerical_features(dummy_product_data_string, numerical_features)
    return are_dfs_equal(actual, expected)


def test_join_to_full_product_data(dummy_processed_numerical_features, dummy_product_data_string, dummy_full_data):
    expected = dummy_full_data
    actual = join_to_full_product_data(dummy_processed_numerical_features, dummy_product_data_string)
    return are_dfs_equal(actual, expected)


def test_create_theme_col(dummy_product_data, dummy_product_data_theme):
    expected = dummy_product_data_theme
    actual = create_theme_col(dummy_product_data)
    return are_dfs_equal(actual, expected)





