import pytest
from model.distance_matching import *
from tests.test_preprocessing.test_distance_matching_processing import dummy_full_data
from pyspark.sql.types import StructField, StringType, FloatType, StructType, IntegerType, ArrayType, DoubleType, \
    LongType
from tests.utils_for_test import spark, are_dfs_equal
from pyspark.testing import assertPandasOnSparkEqual
from preprocessing.distance_matching_processing import convert_table_to_pandas


@pytest.fixture
def dummy_product_info(spark):
    data = [(123, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 50.0, 11.0, 2, '0.0', 'AABCAB0.0',
             [0.12371791482634835, -0.4019870074553858, 0.7833494518006403])
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


@pytest.fixture
def dummy_input_category_code():
    return 'AABCAB0.0'


@pytest.fixture
def dummy_input_vec():
    return [0.12371791482634835, -0.4019870074553858, 0.7833494518006403]


@pytest.fixture
def dummy_product_data_lev(spark):
    data = [(123, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 50.0, 11.0, 2, '0.0', 'AABCAB0.0',
             [0.12371791482634835, -0.4019870074553858, 0.7833494518006403], 0),
            (234, 'A', 'B', 'B', 'C', 'A', 'B', 'two', 20.0, 39.0, 2, '1.0', 'ABBCAB1.0',
             [-0.6185895741317418, 1.375218709715794, 0.7833494518006403], 2),
            (345, 'B', 'C', 'B', 'C', 'A', 'C', 'two', 10.0, 2.0, 1, '1.0', 'BCBCAC1.0',
             [-0.8660254037844385, -0.9732317022604078, -0.26111648393354675], 4),
            (456, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 100.0, None, None, '0.0', 'AABCAB0.0',
             [1.3608970630898318, 0.0, -1.3055824196677337], 0)
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
        StructField("vector", ArrayType(DoubleType(), containsNull=False)),
        StructField("lev_distance", IntegerType(), True)
    ])
    return spark.createDataFrame(data=data, schema=schema)


@pytest.fixture
def dummy_product_data_euc(spark):
    data = [(123, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 50.0, 11.0, 2, '0.0', 'AABCAB0.0',
             [0.12371791482634835, -0.4019870074553858, 0.7833494518006403], 0, 0.0),
            (234, 'A', 'B', 'B', 'C', 'A', 'B', 'two', 20.0, 39.0, 2, '1.0', 'ABBCAB1.0',
             [-0.6185895741317418, 1.375218709715794, 0.7833494518006403], 2, 1.9260011911392212),
            (345, 'B', 'C', 'B', 'C', 'A', 'C', 'two', 10.0, 2.0, 1, '1.0', 'BCBCAC1.0',
             [-0.8660254037844385, -0.9732317022604078, -0.26111648393354675], 4, 1.548167109489441),
            (456, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 100.0, None, None, '0.0', 'AABCAB0.0',
             [1.3608970630898318, 0.0, -1.3055824196677337], 0, 2.460862159729004)
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
        StructField("vector", ArrayType(DoubleType(), containsNull=False)),
        StructField("lev_distance", IntegerType(), True),
        StructField("euc_distance", FloatType(), True)
    ])

    return spark.createDataFrame(data=data, schema=schema)


@pytest.fixture
def dummy_pd_spark_data(spark):
    data = [(123, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 50.0, 11.0, 2, '0.0', 'AABCAB0.0',
             [0.12371791482634835, -0.4019870074553858, 0.7833494518006403], 0, 0.0, -1.0931021656524806),
            (234, 'A', 'B', 'B', 'C', 'A', 'B', 'two', 20.0, 39.0, 2, '1.0', 'ABBCAB1.0',
             [-0.6185895741317418, 1.375218709715794, 0.7833494518006403], 2, 1.9260011911392212, 0.3396232375838375),
            (345, 'B', 'C', 'B', 'C', 'A', 'C', 'two', 10.0, 2.0, 1, '1.0', 'BCBCAC1.0',
             [-0.8660254037844385, -0.9732317022604078, -0.26111648393354675], 4, 1.548167109489441, 0.683239974708457),
            (456, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 100.0, None, None, '0.0', 'AABCAB0.0',
             [1.3608970630898318, 0.0, -1.3055824196677337], 0, 2.460862159729004, 0.07023895336018615)
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
        StructField("vector", ArrayType(DoubleType(), containsNull=False)),
        StructField("lev_distance", IntegerType(), True),
        StructField("euc_distance", FloatType(), True),
        StructField("distance", FloatType(), True)
    ])
    df = spark.createDataFrame(data=data, schema=schema)
    df = convert_table_to_pandas(df)
    df['distance'] = df['distance'].astype('float64')
    return df


@pytest.fixture
def other_products_pd_data(spark):
    data = [(234, 'A', 'B', 'B', 'C', 'A', 'B', 'two', 20.0, 39.0, 2, '1.0', 'ABBCAB1.0',
             [-0.6185895741317418, 1.375218709715794, 0.7833494518006403], 2, 1.9260011911392212, 0.3396232375838375),
            (345, 'B', 'C', 'B', 'C', 'A', 'C', 'two', 10.0, 2.0, 1, '1.0', 'BCBCAC1.0',
             [-0.8660254037844385, -0.9732317022604078, -0.26111648393354675], 4, 1.548167109489441, 0.683239974708457),
            (456, 'A', 'A', 'B', 'C', 'A', 'B', 'one', 100.0, None, None, '0.0', 'AABCAB0.0',
             [1.3608970630898318, 0.0, -1.3055824196677337], 0, 2.460862159729004, 0.07023895336018615)
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
        StructField("vector", ArrayType(DoubleType(), containsNull=False)),
        StructField("lev_distance", IntegerType(), True),
        StructField("euc_distance", FloatType(), True),
        StructField("distance", FloatType(), True)
    ])
    df = spark.createDataFrame(data=data, schema=schema)
    df = convert_table_to_pandas(df)
    df['distance'] = df['distance'].astype('float64')
    return df


@pytest.fixture
def recommendation():
    return [456]


def test_get_input_product_info(dummy_full_data, dummy_product_info):
    expected = dummy_product_info
    actual = get_input_product_info(123, dummy_full_data)
    return are_dfs_equal(actual, expected)


def test_get_input_points(dummy_product_info, dummy_input_category_code, dummy_input_vec):
    expected_cat_code, expected_vec = dummy_input_category_code, dummy_input_vec
    cat_code, vec = get_input_points(dummy_product_info)
    assert all([cat_code == expected_cat_code, vec == expected_vec])


def test_calculate_lev_distance_col(dummy_full_data, dummy_input_category_code, dummy_product_data_lev):
    expected = dummy_product_data_lev
    actual = calculate_lev_distance_col(dummy_full_data, dummy_input_category_code)
    return are_dfs_equal(actual, expected)


def test_calculate_euc_distance_col(dummy_product_data_lev, dummy_input_vec, dummy_product_data_euc):
    expected = dummy_product_data_euc
    actual = calculate_euc_distance_col(dummy_product_data_lev, dummy_input_vec)
    return are_dfs_equal(actual, expected)


def test_unify_distances(dummy_product_data_euc, dummy_pd_spark_data):
    expected = dummy_pd_spark_data
    actual = unify_distances(dummy_product_data_euc, categorical_param=0.5, numerical_param=0.5)
    return assertPandasOnSparkEqual(actual, expected, checkExact=False)


def test_get_other_products(dummy_pd_spark_data, other_products_pd_data):
    expected = other_products_pd_data
    actual = get_other_products(dummy_pd_spark_data, 123)
    return assertPandasOnSparkEqual(actual, expected, checkExact=False, checkRowOrder=False)


def test_find_similar_n_products(other_products_pd_data, recommendation, n=1):
    expected = recommendation
    actual = find_similar_n_products(other_products_pd_data, n)
    assert actual == expected
