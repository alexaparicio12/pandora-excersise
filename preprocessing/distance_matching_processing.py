from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F, types as T, DataFrame
import pyspark.pandas as ps
from typing import List

NUMERICAL_FEATURES = ['size', 'avg_value', 'sales_amt']
#CATEGORICAL_FEATURES = ['category', 'collection', 'color', 'material', 'metal_type', 'stone']
CATEGORICAL_FEATURES = ['category', 'collection', 'color', 'material', 'metal_type', 'stone', 'theme_id']


def create_product_features(sales: DataFrame, df_product: DataFrame) -> DataFrame:
    """
    :param sales: DataFrame containing only sales of products
    :param df_product: DataFrame containing product data
    :return: Collects numerical features from sales data and adds these columns to the product data.
    """
    values = sales[['product_id_h', 'gross_value_dkk']].groupby('product_id_h').agg(F.mean('gross_value_dkk'),
                                                                                    F.count('gross_value_dkk'))
    values = values.withColumnRenamed('avg(gross_value_dkk)', 'avg_value').withColumnRenamed('count(gross_value_dkk)',
                                                                                             'sales_amt')
    product_data = df_product.join(values, on='product_id_h', how='left')
    return product_data


def create_theme_col(df_product: DataFrame) -> DataFrame:
    df_product = df_product.fillna({'theme': 'Null'})
    indexer = StringIndexer(inputCol="theme", outputCol="theme_id")
    indexed_df_product = indexer.fit(df_product).transform(df_product)
    indexed_df_product = indexed_df_product.withColumn('theme_id', F.col('theme_id').cast(T.StringType()))
    return indexed_df_product


def create_category_string(product_data: DataFrame,
                           categorical_features: List[str] = CATEGORICAL_FEATURES) -> DataFrame:
    """
    Creates a column with a string which represents all categorical variables of the product.
    """
    product_data = product_data.withColumn('category_string',
                                           F.concat(*[F.col(column) for column in categorical_features]))
    return product_data


def convert_table_to_pandas(table: DataFrame) -> ps.DataFrame:
    """
    Converts a pyspark DataFrame to a pyspark.pandas DataFrame
    """
    table = table.drop('__index_level_0__')
    pd_table = table.pandas_api()
    return pd_table


def normalise_numerical_features(numeric_product_data: ps.DataFrame, col: str) -> ps.DataFrame:
    """
    Normalises the numerical features by substracting the mean and dividing by the standard deviation
    """
    numeric_product_data[col] = (numeric_product_data[col] - numeric_product_data[col].mean()) / numeric_product_data[
        col].std()
    return numeric_product_data


def create_numeric_vector_col(numeric_product_data: ps.DataFrame,
                              numerical_features: List[str] = NUMERICAL_FEATURES) -> DataFrame:
    """
    Adds a column to numeric_product_data which includes all the numerical features stored into a vector.
    """
    numeric_data_vec = numeric_product_data.to_spark()
    numeric_data_vec = numeric_data_vec.withColumn('vector', F.array(*[F.col(column) for column in numerical_features]))
    return numeric_data_vec


def process_numerical_features(product_data: DataFrame,
                               numerical_features: List[str] = NUMERICAL_FEATURES) -> DataFrame:
    """
    Processes the numerical features. This includes normalisation and the creationg of the vector column.
    """
    numeric_product_data = product_data[['product_id_h'] + numerical_features]
    numeric_product_data = convert_table_to_pandas(numeric_product_data)
    for col in numerical_features:
        if col == 'sales_amt':
            numeric_product_data[col] = numeric_product_data[col].fillna(0)
        else:
            numeric_product_data[col] = numeric_product_data[col].fillna(numeric_product_data[col].mean())
        numeric_product_data = normalise_numerical_features(numeric_product_data, col)
    numeric_data_vec = create_numeric_vector_col(numeric_product_data)
    return numeric_data_vec


def join_to_full_product_data(numeric_data_vec: DataFrame, product_data: DataFrame) -> DataFrame:
    """
    Joins the processed numerical features to the product data.
    """
    product_data = product_data.join(numeric_data_vec[['product_id_h', 'vector']], on='product_id_h', how='left')
    return product_data


def save_product_data(full_product_data: DataFrame,
                      path: str = 'Data/trained_data/full_product_data.parquet') -> DataFrame:
    """
    Saves the processed product data. This enables quick lookups when calling the API and avoids repeating transformations.
    """
    full_product_data.write.parquet(path, mode='overwrite')
    return full_product_data
