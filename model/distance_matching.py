from typing import List

from pyspark.sql import functions as F, types as T, DataFrame
from scipy.spatial import distance
from preprocessing.distance_matching_processing import convert_table_to_pandas
import pyspark.pandas as ps


def get_input_product_info(input: int, product_data: DataFrame) -> DataFrame:
    """
    :param input: Hashed product id
    :param product_data: Table containing all product data
    :return: Returns a 1-row table containing the data belonging to the specified hashed product_id
    """
    input_product_info = product_data.filter(product_data.product_id_h == input)
    return input_product_info


def get_input_points(input_product_info: DataFrame) -> (str, T.array):
    """
    :param input_product_info: 1-row DataFrame containing data for one product
    :return:
        input_category_code: string aggregating all categorical variables for the product
        input_vec: vector aggregating all numerical variables for the product
    """
    input_category_code = input_product_info.rdd.map(lambda x: x['category_string']).collect()[0]
    input_vec = input_product_info.rdd.map(lambda x: x['vector']).collect()[0]
    return input_category_code, input_vec


def calculate_lev_distance_col(product_data: DataFrame, input_category_code: str) -> DataFrame:
    """
    :param product_data: DataFrame containing data for all products
    :param input_category_code: String aggregating categorical data for the desired product
    :return: DataFrame including a column called 'lev_distance' which specifies the Levenshtein distance
        between the input_category_code and the category codes corresponding to all other products in product_data
    """
    product_data_lev = product_data.withColumn('lev_distance',
                                               F.levenshtein(F.lit(input_category_code), F.col('category_string')))
    return product_data_lev


def calculate_euc_distance_col(product_data_lev: DataFrame, input_vec: T.array) -> DataFrame:
    """
    :param product_data_lev: DataFrame containing all product data including Levenshtein distances
    :param input_vec: array containing numerical variables for the desired product
    :return: DataFrame including all columns in product_data_lev as well as a column called 'euc_distance'
        which measures the euclidean distance between input_vec and all other product vectors in the table
    """
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, input_vec)), T.FloatType())
    product_data_dist = product_data_lev.withColumn('euc_distance', distance_udf(F.col('vector')))
    return product_data_dist


def unify_distances(product_data_dist: DataFrame, categorical_param: float = 0.8,
                    numerical_param: float = 0.2) -> ps.DataFrame:
    """
    :param product_data_dist: DataFrame containing product data, as well as Levenshtein and Euclidean distances
    :param categorical_param: Weight associated to the Levenshtein distance component
    :param numerical_param: Weight associated to the Euclidean distance component
    :return: Adds the column 'distance' to product_data_dist. This column is a normalised sum of both Levenshtein and
        Euclidean distances, multiplied by their respective weight.
    """
    pd_spark_data = convert_table_to_pandas(product_data_dist)
    norm_lev_dist = (pd_spark_data['lev_distance'] - pd_spark_data['lev_distance'].mean()) / pd_spark_data[
        'lev_distance'].std()
    norm_euc_dist = (pd_spark_data['euc_distance'] - pd_spark_data['euc_distance'].mean()) / pd_spark_data[
        'euc_distance'].std()
    pd_spark_data['distance'] = categorical_param * norm_lev_dist + numerical_param * norm_euc_dist
    return pd_spark_data


def get_other_products(pd_spark_data: ps.DataFrame, input: int) -> ps.DataFrame:
    """
    :param pd_spark_data: Pyspark pandas dataframe including product data
    :param input: Hashed product id corresponding to the desired product
    :return: DataFrame containing the data for all products except the input product
    """
    other_products = pd_spark_data.loc[pd_spark_data['product_id_h'] != input]
    return other_products


def find_similar_n_products(other_products: ps.DataFrame, n: int) -> List[int]:
    """
    :param other_products: DataFrame containing the data for all products except the input product
    :param n: Number of recommendations to return
    :return: List of length n containing the products which are (in order) closest to the input product
    """
    most_similar = other_products.nsmallest(n, columns=['distance'])
    recommendations = most_similar.drop_duplicates(subset='product_id_h').sort_values('distance')
    return recommendations['product_id_h'].to_list()
