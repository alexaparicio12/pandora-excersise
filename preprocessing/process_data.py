from typing import List

from pyspark.sql import functions as F, SparkSession, DataFrame


def read_table(spark: SparkSession, path: str) -> DataFrame:
    """
    Reads a parquet file saved at the path location specified
    """
    return spark.read.parquet(path)


def hash_ids(table: DataFrame, cols_to_hash: List[str], suffix: str = '_h') -> DataFrame:
    """
    Hashes the string IDs in table at columns cols_to_hash, and includes the hash columns using the suffix for name.
    """
    for col in cols_to_hash:
        table = table.withColumn(col + suffix, F.hash(col))
    return table


def get_hashed_ids_from_input(spark: SparkSession, customer_id: str, product_id: str) -> (int, int):
    """
    Retrieves a hashed customer_id and product_id given the unhashed strings. Must be done through spark to ensure the
    same hashfunction.
    """
    hashed_customer_id, hashed_product_id = spark.createDataFrame(
        [(customer_id, product_id)], ['a', 'b']
    ).select(F.hash('a').alias('hash_customer_id'), F.hash('b').alias('hash_customer_id')).collect()[0][:]
    return hashed_customer_id, hashed_product_id


def save_product_hash(product_df: DataFrame, path: str = '../Data/trained_data/hashtable.parquet',
                      product_id_col: str = 'product_id', hashed_col: str = 'product_id_h') -> DataFrame:
    """
    Saves the table containing the mapping of string product ids and hashed product ids. This is used to lookup and
    convert the results when calling the API, to avoid further calculations.
    """
    hash_table = product_df[[product_id_col, hashed_col]]
    hash_table.write.mode('overwrite').parquet(path)
    return hash_table


def get_product_list(product_df: DataFrame) -> List[str]:
    "Returns a list of all the products available within the product data"
    return product_df.rdd.map(lambda x: x['product_id']).collect()


def get_only_sales(orders_df: DataFrame) -> DataFrame:
    "Filters out any operations which do not correspond to sales in the data."
    sales = orders_df.filter(orders_df.quantity > 0)
    return sales


def unhash_product_ids(product_ids: List[int], hash_table: DataFrame) -> List[str]:
    """
    :param product_ids: List of hashed product IDs
    :param hash_table: Table mapping hashed product IDs to their unhashed original version
    :return: List containing the unhashed product IDs
    """
    unhashed_product_ids = []
    for id in product_ids:
        unhashed_id = hash_table.filter(hash_table.product_id_h == id).rdd.map(lambda x: x['product_id']).collect()[0]
        unhashed_product_ids.append(unhashed_id)

    return unhashed_product_ids


def process_purchases(sales: DataFrame, sales_outlier_cap: int = 6, ignore_amount: bool = True) -> DataFrame:
    """
    Processes the purchases to generate the table required for the training of the Collaborative Filtering model.
    :param sales: Orders data containing only sales
    :param sales_outlier_cap: Value after which a sale is considered an outlier and no longer determines that the
        customer "likes" the product.
    :param ignore_amount:
        If True, the total number of sales is irrelevant to the model. This is what the parameters have been optimised
        for. The model assumes that a customer who buys a product 3 times likes the product as much as a customer who
        buys it only once.
        If set to False, the "score" for the product will be given by the number of times a customer bought it.
    :return:
    """
    collected_buys = sales[['customer_id_h', 'product_id_h', 'quantity']].groupBy(
        ['customer_id_h', 'product_id_h']).sum('quantity')
    collected_buys = collected_buys.withColumnRenamed('sum(quantity)', 'quantity')
    collected_buys = collected_buys.filter(collected_buys.quantity < sales_outlier_cap)
    if ignore_amount:
        collected_buys = collected_buys.withColumn('quantity', F.lit(1))
    return collected_buys
