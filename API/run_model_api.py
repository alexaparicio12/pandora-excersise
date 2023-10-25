from flask import Flask
from pyspark.sql import functions as F
from setup.setup_spark import initialise_spark
from preprocessing.process_data import *
from preprocessing.distance_matching_processing import *
from model.train_collaborative_filtering import *
from model.distance_matching import *
from pipeline.train_pipeline import FIRST_TIME

FIRST_TIME = False

app = Flask(__name__)


@app.route('/')
def index():
    return 'This is the base page. To access model predictions, enter the url you see right now followed by <customer_id>/<product_id>/<number_of_recommendations>' \
           'An example could be de4c10ed-6b29-4458-b76d-edd364b23f23/4f0c1e0b-58ce-4071-8050-4d5ddf195306/4'


@app.route('/<customer_id>/<product_id>/<n_recs>')
def get_predictions(customer_id, product_id, n_recs):
    print(f'processing input customer_id={customer_id}, product_id={product_id}')
    hashed_customer_id, hashed_product_id = get_hashed_ids_from_input(spark, customer_id, product_id)
    n_recs = int(n_recs)
    personalised_recs_bag = customer_recs.filter(customer_recs.customer_id_h == hashed_customer_id).collect()
    if len(personalised_recs_bag) == 0:
        return f'No predictions for customer_id={customer_id}'
    personalised_recs = personalised_recs_bag[0]['recommendations']
    product_ids_1 = []
    for i in range(n_recs):
        rec_hashed = personalised_recs[i]['product_id_h']
        product_ids_1.append(rec_hashed)
    print('Calculated personalised recommendations. Finding similar products now...')

    input_product_info = get_input_product_info(hashed_product_id, full_product_data)
    input_category_code, input_vec = get_input_points(input_product_info)
    product_data_lev = calculate_lev_distance_col(full_product_data, input_category_code)
    product_data_dist = calculate_euc_distance_col(product_data_lev, input_vec)
    pd_spark_data = unify_distances(product_data_dist)
    other_products = get_other_products(pd_spark_data, hashed_product_id)
    product_ids_2 = find_similar_n_products(other_products, n_recs)
    print('Calculated similar products. Collecting results now...')
    product_ids_1 = unhash_product_ids(product_ids_1, product_hash_table)
    product_ids_2 = unhash_product_ids(product_ids_2, product_hash_table)

    print('Finished process succesfully.')

    return f'Recommendations based on your previous purchases: {product_ids_1}.  \
         Similar products to the one you are browsing: {product_ids_2}'


if __name__ == '__main__':
    # Initialise spark and read data
    if FIRST_TIME == True:
        print('Make sure you run train_pipeline before')
        raise ValueError('Models not trained')
    print('Loading models...')
    spark = initialise_spark()
    df_product = read_table(spark, '../Data/product_.parquet')
    df_orders = read_table(spark, '../Data/orders_.parquet')
    df_product = hash_ids(df_product, ['product_id'])
    df_orders = hash_ids(df_orders, ['product_id', 'customer_id'])
    sales = get_only_sales(df_orders)
    product_list = get_product_list(df_product)

    customer_recs = read_table(spark, '../Data/trained_data/collab_filt_recs.parquet')
    full_product_data = read_table(spark, '../Data/trained_data/full_product_data.parquet')
    product_hash_table = read_table(spark, '../Data/trained_data/hashtable.parquet')
    print('Models loaded')

    app.run(port=5000)
