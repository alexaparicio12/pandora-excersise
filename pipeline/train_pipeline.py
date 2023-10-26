from model.train_collaborative_filtering import make_train_test_split, initialise_model, train_model, evaluate_model, \
    save_customer_recs
from preprocessing.distance_matching_processing import create_product_features, create_category_string, \
    process_numerical_features, join_to_full_product_data, save_product_data, create_theme_col
from preprocessing.process_data import save_product_hash, process_purchases, read_table, hash_ids, get_only_sales, \
    get_product_list
from setup.setup_spark import initialise_spark

FIRST_TIME = True

if __name__ == '__main__':
    print('Setting up models...')

    spark = initialise_spark()
    df_product = read_table(spark, '../Data/product_.parquet')
    df_orders = read_table(spark, '../Data/orders_.parquet')
    df_product = hash_ids(df_product, ['product_id'])
    df_orders = hash_ids(df_orders, ['product_id', 'customer_id'])
    sales = get_only_sales(df_orders)
    product_list = get_product_list(df_product)
    print('saving_product_hash')
    product_hash_table = save_product_hash(df_product, path='../Data/trained_data/hashtable.parquet')

    # Train collaborative filtering
    collected_buys = process_purchases(sales, sales_outlier_cap=6, ignore_amount=True)
    train, test = make_train_test_split(collected_buys)
    collaborative_filtering_model = initialise_model()
    trained_model = train_model(collaborative_filtering_model, train)
    mae = evaluate_model(trained_model, test)
    print(f'Collaborative filtering trained. Testing MAE: {mae}')

    customer_recs = save_customer_recs(trained_model, path='../Data/trained_data/collab_filt_recs.parquet',
                                       n_max_recs=10)

    # Set up distances
    print('Calculating distance based matching')
    product_data = create_product_features(sales, df_product)
    product_data = create_theme_col(product_data)
    product_data = create_category_string(product_data)
    numeric_data_vec = process_numerical_features(product_data)
    full_product_data = join_to_full_product_data(numeric_data_vec, product_data)
    full_product_data = save_product_data(full_product_data, path='../Data/trained_data/full_product_data.parquet')
    print('API setup complete and models saved.')
    FIRST_TIME = False
