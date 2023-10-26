This model implements two different recommender engines, a proximity based matching and a collaborative filtering.

To run this model API:
1. Please make sure that the orders_.parquet and product_.parquet files are saved in a folder called Data at the repo root level
2. Install requirements from the requirements.txt file
3. Run train_pipeline.py
4. Run run_model_api.py
5. Access your local port 5000 through your browser, for example: http://127.0.0.1:5000/
6. Input a valid customer id, product id and number of desired product recommendations: http://127.0.0.1:5000/<customer_id>/<product_id>/<n_recommendations>
