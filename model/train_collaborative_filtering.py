from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.ml import Model


def make_train_test_split(collected_buys: DataFrame, proportion: float = 0.8) -> (DataFrame, DataFrame):
    """
    :param collected_buys: DataFrame detailing how many times a customer has bought a specific product
    :param proportion: Proportion of data used for training
    :return: A random sample of training and testing data
    """
    training, test = collected_buys.randomSplit([proportion, 1 - proportion], seed=42)
    return training, test


def initialise_model() -> Model:
    """
    Initialises the model. Parameters have been optimised through experimentation.
    """
    als_model = ALS(maxIter=10,
                    regParam=0.01,
                    implicitPrefs=False,
                    rank=20,
                    userCol="customer_id_h",
                    itemCol="product_id_h",
                    ratingCol="quantity",
                    coldStartStrategy="drop")
    return als_model


def train_model(als_model: Model, training: DataFrame) -> Model:
    """
    Trains model using the training data.
    """
    return als_model.fit(training)


def evaluate_model(trained_model: Model, test: DataFrame) -> float:
    """
    Evaluates the Mean Absolute Error of the model. This metric seemed more appropriate as the "scores" are either 0 or 1.
    """
    predictions = trained_model.transform(test)
    evaluator = RegressionEvaluator(metricName="mae", labelCol="quantity", predictionCol="prediction")
    mae = evaluator.evaluate(predictions)
    print("Mean absolute error = " + str(mae))
    return mae


def save_customer_recs(trained_model: Model, path: str = 'Data/trained_data/collab_filt_recs.parquet',
                       n_max_recs: int = 10) -> DataFrame:
    """
    Calculates the product recommendations for all customers and saves them into a table. This enables a quick
    search when calling the API and avoids any further calculations. This function takes slightly under 1 minute to run.
    """
    customer_recs = trained_model.recommendForAllUsers(n_max_recs)
    customer_recs.write.mode('overwrite').parquet(path)
    return customer_recs
