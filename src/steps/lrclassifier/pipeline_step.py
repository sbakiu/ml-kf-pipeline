import click
import logging

from src.steps.lrclassifier.LRClassifier import LRClassifier
from src.steps.util.Config import Config
from src.steps.util.Constants import Constants
from src.steps.util.ModelUtil import ModelUtil

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--labels-data', default="labels.data")
@click.option('--vectors-data', default="vectors.data")
def run_pipeline(
        labels_data,
        vectors_data
):

    classifier = LRClassifier(
        action="train",
        labels_data=labels_data,
        vectors_data=vectors_data
    )

    classifier.fit()

    location = ModelUtil.upload_model_with_uuid(
        file_name=Constants.LR_MODEL_FILENAME,
        model=classifier.lr_model,
        bucket_name=Config.bucket_name,
        model_location_file=Constants.KF_MODEL_OUT_LR_MODEL
    )

    execution = ModelUtil.create_execution()
    ModelUtil.log_model(execution=execution, model_uri=location, model_name="LRClassifier")


if __name__ == "__main__":
    run_pipeline()
