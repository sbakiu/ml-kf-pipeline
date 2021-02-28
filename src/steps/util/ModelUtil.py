import logging
from datetime import datetime

import dill
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3

from botocore.exceptions import ClientError
from kubeflow.metadata import metadata


class ModelUtil(object):
    METADATA_STORE_HOST = "metadata-grpc-service.kubeflow"
    # default DNS of Kubeflow Metadata gRPC serivce.
    METADATA_STORE_PORT = 8080

    @staticmethod
    def download_model(model_location_file, model_destionation):
        logging.info(f"Reading model location file: {model_location_file}")
        with open(model_location_file, "r") as in_f:
            model_location = in_f.read()
        logging.info(f"Model is located in: {model_location}")
        model_location = model_location.strip()

        logging.info(f"Downloading model {model_location} to {model_destionation}.")
        ModelUtil.download_to_location(model_location, model_destionation)

    @staticmethod
    def download_to_location(
            source_file="",
            destination_file_name=""
    ):
        """
        Downloads a blob from the bucket.
        :param source_file: bucket name
        :param destination_file_name: store to file locally
        :return:
        """
        o = urlparse(source_file, allow_fragments=False)

        bucket_name = o.netloc
        filename = o.path.lstrip('/')
        destination_path = Path(destination_file_name)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"bucket_name: {bucket_name}")
        logging.info(f"filename: {filename}")
        logging.info(f"destination_path: {destination_path}")

        logging.info("Downloading from S3 Bucket")

        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, filename, destination_file_name)

        logging.info(f"Blob {filename} downloaded to {destination_file_name}.")

    @staticmethod
    def upload_model_with_uuid(file_name, model, bucket_name, model_location_file):
        uuid_str = str(uuid.uuid1())
        model_path_with_uuid = f"{uuid_str}_{file_name}"

        logging.info(f"Dumping model to {model_path_with_uuid}")
        with open(model_path_with_uuid, "wb") as out_f:
            dill.dump(model, out_f)

        logging.info(f"Uploading {model_path_with_uuid} to S3")
        model_location_in_bucket = ModelUtil.upload_blob(
            source_file_name=model_path_with_uuid,
            destination_bucket_name=bucket_name,
            destination_filename=model_path_with_uuid
        )
        with open(model_location_file, "w") as out_f:
            out_f.write(model_location_in_bucket)

        return model_location_in_bucket

    @staticmethod
    def upload_blob(
            source_file_name="",
            destination_bucket_name="",
            destination_filename=""
    ):
        """
        Uploads a blob to the bucket.
        :param source_file_name: source file name
        :param destination_bucket_name: bucket name
        :param destination_filename: destination file name
        :return:
        """
        logging.info("Uploading to S3.")

        s3_client = boto3.client('s3')

        try:
            s3_client.upload_file(source_file_name, destination_bucket_name, destination_filename)
        except ClientError as e:
            logging.error(e)

        logging.info(f"Blob {source_file_name} uploaded to {destination_bucket_name} / {destination_filename}")

        file_path = f"s3://{destination_bucket_name}/{destination_filename}"
        return file_path

    @staticmethod
    def log_model(execution, model_name, model_uri):
        """
        Log to Kubeflow artifacts the model
        :param execution:
        :param project_name:
        :param project_version:
        :param model_name:
        :param model_version:
        :param storage_bucket:
        :param file_name:
        :return:
        """

        model = metadata.Model(
            name=model_name,
            uri=model_uri,
            version="1.0.0"
        )

        execution.log_output(model)
        return model

    @staticmethod
    def create_execution():
        """
        Prepare an execution for artifacts storage
        :return:
        """
        metadata_store = metadata.Store(
            grpc_host=ModelUtil.METADATA_STORE_HOST,
            grpc_port=ModelUtil.METADATA_STORE_PORT
        )

        workspace = metadata.Workspace(
            # Connect to metadata service in namespace kubeflow in k8s cluster.
            store=metadata_store,
            name="workspace_1",
            description="a workspace for testing",
            labels={"n1": "v1"}
        )

        run = metadata.Run(
            workspace=workspace,
            name="run-" + datetime.utcnow().isoformat("T"),
            description="a run in ws_1",
        )

        execution = metadata.Execution(
            name="execution" + datetime.utcnow().isoformat("T"),
            workspace=workspace,
            run=run,
            description="execution example",
        )
        print("An execution was created with id %s" % execution.id)
        return execution
