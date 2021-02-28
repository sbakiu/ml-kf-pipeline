import kfp.dsl as dsl
import kfp.aws as aws
import logging
import kfp.compiler as compiler

logging.basicConfig(level=logging.INFO)

TAG = "1.0.0"
REGISTRY = "<ID>.dkr.ecr.eu-central-1.amazonaws.com/<registry>"


@dsl.pipeline(
    name='Pipeline',
    description='A pipeline for training with Reddit response dataset'
)
def kf_pipeline(
        input_data='reddit_train.csv',
):
    """
    Pipeline
    """

    tokenize_training_step = dsl.ContainerOp(
        name='tokenize',
        image=f"{REGISTRY}/tokenize:{TAG}",
        command="python",
        arguments=[
            "-m",
            "src.steps.tokenize.pipeline_step"
        ],
        file_outputs={"tokenize_location": "/tokenized_location.txt",
                      "labels_location": "/labels_location.txt"},
        pvolumes={}
    ).apply(aws.use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')) \
        .set_image_pull_policy('Always')

    vectorize_training_step = dsl.ContainerOp(
        name='vectorize',
        image=f"{REGISTRY}/tokenize:{TAG}",
        command="python",
        arguments=[
            "-m",
            "src.steps.tfidftransformer.pipeline_step",
            "--input-data", tokenize_training_step.outputs['tokenize_location']
        ],
        file_outputs={"tfidftransformer_location": "/vectorizer_location.txt",
                      "tfidfvectors_location": "/vectors_location.txt"},
        pvolumes={}
    ).apply(aws.use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')) \
        .set_image_pull_policy('Always')

    lr_training_step = dsl.ContainerOp(
        name='logical regression',
        image=f"{REGISTRY}/tokenize:{TAG}",
        command="python",
        arguments=[
            "-m",
            "src.steps.lrclassifier.pipeline_step",
            "--labels-data", tokenize_training_step.outputs['labels_location'],
            "--vectors-data", vectorize_training_step.outputs['tfidfvectors_location']
        ],
        file_outputs={"lr_model_location": "/lr_model_location.txt"},
        pvolumes={}
    ).apply(aws.use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')) \
        .set_image_pull_policy('Always')

    tokenize_build_step = dsl.ContainerOp(
        name='Build tokenize Serving',
        image=f"{REGISTRY}/kaniko-executor:{TAG}",
        arguments=[
            "--dockerfile=Dockerfile",
            f"--build-arg=TOKENIZE_MODEL={tokenize_training_step.outputs['tokenize_location']}",
            "--context=dir:///workspace",
            f"--destination={REGISTRY}/tokenizeserving:{TAG}"
        ],
        pvolumes={}
    ).apply(aws.use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')) \
        .set_image_pull_policy('Always') \
        .after(lr_training_step)

    vectorize_build_step = dsl.ContainerOp(
        name='Build vectorize Serving',
        image=f"{REGISTRY}/kaniko-executor:{TAG}",
        arguments=[
            "--dockerfile=Dockerfile",
            f"--build-arg=VECTORIZE_MODEL={vectorize_training_step.outputs['tfidftransformer_location']}",
            "--context=dir:///workspace",
            f"--destination={REGISTRY}/vecotrizeserving:{TAG}"
        ],
        pvolumes={}
    ).apply(aws.use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')) \
        .set_image_pull_policy('Always') \
        .after(tokenize_build_step)

    lr_model_build_step = dsl.ContainerOp(
        name='Build LR Serving',
        image=f"{REGISTRY}/kaniko-executor:{TAG}",
        arguments=[
            "--dockerfile=Dockerfile",
            f"--build-arg=LR_MODEL={lr_training_step.outputs['lr_model_location']}",
            "--context=dir:///workspace",
            f"--destination={REGISTRY}/lrserving:{TAG}"
        ],
        pvolumes={}
    ).apply(aws.use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')) \
        .set_image_pull_policy('Always') \
        .after(vectorize_build_step)


if __name__ == '__main__':

    pipeline_file_path = __file__ + '.yaml'
    compiler.Compiler().compile(kf_pipeline, pipeline_file_path)

