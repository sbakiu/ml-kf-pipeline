REGISTRY=<ECR_REGISTRY>
TAG=1.0.0

function dockerbuildkaniko() {
  docker build -f Dockerfilekaniko -t $REGISTRY/kaniko-executor:$TAG .
}

function dockertagandpushkaniko() {
  IMAGE_NAME=$1
  echo "Pushing $REGISTRY/$IMAGE_NAME:$TAG"
  docker push $REGISTRY/$IMAGE_NAME:$TAG
}

IMAGENAME=tokenize
docker build . -t $REGISTRY/$IMAGENAME:$TAG
docker push $REGISTRY/$IMAGENAME:$TAG

# Build kaniko executor
dockerbuildkaniko
dockertagandpushkaniko kaniko-executor
