# ml-kf-pipeline
A big challenge business faces is the deployment of machine learning models in production environments.
It requires dealing with a complex set of moving workloads through different pipelines. 
Once the machine learning models are developed, they need to be trained, deployed, monitored and kept track of.

In this repo it is show how to build and deploy a simple pipeline using Kubernetes, Kubeflow pipelines and seldon-core.

## Prerequisites
 1.  Create an AWS EKS cluster and a node group with at least 3 nodes of size `t3.xlarge`. 
   Use the instruction from https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html
 2.  Connect to the cluster
```bash
aws eks --region region update-kubeconfig --name clustername
```

 3.  Install [kubeflow-pipelines](https://github.com/kubeflow/pipelines)
```bash
git clone https://github.com/kubeflow/pipelines.git
cd pipelines/manifests/kustomize
kubectl apply -k cluster-scoped-resources/
kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s
kubectl apply -k env/platform-agnostic/
kubectl wait applications/pipeline -n kubeflow --for condition=Ready --timeout=1800s
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
and navigate to `localhost:8080` to verify installation.

 4. Install [helm](https://helm.sh/). Follow instructions https://helm.sh/docs/intro/install/
 
 5. Install [ambassador](https://www.getambassador.io/) in the cluster
```bash
kubectl create ns ambassador
helm repo add datawire https://www.getambassador.io
helm install ambassador datawire/ambassador \
  --set image.repository=quay.io/datawire/ambassador \
  --set enableAES=false \
  --set crds.keep=false \
  --namespace ambassador
```

 6. Install [seldon-core](https://docs.seldon.io/projects/seldon-core/en/latest/) in the cluster
```bash
kubectl create ns seldon
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set ambassador.enabled=true \
    --set crd.create=true \
    --namespace seldon
```

## Add secrets in the cluster
Make sure to create a secret named `aws-secret` containing `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` values
```bash
kubectl apply -f kubeflow-secret.yaml
```

## Upload the pipeline 
 1.  Build pipeline
```bash
python tokenize_pipeline.py  
```

 2.  Navigate to `localhost:8080`, upload `tokenize_pipeline.py.yaml` and trigger execution.

## Create SeldonDeployment
```bash
kubectl apply -f deployment.yaml
```

Enjoy!