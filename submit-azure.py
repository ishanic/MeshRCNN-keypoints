"""ischakra: The script to submit a PyTorch training to Azure ML
"""

import os
import webbrowser

from azureml.core import Experiment
from azureml.core import Workspace, Environment, ContainerRegistry
from azureml.train.dnn import PyTorch

import argparse

parser = argparse.ArgumentParser(description='Launch job in Azure')
parser.add_argument("--subscription-id",
                        required=True,
                        help="32 hexadecimal GUID")
parser.add_argument("--resource-group",
                        required=True,
                        help="RESOURCE_GROUP")
parser.add_argument("--workspace-name",
                        required=True,
                        help="Workspace Name")
parser.add_argument("--cluster-name",
                        required=True,
                        help="Cluster Name")
parser.add_argument("--dataset_directory",
                        required=True,
                        help="Path to data directory inside datastore")
parser.add_argument("--model_directory",
                        required=True,
                        help="Path to model directory inside modelstore")
parser.add_argument("--custom_docker_image",
                        required=True,
                        help="Custom docker image")
parser.add_argument("--container_registry_address",
                        required=True,
                        help="Docker container registry address")
parser.add_argument("--username",
                        required=True,
                        help="Docker container registry username")
parser.add_argument("--password",
                        required=True,
                        help="Docker container registry password")


args = parser.parse_args()

SUBSCRIPTION_ID = args.subscription_id
RESOURCE_GROUP = args.resource_group
WORKSPACE_NAME = args.workspace_name
CLUSTER_NAME = args.cluster_name

from azureml.core import Datastore
from azureml.data.data_reference import DataReference

def main():
    workspace = Workspace(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
    compute_target = workspace.compute_targets[CLUSTER_NAME]
    input_datastore = Datastore.get(workspace, args.dataset_directory)
    data_dir = DataReference(datastore=input_datastore, path_on_datastore="")
    input_modelstore = Datastore.get(workspace, args.model_directory)
    model_dir = DataReference(datastore=input_modelstore,
    path_on_datastore="pix3d-meshrcnn/baseline/")    
    user_alias = os.getlogin()
    experiment_name = user_alias + '-dummy0'
    experiment = Experiment(workspace, experiment_name)
    script_name = 'tools/train_net.py'
    
    script_params = ['--num-gpus', 4, '--num-machines', 1,
                     '--config-file', 'configs/pix3d/meshrcnn_R50_FPN.yaml', 
                     "DATASETS.DATADIR", data_dir, "OUTPUT_DIR", model_dir]
                     #"DATASETS.DATADIR", data_dir]

    container_registry = ContainerRegistry()
    container_registry.address = args.container_registry_address
    container_registry.username = args.username
    container_registry.password = args.password

    
    estimator = PyTorch(source_directory='.',
                        entry_script=script_name,
    			        compute_target=compute_target,
                        use_gpu=True,
                        image_registry_details=container_registry,
                        custom_docker_image=args.custom_docker_image,
                        user_managed=True,
                        node_count=1,
                        )

    
    from azureml.pipeline.steps import EstimatorStep
    est_step = EstimatorStep(name="Estimator_Train", 
                        estimator=estimator, 
                        estimator_entry_script_arguments=script_params,
                        inputs=[data_dir, model_dir],
                        compute_target=compute_target,
                        allow_reuse=True)
    
    from azureml.pipeline.core import Pipeline
    pipeline = Pipeline(workspace=workspace, steps=[est_step])

    run = experiment.submit(pipeline, regenerate_outputs=True)
    #run = experiment.submit(estimator)
    print(run.get_portal_url())


if __name__ == '__main__':
    main()
