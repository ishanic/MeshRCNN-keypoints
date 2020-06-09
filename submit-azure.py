"""ischakra: The script to submit a PyTorch training to Azure ML
"""

import os
import webbrowser

from azureml.core import Experiment
from azureml.core import Workspace, Environment, ContainerRegistry
from azureml.train.dnn import PyTorch


SUBSCRIPTION_ID = 'b8da985a-830d-4d20-b9e5-8d4c0d798c7f'
RESOURCE_GROUP = 'mr-rg'
#WORKSPACE_NAME = 'mr-sandbox1-we-ws'
#CLUSTER_NAME = 'k80x4'
WORKSPACE_NAME = 'mr-shared1-we-ws'
CLUSTER_NAME = 'p100x4'

from azureml.core import Datastore
from azureml.data.data_reference import DataReference

def main():
    workspace = Workspace(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
    compute_target = workspace.compute_targets[CLUSTER_NAME]
    input_datastore = Datastore.get(workspace, 'ischakra_pubdatasets')
    data_dir = DataReference(datastore=input_datastore,
    path_on_datastore="")
    input_modelstore = Datastore.get(workspace, 'ischakra_models')
    model_dir = DataReference(datastore=input_modelstore,
    path_on_datastore="pix3d-meshrcnn/baseline/")    
    user_alias = os.getlogin()
    experiment_name = user_alias + '-dummy0'
    experiment = Experiment(workspace, experiment_name)
    script_name = 'tools/train_net.py'
    #script_params = {
    #    '--num-gpus': 4,
    #    '--config-file': 'configs/pix3d/meshrcnn_R50_FPN.yaml',
    #    '': data_dir,
    #    #'--model_dir': './outputs',
    #}
    script_params = ['--num-gpus', 4, '--num-machines', 1,
                     '--config-file', 'configs/pix3d/meshrcnn_R50_FPN.yaml', 
                     "DATASETS.DATADIR", data_dir, "OUTPUT_DIR", model_dir]
                     #"DATASETS.DATADIR", data_dir]

    container_registry = ContainerRegistry()
    container_registry.address = "ischakra.azurecr.io"
    container_registry.username = "ischakra"
    container_registry.password = "vW=2AX99Rfjx0mk6hvkrBWGHpNYAOhBW"

    #from azureml.core.runconfig import MpiConfiguration
    #distributed_training = MpiConfiguration()
    #distributed_training.process_count_per_node = 4

    estimator = PyTorch(source_directory='.',
                        entry_script=script_name,
                        #script_params=script_params,
    			        compute_target=compute_target,
                        #framework_version='1.3',
                        use_gpu=True,
                        image_registry_details=container_registry,
                        custom_docker_image="fair3d:1.1",
                        user_managed=True,
                        node_count=1,
                        #distributed_training=distributed_training,
                        #pip_packages=["torch","torchvision"],
                        )

    
    from azureml.pipeline.steps import EstimatorStep
    est_step = EstimatorStep(name="Estimator_Train", 
                        estimator=estimator, 
                        estimator_entry_script_arguments=script_params,
                        inputs=[data_dir, model_dir],
                        #outputs=["./outputs"],
                        compute_target=compute_target,
                        allow_reuse=True)
    
    from azureml.pipeline.core import Pipeline
    pipeline = Pipeline(workspace=workspace, steps=[est_step])

    run = experiment.submit(pipeline, regenerate_outputs=True)
    #run = experiment.submit(estimator)
    print(run.get_portal_url())


if __name__ == '__main__':
    main()
