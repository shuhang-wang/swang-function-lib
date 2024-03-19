# Fast Fine-Tune

## Patch Selection
- Command
```
    python 4x_confluence_24_well_edge_fine_tune/patch_selection_v10.py
```
- Output wandb artifacts
    - 4x_conf_retrain_validation_patchdataset:v17
    - 4x_conf_retrain_training_patchdataset:v16

## Serialization

- Command

```
    python 4x_confluence_24_well_edge_fine_tune/patch_serializer_v10.py
```

## Upload tfrecords

- Command
```
    python 4x_confluence_24_well_edge_fine_tune/file_upload.py
```

- Output wandb artifacts
    - 4x_conf_retrain_training:v40
    - 4x_conf_retrain_validation:v32


## Training Pipeline
- [Go to Vertex AI/Workbench/JUPYTERLAB](https://console.cloud.google.com/vertex-ai/workbench/locations/us-central1-c/user-managed/swang-notebook?hl=en&project=project-ml-training-prod)
- Activate virtual env
```
    jupyter@swang-notebook:~$ source ~/.bashrc
    (base) jupyter@swang-notebook:~$ pyenv activate conf or pyenv activate conf_base
    (conf) (base) jupyter@swang-notebook:~$ conda deactivate
```

- Fine-tune (model-path)
    - Command
        ```
        python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 20 --batch-size 64 --val-batch-size 128 --train-steps 100 --learning-rate 1e-30 --optimizer-type adam --loss-type focal_loss --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v40 --val-dir 4x_conf_retrain_validation:v32 --loss-params \{\"alpha\":0.65,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --model-path model-magic-shadow-537:latest --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc
        ```
    - Reference command
        
        https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/2rwi1net/overview?workspace=user-lskrabonja-cellinobio

    - Track pipeline
        - [link](https://console.cloud.google.com/vertex-ai/pipelines/runs?project=cellino-plate-db)

    - Output wandb artifacts
        - from 'model-kind-wave-816' to 'model-eager-aardvark-821'

# Retrain

## Patch Selection
- Command
```
    python 4x_confluence_24_well_edge_fine_tune/patch_selection_v10.py
```
- Output wandb artifacts
    - 4x_conf_retrain_validation_patchdataset:v18
    - 4x_conf_retrain_training_patchdataset:v17

## Serialization

- Command

```
    python 4x_confluence_24_well_edge_fine_tune/patch_serializer_v10.py
```

## Upload tfrecords

- Command
```
    python 4x_confluence_24_well_edge_fine_tune/file_upload.py
```

- Output wandb artifacts
    - 4x_conf_retrain_training:v41
    - 4x_conf_retrain_validation:v33

## Training Pipeline
- Command

```
    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 263 --learning-rate 1e-5 --optimizer-type adam --loss-type focal_loss --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v41 --val-dir 4x_conf_retrain_validation:v33 --loss-params \{\"alpha\":0.65,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc

```
- [Go to Vertex AI/Workbench/JUPYTERLAB](https://console.cloud.google.com/vertex-ai/workbench/locations/us-central1-c/user-managed/swang-notebook?hl=en&project=project-ml-training-prod)
- Activate virtual env
```
    jupyter@swang-notebook:~$ source ~/.bashrc
    (base) jupyter@swang-notebook:~$ pyenv activate conf
    (conf) (base) jupyter@swang-notebook:~$ conda deactivate
```

- Track pipeline
    - [link](https://console.cloud.google.com/vertex-ai/pipelines/runs?project=cellino-plate-db)

- Output wandb artifacts
    - from 'model-super-field-826' to 'model-rose-cherry-830'


## Report


## Submission
- model files
   https://console.cloud.google.com/storage/browser/cellino-ml-models/confluence/model-super-field-826_v78?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=0&project=project-ml-storage&prefix=&forceOnObjectsSortingFiltering=false

- eval script


- wand report
    https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/reports/4x-Confluence-V10-Challenger-Report--VmlldzoyMjM

- model development procedure
    https://cellinobio.atlassian.net/wiki/spaces/CELLINOBIO/pages/195133451/Model+Deployment+Procedure