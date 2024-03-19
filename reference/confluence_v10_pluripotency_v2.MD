# Confluence V10 and Pluripotency V2: Development and Evaluation

This document details the development and evaluation processes for Confluence V10 and Pluripotency V2 models.

## Relevant Links

- [**Epic**](https://cellinobio.atlassian.net/browse/AIE-1705)
- [**Confluence Page**](https://cellinobio.atlassian.net/wiki/spaces/ML/pages/edit-v2/283574338)
- [**Deck**](https://docs.google.com/presentation/d/1BuOXoKxwzRK6SGUNp0E1uFEb8DLjmF8eMw7Q13sQ0A0/edit?usp=drive_link)
- [**Cell ID Ground Truth**](https://docs.google.com/spreadsheets/d/1EqEGWOXfiB1oR6FzuAdpJd4fpbQgpLcftNHL7VYfigo/edit#gid=0)
- [**Confluence Negative Patches**](https://docs.google.com/spreadsheets/d/1ERFB9gsq1Y3KVTFv_mDodFZmNTMOLBdwg2_Nn_67Oxc/edit#gid=392678033)
- [**Confluence Positive Patches**](https://docs.google.com/spreadsheets/d/1D1xM7YrmBOHgKxs6vRX9XjhHXNDjGvFsoMw0kO_4was/edit#gid=0)
- [**Test Set**](https://docs.google.com/spreadsheets/d/1wrSsTuLU33ipy2GwGxWoxAMi70xcaSJ7Yl_93GDDkEc/edit#gid=0)
    - The common test set for both confluence model and pluripotency model
- **Scripts**
    - swang_lib/data_stats.py
    - 4x_pluripotency_tf

## Confluency Model v10

### Train_val patchdataset artifact
- [manual_v8](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_data_preprocess/artifacts/conf_patchdataset_json/manual_v8/v0)
- [auto_CELL-001859:v0](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_data_preprocess/artifacts/conf_patchdataset_json/auto_CELL-001859/v0)
- [edge_train_val:v0](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_data_preprocess/artifacts/conf_patchdataset_json/edge_train_val/v0)
- [crystal_train_val:v0](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_data_preprocess/artifacts/conf_patchdataset_json/crystal_train_val/v0)
    

### Test patchdataset artifact
- [edge_test:v0](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_data_preprocess/artifacts/conf_patchdataset_json/edge_test/v0)
- [crystal_test:v0](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_data_preprocess/artifacts/conf_patchdataset_json/crystal_test/v0)


## Pluripotency Model v1

### Dataset Generation

- **Test set (Hard Coded)**
    - `TEST_DATA` in `pluritrainer_refactor_v1.py`

- **Workflow to Generate Two tfrecord Artifacts**
    1. **Create Plurimask:**
        - File: `Cell ID - Centralized Ground Truth Log - 4X Confluence Ground Truth.csv`
        - Script: [wandb:create_plurimask_from_tags.py](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_pluripotency/runs/lh4mmjha/code) or [github repo](https://github.com/cellino-biotech/wp_cell_id/tree/main/scripts), [new script](https://github.com/cellino-biotech/4x_pluripotency_tf/blob/main/src/aims_tools.py#L30C5-L30C5)
        - Output 1: `gt_pluri_4x_manual:v1(file_table.csv)`
        - Output 2: `/media/slee/DATA1/DL/data/pluripotency/Pluri_manual_mask (zarr files)` --> Manually uploaded to `cellino-ml-training/4x_pluri/pluri_manual_mask_ver0`

    
    2. **From Ground Truth to Edge TFRecord:**
        - Source: `gt_pluri_4x_manual:v1(file_table.csv)`
        - Script 1: `gen_patchdataset_edge_v1.py` 
        - Intermediate Output: `patch_dataset_ibidi_edge:v0`
        - Script 2: `gen_tfrecords_edge_v1.py`
        - Final Output: `TFRECORD-manual_pluri_ibidi_edge`
    
    3. **From Ground Truth to Standard TFRecord:**
        - Source: `gt_pluri_4x_manual:v1(file_table.csv)`
        - Script 1: `gen_patchdataset_v1.py` 
        - Intermediate Output: `patch_dataset_ibidi:v0`
        - Script 2: `gen_tfrecords_v1.py`
        - Final Output: `TFRECORD-manual_pluri_ibidi`

- **Using tfrecord Artifacts and TEST_DATA to Generate Train and Validation Sets**
    - Inputs:
        - `TFRECORD-manual_pluri_ibidi:v0, file_table.csv`
        - `TFRECORD-manual_pluri_ibidi_edge:v0, file_table.csv`
    - Script: Run `get_train_val_list.py` in `pluritrainer_refactor_v1.py`


### Model Training
Execute the following command to train the model
- `python pluritrainer_refactor_v1.py`
- Selected model for deployment: [stellar-brook-129](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_pluripotency/runs/kqqxbm2g)

### Evaluation
Execute the following command to evalue the model
- `evaluation_script2.py`


## Pluripotency Model v2

### Dataset Generation

- Main script: 4x_pluripotency_tf/src/dataset_generation.py

- **Manually Selected Test Set**
    - Test Well - C10V2 at https://docs.google.com/spreadsheets/d/1EqEGWOXfiB1oR6FzuAdpJd4fpbQgpLcftNHL7VYfigo/edit#gid=0

- **Create Plurimask:**
    Same as v1
    - File: `Cell ID - Centralized Ground Truth Log - 4X Confluence Ground Truth.csv`
    - Script: [wandb:create_plurimask_from_tags.py](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_pluripotency/runs/lh4mmjha/code) or [github repo](https://github.com/cellino-biotech/wp_cell_id/tree/main/scripts)
    - Output: `gt_pluri_4x_manual:v1(file_table.csv)`

- **From Ground Truth to Edge TFRecord:**
    Same as v1
    - Source: `gt_pluri_4x_manual:v1(file_table.csv)`
    - Script 1: `gen_patchdataset_edge_v2.py` 
    - Intermediate Output: `patch_dataset_ibidi_edge:v0`
    - Script 2: `gen_tfrecords_edge_v1.py`
    - Final Output: `TFRECORD-manual_pluri_ibidi_edge:v0`

- **From Ground Truth to Standard TFRecord:**
    Same as v1
    - Source: `gt_pluri_4x_manual:v1(file_table.csv)`
    - Script 1: `gen_patchdataset_v1.py` 
    - Intermediate Output: `patch_dataset_ibidi:v0`
    - Script 2: `gen_tfrecords_v1.py`
    - Final Output: `TFRECORD-manual_pluri_ibidi:v0`

### Training Pipeline

- [Go to Vertex AI/Workbench/JUPYTERLAB](https://console.cloud.google.com/vertex-ai/workbench/locations/us-central1-c/user-managed/swang-notebook?hl=en&project=project-ml-training-prod)
- source ~/.bashrc
- Clone [mlops](https://github.com/cellino-biotech/mlops)
- Run mlops$ python run_4x_pluripotency_pipeline_v2.py
- [Track pipeline](https://console.cloud.google.com/vertex-ai/pipelines/runs?project=project-ml-training-prod)




## Workflow to test the training

Base trainer in repo 1

confluence trainer in repo 2
pluri trainer in repo 3


In the situation base trainer is updated, how to check if it affects the confluence trainer on Vertex AI ?
Can we test it automatically?


Docker
packages installation
