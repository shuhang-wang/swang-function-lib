# Confluence V10 and Pluripotency V2: Development and Evaluation

This document details the development and evaluation processes for Confluence V10 and Pluripotency V2 models.

## Relevant Links

- [**Epic**](https://cellinobio.atlassian.net/browse/AIE-1705)
- [**Confluence Page** ](https://cellinobio.atlassian.net/wiki/spaces/ML/pages/edit-v2/283574338)


## Pluripotency Model

### Dataset Generation
Use the following script to generate the patch dataset:
- `gen_patchdataset_v1.py`

### Model Training
Execute the following command to train the model:
- `python pluritrainer_refactor_v1.py`