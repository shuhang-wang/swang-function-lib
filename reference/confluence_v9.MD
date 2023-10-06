
# Training Confluence Workflow

## Patch Selection
- Command
```
    python 4x_confluence_24_well_edge_fine_tune/patch_selection_v9.py
```

- [wandb run](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/28chso8d/overview?workspace=user-swang)
    - 4x_conf_retrain_training_patchdataset:v15
    - 4x_conf_retrain_validation_patchdataset:v16
    - 4x_conf_retrain_test_edges_patchdataset:v3
    - 4x_conf_retrain_test_crystals_patchdataset:v3

## Serialization

- Command

```
    python 4x_confluence_24_well_edge_fine_tune/patch_serializer_v9.py
```

- Generate tfrecords locally
    - ~/Documents/Dataset/patch_selection_v9/tfrecord

## Upload tfrecords

- Command
```
    python 4x_confluence_24_well_edge_fine_tune/file_upload.py
```

- [wandb run](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/374f1hm2)
    - 4x_conf_retrain_training:v39
    - 4x_conf_retrain_validation:v31


<!-- ## Lydia's latest run:
- W&B:
    - [WandB Link](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/3g7045ub/overview?workspace=user-swang)
- Command:
```
/4x_confluence_24_well_edge_fine_tune/run_training.py --wandb-project 4x_conf_retrain --wandb-run-id 3g7045ub --wandb-dataset True --input-shape [256,256,4] --label-shape [256,256,1] --data-input-shape [4,256,256] --data-label-shape [256,256,1] --train-dir 4x_conf_retrain_training:latest --val-dir 4x_conf_retrain_validation:latest --epochs 300 --batch-size 128 --val-batch-size 512 --train-steps 129 --learning-rate 1e-5 --learning-rate-params {} --optimizer-type adam --loss-type bce --loss-params {"alpha":0.55,"gamma":2.0} --normalization-type batch_norm --random-seed 42 --model-path None --serialized-channels-first --serialized-channels-first
``` -->


## Training Pipeline

- Clone [mlops](https://github.com/cellino-biotech/mlops)

- Install requirement.in

- Set wandb Key

```
    export WANDB_KEY=local-1d74cac7c473289d563ef7902e8b78f0109d7015
```
- Set Google Credentials

 ```
    export GOOGLE_APPLICATION_CREDENTIALS=/home/jupyter/.config/gcloud/application_default_credentials.json
 ```

- Command

    - Quick test  
    
    ```
    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 20 --batch-size 64 --val-batch-size 128 --train-steps 2 --learning-rate 1e-27 --optimizer-type adam --loss-type focal_loss --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.65,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc
    ```


    - Real train  

    ```
    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 1e-5 --optimizer-type adam --loss-type focal_loss --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.65,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc

    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 5e-5 --optimizer-type adam --loss-type focal_loss --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.65,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc

    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 1e-5 --optimizer-type adam --loss-type bce --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.65,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc

    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 5e-5 --optimizer-type adam --loss-type bce --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.65,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc


    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 1e-5 --optimizer-type adam --loss-type focal_loss --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.55,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc

    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 5e-5 --optimizer-type adam --loss-type focal_loss --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.55,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc

    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 1e-5 --optimizer-type adam --loss-type bce --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.55,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc

    python3 run_4x_conf_retrain_pipeline.py --wandb-project 4x_conf_retrain --data-input-shape \[4,256,256\] --data-label-shape \[256,256,1\] --input-shape \[256,256,4\] --label-shape \[256,256,1\] --save-model-wandb-cb True --epochs 300 --batch-size 64 --val-batch-size 128 --train-steps 337 --learning-rate 5e-5 --optimizer-type adam --loss-type bce --normalization-type batch_norm --serialized-channels-first --model-path None --train-dir 4x_conf_retrain_training:v39 --val-dir 4x_conf_retrain_validation:v31 --loss-params \{\"alpha\":0.55,\"gamma\":2.0\} --wandb-dataset True --random-seed 42 --learning-rate-params {} --train-machine-type n1-highmem-16 --container-image us-central1-docker.pkg.dev/project-ci-cd/ml-modelling/ml-confluence-patcher:51baca44765226716addc2a68359ecbfadb3fdfc
    ```

    - Changeable parameters  
    
        --loss-params \{\"alpha\":0.55,\"gamma\":2.0\} # often change alpha  
        --learning-rate 1e-5  
        --loss-type focal_loss  
        --loss-type bce  
        --train-machine-type n1-highmem-16 #n1-highmem-8  
    

- Track pipeline
    - [link](https://console.cloud.google.com/vertex-ai/pipelines/runs?project=cellino-plate-db)



- Discussion
    - somewhere in the pipeline is a memory leak and so we've had to downgrade the batch sizes, or up the vm memory to account for it (added to the list of "things to improve about the pipeline) - i pulled the validation batch size down to 128 (it's more of a time saver to have it larger) and pushed the train-machine-type up to a n1-highmem-16 and pulled the batch size down to 64 and that worked - this is the run that succeeded: https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/3tr7qyz4/overview?workspace=user-lskrabonja-cellinobiome


- [wandb Runs](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/2pg6ukmu)
    - confluence_v9_v2-733 - confluence_v9_v2-740
- Selected challenger
    - sunny-waterfall-740:latest: this one is the version from the last epoch
    - model-waterfall-740:latest: this one is the version from the epoch with the lowest loss 

## Evaluation

- Command
```
    python eval_v9.py
```

- wandb runs:
    - [link](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/2qihbp7w)
        - Qualitative Table
        - Evaluation Table
        - 4x_conf_retrain_test_edges_patchdataset:v3
        - 4x_conf_retrain_test_crystals_patchdataset:v3
    - [link](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/1cqjei8c)
        - Differentiation Recall
    
## Reports

- [wandb report](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/reports/4x-Confluence-V9-Challenger-Report--VmlldzoxNjk?accessToken=0qxrjrvsc9cpv3tb9njh9msh0t222lb3rqgj2j8rjllvxlhrpobl3a0yc81c01z4)
- [Deck](https://docs.google.com/presentation/d/1_5tFtDnx_C2OJij3ZDbG1wFYoNYiwho-98Qw1ERSRwE/edit?usp=sharing)
