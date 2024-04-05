# Permission Enabling Scripts to Access Google Cloud
```
gcloud auth application-default login
gcloud config set project project-aims-prod
```
or 
```
export GCP_KEY_PATH=/home/shuhangwang/Documents/Keys/training-ml-worker-key.json

export GCP_KEY_PATH=/home/jupyter/Documents/Keys/training-ml-worker-key.json

```

- GOOGLE_APPLICATION_CREDENTIALS is for personal account
- GCP_KEY_PATH is for system account
<!-- export GOOGLE_APPLICATION_CREDENTIALS=/home/shuhangwang/.config/gcloud/application_default_credentials.json -->
<!-- service account -->

<!-- export GOOGLE_APPLICATION_CREDENTIALS=/home/shuhangwang/Documents/Keys/training-ml-worker-key.json

export GOOGLE_APPLICATION_CREDENTIALS=/home/jupyter/Documents/Keys/training-ml-worker-key.json -->

# Resources
- Files needed to run patch_selection_v8.py:
    - [Google Drive Link](https://drive.google.com/drive/folders/1ESoFxlmQYtxSVx340L_l9fVV9DepVOXA)
- Ground truth sheet
    - https://docs.google.com/spreadsheets/d/1EqEGWOXfiB1oR6FzuAdpJd4fpbQgpLcftNHL7VYfigo/edit#gid=0



# Virtual Environment Setup on M2 Mac (Apple Silicon)
https://cellino-biotech.github.io/python/

- Create environment using conda
- Run the following commands to give permission for some packages in the requirement.txt:
```
gcloud auth application-default login  
python -m pip install -U pip keyrings.google-artifactregistry-auth  
pip install ml-util --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple  
```
- install packages by requirement.txt:
```
pip install -r requirement.in
```
or
```
pip install -r requirement.txt
```

```
pip install cellino-tiler==0.4.3 --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple
```

- Third, install Tensorflow on M1 Mac:
    - [Apple Developer Link](https://developer.apple.com/metal/tensorflow-plugin/)
- Fourth, install tensorflow-io instead of tensorflow-io-gcs-filesystem==0.31.0:
    - [StackOverflow Link](https://stackoverflow.com/questions/70277737/cant-install-tensorflow-io-on-m1)
- Fifth, install cellino_tiler:
    - [GitHub Repository](https://github.com/cellino-biotech/tiler)
```
pip install keyrings.google-artifactregistry-auth
pip install --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple cellino_tiler
pip install git+https://github.com/cellino-biotech/tiler.git
```

- Sove problems related to wandb
```
conda install -c conda-forge wandb
```
- Solve possible path problems
```
export DYLD_LIBRARY_PATH=/Users/shuhangwang/anaconda3/envs/conf_mac2/lib/python3.9/site-packages/tensorflow/:$DYLD_LIBRARY_PATH
```
- Train to test tensorflow
```
GOOGLE_APPLICATION_CREDENTIALS=/Users/shuhangwang/Documents/Keys/training-ml-worker-key.json python train_attempt.py     
```



# Emulating Intel Python on Apple Silicon
Rosetta 2 can be used as the tool to build the emulation.
```
arch -x86_64 /usr/bin/python3 -m venv iconfluence  
arch -x86_64 /bin/zsh  
source iconfluence/bin/activate  
```

# Install Private Packages

```
gcloud auth application-default login
python -m pip install -U pip keyrings.google-artifactregistry-auth
pip install ml-util --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple

pip install git+https://TOKEN@github.com/USERNAME/REPO_NAME.git@BRANCH_NAME

pip install git+https://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@github.com/cellino-biotech/ml_util.git@swang/serializer_related
pip install git+https://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@github.com/cellino-biotech/zarr_tf_util.git@swang/zarr_dim

zarr_tf_util/tree/swang/zarr_dim

```

# CUDA Problem on Lambda


- Run these commands
```
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
ln -s /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc ./libdevice.10.bc

```

# Wandb Installation

```
wandb login --relogin --host=https://cellinobio.wandb.io
```


# Vertex AI Setup
https://drive.google.com/file/d/1mG_cH2ThEcMtnmoGUhG17f9jYwDg9V8h/view


# Zarr

- zarr dimension
time, channel, z, y, x

# Miscellaneous

- Cellino Query: https://insights.cellinobio.app/superset/sqllab

- Check if the new release registry works: https://console.cloud.google.com/artifacts/python/project-ci-cd/us-central1/stargaze/ml-util?project=project-ci-cd

<!-- # Permission Enabling Scripts to Access Google Cloud
```
gcloud auth application-default login
gcloud config set project project-aims-prod
```
or 
```
export GOOGLE_APPLICATION_CREDENTIALS=/home/shuhangwang/Documents/Keys/training-ml-worker-key.json
```


# Data
* Files needed to run patch_selection_v8.py
[Google Drive Link](https://drive.google.com/drive/folders/1ESoFxlmQYtxSVx340L_l9fVV9DepVOXA)

# Training
## Lydia's latest run:
* W&B:
[WandB Link](https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/3g7045ub/overview?workspace=user-swang)

* Command:
```
/4x_confluence_24_well_edge_fine_tune/run_training.py --wandb-project 4x_conf_retrain --wandb-run-id 3g7045ub --wandb-dataset True --input-shape [256,256,4] --label-shape [256,256,1] --data-input-shape [4,256,256] --data-label-shape [256,256,1] --train-dir 4x_conf_retrain_training:latest --val-dir 4x_conf_retrain_validation:latest --epochs 300 --batch-size 128 --val-batch-size 512 --train-steps 129 --learning-rate 1e-5 --learning-rate-params {} --optimizer-type adam --loss-type bce --loss-params {"alpha":0.55,"gamma":2.0} --normalization-type batch_norm --random-seed 42 --model-path None --serialized-channels-first --serialized-channels-first
```

# Virtual Environment Setup on M2 Mac (Apple Silicon)
* First, Run the following commands to give permission for some packages in the requirement.txt:
```
gcloud auth application-default login  
python -m pip install -U pip keyrings.google-artifactregistry-auth  
pip install ml-util --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple  
```

* Second, install packages by requirement.txt:
```
pip install -r requirement.in
```
or
```
pip install -r requirement.txt
```

* Third, install Tensorflow on M1 Mac:
[Apple Developer Link](https://developer.apple.com/metal/tensorflow-plugin/)

* Fourth, install tensorflow-io instead of tensorflow-io-gcs-filesystem==0.31.0:
[StackOverflow Link](https://stackoverflow.com/questions/70277737/cant-install-tensorflow-io-on-m1)

* Fifth, install cellino_tiler:
[GitHub Repository](https://github.com/cellino-biotech/tiler)
```
pip install keyrings.google-artifactregistry-auth
pip install --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple cellino_tiler
pip install git+https://github.com/cellino-biotech/tiler.git
```


# Emulating Intel Python on Apple Silicon
Rosetta 2 can be used as the tool to build the emulation.
```
arch -x86_64 /usr/bin/python3 -m venv iconfluence  
arch -x86_64 /bin/zsh  
source iconfluence/bin/activate  
```
 -->





<!-- # Permission Enabling Scripts to Access Google Cloud
```
gcloud auth application-default login
gcloud config set project project-aims-prod
```
or 

```
export GOOGLE_APPLICATION_CREDENTIALS=/home/shuhangwang/Documents/Keys/training-ml-worker-key.json
```



# Data

* Files needed to run patch_selection_v8.py

https://drive.google.com/drive/folders/1ESoFxlmQYtxSVx340L_l9fVV9DepVOXA




# Virtual Environment Setup on M2 Mac (Apple Silicon)

* First, Run the following commands to give permission for some packages in the requirement.txt
```
gcloud auth application-default login  
python -m pip install -U pip keyrings.google-artifactregistry-auth  
pip install ml-util --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple  
```


* Second, install packages by requirement.txt

```
pip install -r requirement.in
``` 
or
```
pip install -r requirement.txt
```



* Thrid, install Tensorflow on M1 Mac
https://developer.apple.com/metal/tensorflow-plugin/


* Fourth, install tensof-io instead of tensorflow-io-gcs-filesystem==0.31.0
https://stackoverflow.com/questions/70277737/cant-install-tensorflow-io-on-m1


* Fifth, install cellino_tiler
https://github.com/cellino-biotech/tiler
```
pip install keyrings.google-artifactregistry-auth
pip install --extra-index-url https://us-central1-python.pkg.dev/project-ci-cd/stargaze/simple cellino_tiler
pip install git+https://github.com/cellino-biotech/tiler.git

cellino_tiler, cannot be installed correctly...
``` 


# Training

## Lydia's latest run:
* W&B

https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/3g7045ub/overview?workspace=user-swang

* Command

```
/4x_confluence_24_well_edge_fine_tune/run_training.py --wandb-project 4x_conf_retrain --wandb-run-id 3g7045ub --wandb-dataset True --input-shape [256,256,4] --label-shape [256,256,1] --data-input-shape [4,256,256] --data-label-shape [256,256,1] --train-dir 4x_conf_retrain_training:latest --val-dir 4x_conf_retrain_validation:latest --epochs 300 --batch-size 128 --val-batch-size 512 --train-steps 129 --learning-rate 1e-5 --learning-rate-params {} --optimizer-type adam --loss-type bce --loss-params {"alpha":0.55,"gamma":2.0} --normalization-type batch_norm --random-seed 42 --model-path None --serialized-channels-first --serialized-channels-first
```



# Emulating Intel Python on Apple Silicon
Rosetta 2 can be used as the tool to build the emulation.

```
arch -x86_64 /usr/bin/python3 -m venv iconfluence  
arch -x86_64 /bin/zsh  
source iconfluence/bin/activate  
``` -->


