import wandb
import os
CELLINO_PROJECT_ID = 'project-ml-storage'
WB_ENTITY = "cellino-ml-ninjas"
WB_PROJECT_NAME = "constants"
pj = os.path.join

def main():

    run = wandb.init(
        id=CELLINO_PROJECT_ID,
        project=WB_PROJECT_NAME,
        entity=WB_ENTITY,
    )

    cred_cell_id_google_sheet = '/home/shuhangwang/Documents/Keys/cred_cell_id_google_sheet.json'

    # Create an artifact (Artifacts can represent datasets, models, and other files related to your ML workflow)
    artifact = wandb.Artifact('cred_cell_id_google_sheet', type='credentials')

    # Add the local JSON file to the artifact
    artifact.add_file(cred_cell_id_google_sheet)

    # Log the artifact to your W&B run
    run.log_artifact(artifact)

    # Finish the run
    run.finish()

def test():
    run = wandb.init(
        id=CELLINO_PROJECT_ID,
        project=WB_PROJECT_NAME,
        entity=WB_ENTITY,
    )
    artifact = wandb.use_artifact('cellino-ml-ninjas/constants/cred_cell_id_google_sheet:latest', type='credentials')
    artifact_dir = artifact.download()
    print(pj(artifact_dir, 'cred_cell_id_google_sheet.json'))
    run.finish()

if __name__=='__main__':
    # main()
    test()