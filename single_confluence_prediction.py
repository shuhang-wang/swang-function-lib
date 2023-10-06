import json
import keras
import os
from pathlib import Path
from PIL import Image
import wandb
from stargaze_util.cellino_zarr import CellinoZarr
from zarrtfutil.zarrconfluence import ZarrConfluence
from zarrtfutil.helpers import normalize_well
from zarrutil.zarr_util import applyfun2d
pj = os.path.join


def artifact_path_context(context_file):
    '''
        read artifact path from context file
    '''
    with open(context_file, 'r') as file:
        data = json.load(file)
    return data['context']['artifactPath']


def single_confluence_predict(model_name, context_file):
    '''
    predict confluence map using a single model on a single zarr file
    '''

    batch_size = 1000
    prob_thresh = 0.5
    scale_level = 0
    tile_size = 256
    z_indices = [0, 1, 2, 3]

    api = wandb.Api()
    entity = 'cellino-ml-ninjas'
    project = '4x_conf_retrain'
    model_art = api.artifact(f'{entity}/{project}/{model_name}')
    model_path = Path(model_art.download())
    model = keras.models.load_model(model_path, compile=False)

    artifact_path = artifact_path_context(context_file)

    zarr = CellinoZarr(pj(artifact_path['bucket'], artifact_path['blob_path']), is_local=False, 
                           project_id=artifact_path['project'], credential_file=os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    time_slice_index = artifact_path['time_slice_index']

    # Prep for confluence prediction
    preproc_funcs = [
        {
            'func': applyfun2d,
            'kwargs': {'fun': normalize_well, 'axes': [3, 4], 'bdelayed': False, 'centerarea': []},
        }
    ]
    zcf = ZarrConfluence([zarr])
    # import pdb; pdb.set_trace()
    zcf.load_data(array_name=str(scale_level), t_index=time_slice_index, c_index=0, z_indices=[z_indices], function_list=[preproc_funcs, []])
    zcf.tile_size = tile_size
    zcf.curzarrinx = 0

    # Predict confluence mask
    mask, _ = zcf.predict_zarr(model.predict, prob_thresh=prob_thresh,
                                            batch_size=batch_size)
    mask_img = Image.fromarray(mask)

    return mask, mask_img

def main():
    # model_name = 'model-magic-shadow-537:latest'
    # model_name = 'sunny-waterfall-740:latest' # v9_v2
    model_name = 'model-sunny-waterfall-740:v0'
    context_file = 'context-002612-Oct_06_2023 09_29_02.json'
    mask, mask_img = single_confluence_predict(model_name, context_file)
    mask_img.save(pj('./tmp', model_name + '.png'))

if __name__=='__main__':
    main()