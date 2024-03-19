import json
import keras
import os
from pathlib import Path
from PIL import Image
import wandb
import shutil
import dask.array as da
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

    # save
    file_name = f"CELL-{'-'.join(artifact_path['blob_path'].split('/')[1:3])}-t{artifact_path['time_slice_index']}"
    mask_zarr_path = pj('/home/shuhangwang/Documents/Code/swang_lib/tmp/zarr', f'{file_name}-{model_name}')
    chunk_size = (1024,1024)

    # mask_dask = da.from_array(mask, chunks=chunk_size)
    # import pdb; pdb.set_trace()
    
    mask_dask = da.from_array(mask.astype('float32'), chunks=chunk_size)
    mask_dask = da.reshape(mask_dask, (1,1,1)+mask_dask.shape)
    if os.path.exists(mask_zarr_path):
        shutil.rmtree(mask_zarr_path)
    CellinoZarr(mask_zarr_path, is_local=True).load_data_from_dask_array(mask_dask, '0')

    return mask, mask_img, file_name

def main():
    # model_name = 'model-magic-shadow-537:latest' # v8
    # model_name = 'sunny-waterfall-740:latest' # v9_v2, last epoch
    # model_name = 'model-sunny-waterfall-740:v14' # 
    # model_name = 'model-sunny-waterfall-740:v39' # v9_v2, lowest loss
    # context_file = 'context-002612-Oct_06_2023 09_29_02.json'
    # context_file = 'context-002653-Oct_16_2023 09_07_16.json'
    # context_file = 'context-001897-Oct_17_2023 10_21_00.json'
    # context_file = 'context-002658-Oct_17_2023 13_02_38.json'
    # context_file = 'context-002530-Oct_17_2023 13_13_57.json'

    context_file = '/home/shuhangwang/Documents/Dataset/conf_v10/test_contexts/CELL-002802-A4-2.json'
    # context_file = '/home/shuhangwang/Documents/Dataset/conf_v10/test_contexts/CELL-002802-A4-5.json'
    # context_file = '/home/shuhangwang/Documents/Dataset/conf_v10/test_contexts/CELL-002804-C3-2.json'
    # context_file = '/home/shuhangwang/Documents/Dataset/conf_v10/test_contexts/CELL-002804-C3-5.json'


    # for model_name in ['model-sunny-waterfall-740:v14', 'model-magic-shadow-537:latest']:
    # for model_name in ['model-magic-shadow-537:latest', 'model-kind-wave-816:v5']:
    # for model_name in ['model-kind-wave-816:v10', 'model-kind-wave-816:v5']:
    # for model_name in ['model-kind-wave-816:v0', 'model-kind-wave-816:v1']:
    # for model_name in ['model-eager-aardvark-821:v0', 'model-eager-aardvark-821:v5']:
    # for model_name in ['model-rose-cherry-830:v62', 'model-rose-cherry-830:v20']:
    for model_name in ['model-super-field-826:v78',]:
    # for model_name in ['model-magic-shadow-537:latest',]:
        mask, mask_img, file_name = single_confluence_predict(model_name, context_file)
        mask_img.save(pj('./tmp', f"{file_name}-{model_name}.png"))

if __name__=='__main__':
    main()