import os
import socket
import numpy as np
import dask.array as da
import pandas as pd
from os.path import join as pj
import wandb
import pandas as pd

from stargaze_util.aims_client import AIMSClient_Sync
from stargaze_util.cellino_zarr import CellinoZarr
from stargaze_util.helpers.zarr_helpers import copy_to_new_zarr
from aims_api_utils import image_analysis as img_util
from aims_api_utils import functions as aims_func

from ml_util.file_reader import read_google_sheet





# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/shuhangwang/Documents/Keys/training-ml-worker-key.json' #



def save_scaled_array_tmp(arr, file_name=None):
    from PIL import Image
    scale_factor = 0.2
    new_size = (int(arr.shape[-2] * scale_factor), int(arr.shape[-1] * scale_factor))
    img = Image.fromarray(arr).resize(new_size).convert('RGB')
    if file_name is None:
        file_name = 'tmp.png'
    else:
        file_name = 'tmp_' + file_name
    img.save(pj('./tmp', file_name))

def get_zarr_obj(path, project_id, credential_file=None):
    z = CellinoZarr(file_path=path,
                    project_id=project_id,
                    credential_file=credential_file,
                    is_local=False)
    return z


def create_pluri_mask(confluence_mask_filepath, ime_id, plate, well, data_dir):

    project_aims = 'project-ml-prod'
    aims = AIMSClient_Sync(project_id=project_aims)

    import pdb; pdb.set_trace()

    #generate mask file 
    # available_tags = aims_func.get_finding_tags_by_image_event_api(aims, ime_id)
    # tag_name_values = { x: 1 for x in available_tags}#???
    tag_name_values = {'differentiation': 1}
    mask_file = img_util.generate_finding_mask_api(aims, ime_id, tag_name_values, additive=False, credential_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))


    #pulling data from cellino-plate-db gcloud *notice that GOOGLE_APPLICATION_CREDENTIALS' are hardcoded into function 
    zarrroot = get_zarr_obj(path=confluence_mask_filepath, 
                            project_id='cellino-plate-db', 
                            credential_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')).get_zarr_root()

    if zarrroot['0'].ndim == 3:
        image_da = da.from_zarr(zarrroot['0'])[0, :, :].squeeze()
    elif zarrroot['0'].ndim == 5:    
        image_da = da.from_zarr(zarrroot['0'])[0,0,0,:,:].squeeze()
    else:
        raise ValueError(f"data ndim = {zarrroot['0'].ndim} not supported")
    image_array = image_da.compute()
    image_array = np.clip(image_array, 0, 1)
    im_filter = da.subtract(image_array, mask_file)
    proc_filter = np.clip(im_filter, 0, 1)
    proc_filter = np.reshape(proc_filter,[1, 1, 1, 10240, 10240])
    proc_da = proc_filter.rechunk(1024, 1024)

    processed_zarr_path = os.path.join(data_dir, f'{plate}_{well}')
    cz = CellinoZarr(processed_zarr_path, is_local=True)
    cz.load_data_from_dask_array(proc_da, '0')


def generate_save_pluri_zarr():
    OUTPUT_BUCKET_NAME = 'cellino-ml-training/4x_pluri'
    save_dir = '/media/slee/DATA1/DL/data/pluripotency/Pluri_manual_mask'
    output_csv_file = './scripts/file_table.csv'

    celldf = read_google_sheet(filename='Cell ID - Centralized Ground Truth Log', sheetname='4X Confluence Ground Truth')
    select_columns = ['ime_id', 'plate_barcode', 'well_name', 'zarr_path', 'time_slice_index', 'gt_zarr_path', 'Notes']
    output_columns = ['ime_id', 'plate_barcode', 'well_name', 'zarr_path', 'time_slice_index', 'confluence_mask_zarrpath', 'plate_type', 'plurimask_zarrpath']
    file_list = pd.DataFrame(columns=output_columns)
    # import pdb; pdb.set_trace()
    for idx in range(16, len(celldf)):
        ime_id, plate, well, brt_zarr_path, time_slice_index, confluence_mask_filepath, notes = celldf.loc[idx, select_columns]

        create_pluri_mask(confluence_mask_filepath, ime_id, plate, well, save_dir)
        plurimask_zarr_path = os.path.join(save_dir, f'{plate}_{well}')
        cellino_zarr_src = CellinoZarr(plurimask_zarr_path, True)

        plurimask_name = f'pluri_manual_mask_ver0_{plate}_{well}'

        plurimask_remote_zarr_path = os.path.join(OUTPUT_BUCKET_NAME, plurimask_name)
                                                
        # cellino_zarr_dst = CellinoZarr(plurimask_remote_zarr_path, False, credential_file =crd_file, project_id='project-ml-storage')
        # copy_to_new_zarr(cellino_zarr_src, cellino_zarr_dst)

        if notes.lower().find('greiner') > -1:
            plate_type = 'greiner'
        elif notes.lower().find('ibidi') > -1:
            plate_type = 'ibidi'
        else:
            plate_type = None
        file_list.loc[len(file_list)] = [ime_id, plate, well, brt_zarr_path, time_slice_index, confluence_mask_filepath, plate_type, plurimask_remote_zarr_path]

    file_list.to_csv(output_csv_file, index=False)



if __name__=='__main__':

    generate_save_pluri_zarr()

 
    # upload to wandb

    PROJECT_NAME = "4x_pluripotency"
    DATA_NAME = 'Pluri_manual_mask'
    WORKING_DIR = f'/media/slee/DATA1/DL/data/pluripotency/{DATA_NAME}/'
    GT_ARTIFACT = f"gt_pluri_4x_manual"
    CSV_FILE = './scripts/file_table.csv'


    zarr_func_list='["apply_global_thresh_da_zslice","apply_binary_opening_da_zslice","apply_binary_closing_da_zslice"]'
    zarr_kwargs_list='[{"thresh": "0.3"},{"ball_radius": "convert_um2pix(15)"},{"ball_radius": "convert_um2pix(15)"}]'
    data_meta = {'confluence_mask_generation_function_list': zarr_func_list, 'confluence_mask_generation_parameter_list': zarr_kwargs_list}
    

    with wandb.init(project=PROJECT_NAME,  entity="cellino-ml-ninjas", dir='/media/slee/DATA1/DL/data/wandb',
                    config=data_meta, job_type="dataset") as run:
        # adding file infos from spread sheet
        art = wandb.Artifact(GT_ARTIFACT, type="dataset")
        data_list = pd.read_csv(CSV_FILE)
        file_table = wandb.Table(dataframe=data_list)
        # art.add(file_table, "file_table")
        art.add_file(CSV_FILE)
        run.log({'file_list': file_table})
        run.log_artifact(art)