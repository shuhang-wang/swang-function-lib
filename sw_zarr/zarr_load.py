
# public packages
import os
from os.path import join as pj


# Cellino packages
from stargaze_util.cellino_zarr import CellinoZarr
import dask.array as da
from ml_util.file_reader import read_google_sheet

# developing packages

zarr_original_dir = '/home/shuhangwang/Documents/Dataset/zarr_orignal'

def load_remote_zarr_to_local(remote_zarr_path, project_id = 'cellino-plate-db', local_zarr_path = 'test'):
    '''
        This function read remote zarr and load to local machine
    '''
    # loading the zarr
    
    cellino_zarr_remote = CellinoZarr(remote_zarr_path, False, credential_file =os.environ['GOOGLE_APPLICATION_CREDENTIALS'], project_id=project_id)
    root = cellino_zarr_remote.get_zarr_root()
    darr = da.from_zarr(root['0'])

    # creating local zarr
    cellino_zarr_local = CellinoZarr(local_zarr_path, True)
    cellino_zarr_local.load_data_from_dask_array(darr, '0')

def zarr_path_google_sheet(df_row):
    '''
        get artifact path form google sheet
    '''
    bucket_blob_split_index = df_row['zarr_path'].find('/')

    artifact_path = {'project': df_row['project'], 
                     'bucket': df_row['zarr_path'][:bucket_blob_split_index], 
                     'blob_path': df_row['zarr_path'][bucket_blob_split_index+1:], 
                     'time_slice_index': df_row['time_slice_index'],
                     'datatype': 'zarr', }
    return artifact_path

def pull_zarr_cell_id_ground_truth():
    
    df = read_google_sheet(filename='Cell ID - Centralized Ground Truth Log', sheetname='4X Confluence Ground Truth')

    for idx in range(len(df)):
        print(f'{idx} out of {len(df)}')
        row = df.iloc[idx]
        
        artifact_path = zarr_path_google_sheet(row)
        remote_zarr_path = pj(artifact_path['bucket'], artifact_path['blob_path'])
        project = artifact_path['project']

        local_zarr_path = pj(zarr_original_dir, '-'.join([row['plate_barcode'], row['well_name'], str(row['time_slice_index'])]))
        if not os.path.isdir(local_zarr_path):
            load_remote_zarr_to_local(remote_zarr_path, project, local_zarr_path)

if __name__=='__main__':
    # remote_zarr_path = 'starlight-computing/IMX_Imaging_v2/CELL-001728/4x-confluence-gt_t0.3/Processed_CELL-001728_D2_t0'
    # load_remote_zarr_to_local(remote_zarr_path, project_id = 'cellino-plate-db', local_zarr_path = 'test')
    pull_zarr_cell_id_ground_truth()