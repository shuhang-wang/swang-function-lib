
# public packages
import os
import json
import numpy as np
from multiprocessing.pool import Pool
from os.path import join as pj

# Cellino packages
from stargaze_util.cellino_zarr import CellinoZarr
import dask.array as da
from ml_util.file_reader import read_google_sheet

# developing packages

zarr_original_dir = '/home/shuhangwang/Documents/Dataset/zarr_orignal_correct'

def load_remote_zarr_to_local(remote_zarr_path, time_slice_index, project_id = 'cellino-plate-db', local_zarr_path = 'test'):
    '''
        This function read remote zarr and load to local machine,
        only one time_slice_index
    '''
    # loading the zarr
    # import pdb; pdb.set_trace()
    cellino_zarr_remote = CellinoZarr(remote_zarr_path, False, credential_file =os.environ['GOOGLE_APPLICATION_CREDENTIALS'], project_id=project_id)
    root = cellino_zarr_remote.get_zarr_root()
    darr = da.from_zarr(root['0'])[time_slice_index:time_slice_index+1,:,:,:,:]

    x = darr.compute()
    num_inf = np.isinf(x).sum()
    print(f'{remote_zarr_path}, time_slice_index: {time_slice_index}')
    print(f'...num_inf in the remote: {num_inf}...')
    print(f'darr shape: {x.shape}')

    # creating local zarr
    cellino_zarr_local = CellinoZarr(local_zarr_path, True)
    cellino_zarr_local.load_data_from_dask_array(darr, '0')

    print(f'Done: {remote_zarr_path}')

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

def pull_zarr_contamination():
    '''
        Download zarr files for contamination instances
    '''

    df = read_google_sheet(filename='ground_truth_tags', sheetname='contamination')
    df['artifact_path'] = df['artifact_path'].apply(json.loads)
    # for idx in range(0, len(df)):
    for idx in range(1, len(df)):
        print(f'{idx} out of {len(df)}')
        row = df.iloc[idx]
        
        artifact_path = row['artifact_path']
        time_slice_index = artifact_path['time_slice_index']
        remote_zarr_path = pj(artifact_path['bucket'], artifact_path['blob_path'])
        project = artifact_path['project']

        local_zarr_path = pj(zarr_original_dir, '-'.join([row['plate_barcode'], row['well_name'], str(artifact_path['time_slice_index'])]))
        if not os.path.isdir(local_zarr_path):
            load_remote_zarr_to_local(remote_zarr_path, time_slice_index, project, local_zarr_path)
        break

def pull_zarr_contamination_mprocess():
    '''
        Download zarr files for contamination instances
    '''

    df = read_google_sheet(filename='ground_truth_tags', sheetname='contamination')
    df['artifact_path'] = df['artifact_path'].apply(json.loads)

    single_patch_args = []

    for idx in range(0, len(df)):
        print(f'{idx} out of {len(df)}')
        row = df.iloc[idx]
        
        artifact_path = row['artifact_path']
        remote_zarr_path = pj(artifact_path['bucket'], artifact_path['blob_path'])
        time_slice_index = artifact_path['time_slice_index']
        project = artifact_path['project']
        local_zarr_path = pj(zarr_original_dir, '-'.join([row['plate_barcode'], row['well_name'], str(artifact_path['time_slice_index'])]))
        
        if not os.path.isdir(local_zarr_path):
            single_patch_args.append((remote_zarr_path, time_slice_index, project, local_zarr_path))

    # load_remote_zarr_to_local(*single_patch_args[0])
    with Pool() as pool:
        results = pool.starmap_async(load_remote_zarr_to_local, single_patch_args, chunksize=10)
        results.get()

    
    df = read_google_sheet(filename='Cell ID - Centralized Ground Truth Log', sheetname='4X Confluence Ground Truth')

    single_patch_args = []

    for idx in range(len(df)):
        print(f'{idx} out of {len(df)}')
        row = df.iloc[idx]
        
        artifact_path = zarr_path_google_sheet(row)
        remote_zarr_path = pj(artifact_path['bucket'], artifact_path['blob_path'])
        project = artifact_path['project']

        local_zarr_path = pj(zarr_original_dir, '-'.join([row['plate_barcode'], row['well_name'], str(row['time_slice_index'])]))
        if not os.path.isdir(local_zarr_path):
            single_patch_args.append((remote_zarr_path, project, local_zarr_path))

    with Pool() as pool:
        results = pool.starmap_async(load_remote_zarr_to_local, single_patch_args, chunksize=10)
        results.get()

def read_zarr(context_file=None, artifact_path=None, zarr_path=None, is_local=False):
    '''
    one out of context_file, artifact_path and zarr_path should be provided
    read zarr data as numpy array
    '''
    zarr = CellinoZarr(zarr_path, is_local=True)
    time_slice_index = 0

    zarr_root = zarr.get_zarr_root()
    
    zarr_root = da.from_zarr(zarr_root['0'])
    # for dim of (time, channel, z, y, x)
    # darr = zarr_root[time_slice_index:time_slice_index+1, :, :, :,:].astype(np.float16)
    darr = zarr_root[time_slice_index:time_slice_index+1, :, :, :,:]
    
    return (zarr, darr, artifact_path,)

def read_test():
    from PIL import Image
    

    # local_zarr_path = '/home/shuhangwang/Documents/Dataset/zarr_orignal_correct/CELL-002176-B8-5'
    local_zarr_path = '/home/shuhangwang/Documents/Dataset/zarr_orignal_correct/CELL-002176-B8-4'
    _, darr, _ = read_zarr( zarr_path=local_zarr_path, is_local=True)
    locs = np.where(np.isinf(darr.astype(np.float32)).compute())
    import pdb; pdb.set_trace()

    arr = darr.compute()
    num_inf = np.isinf(arr).sum()
    print(f'...num_inf in the local: {num_inf}...')
    print(f'darr shape: {darr.shape}')

    
    import scipy.ndimage as ndimage
    inf_mask = np.isinf(np.transpose(arr[0,0,:,:,:], (1, 2, 0)))[:,:,0]
    inf_mask = ndimage.grey_dilation(inf_mask, size=(20, 20))  # Adjust size as needed
    inf_mask = (inf_mask * 255).astype(np.uint8)
    image_to_save = Image.fromarray(inf_mask)
    image_to_save = image_to_save.convert('L')
    image_to_save.save('xxx1.png')

if __name__=='__main__':

    # pull_zarr_contamination()
    # pull_zarr_contamination_mprocess()

    read_test()