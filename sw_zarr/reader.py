
# pulic package
import json
import os
from os.path import join as pj
import dask.array as da
import numpy as np

# cellino package
from stargaze_util.cellino_zarr import CellinoZarr
from ml_util.file_reader import read_google_sheet

def load_gaia_context_file(context_file):
    '''
        load the data from context file
    '''
    with open(context_file, 'r') as file:
        data = json.load(file)
    # return data['context']['artifactPath']
    return data['context']

def zarr_path_google_sheet(series):
    '''
        get artifact path form google sheet
    '''
    bucket_blob_split_index = series['zarr_path'].find('/')

    artifact_path = {'project': series['project'], 
                     'bucket': series['zarr_path'][:bucket_blob_split_index], 
                     'blob_path': series['zarr_path'][bucket_blob_split_index+1:], 
                     'time_slice_index': series['time_slice_index'],
                     'datatype': 'zarr', }
    return artifact_path

def read_zarr(context_file=None, artifact_path=None, zarr_path=None, is_local=False):
    '''
    one out of context_file, artifact_path and zarr_path should be provided
    read zarr data as numpy array
    '''
    # import pdb; pdb.set_trace()
    if is_local:
        zarr = CellinoZarr(zarr_path, is_local=True)
        time_slice_index = 0
    else:
        if context_file is not None:
            artifact_path = load_gaia_context_file(context_file)['artifactPath']
            
        zarr = CellinoZarr(pj(artifact_path['bucket'], artifact_path['blob_path']), is_local=False, 
                           project_id=artifact_path['project'], credential_file=os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
        time_slice_index = artifact_path['time_slice_index']
    
    zarr_root = zarr.get_zarr_root()
    
    # ‘0’ refers to the full resolution of the image. 0 means no multiscale
    # if the image is multiscaled, it should have ‘1’, ‘2’, 
    # the number is the downscaled number
    zarr_root = da.from_zarr(zarr_root['0'])
    if len(zarr_root.shape)==3:
        # for mask of ground truth
        # darr = zarr_root[time_slice_index, slice(None), slice(None)].astype(np.float16).compute()
        darr = zarr_root[time_slice_index, slice(None), slice(None)].astype(np.float32)
    else:
        # for dim of (time, channel, z, y, x)
        # TODO: [time_slice_index, 0, 0, slice(None), slice(None)]-->[time_slice_index, 0, :, slice(None), slice(None)]
        # Some functions based on this function might to be updated
        # darr = zarr_root[time_slice_index, 0, 0, slice(None), slice(None)].astype(np.float16).compute()
        darr = zarr_root[time_slice_index, 0, :, slice(None), slice(None)].astype(np.float32)
        darr = np.transpose(darr, (1, 2, 0))
    
    return (zarr, darr, artifact_path,)

if __name__=='__main__':
    
    context_file = '/home/shuhangwang/Documents/Code/swang_lib/sw_zarr/context-002574-Jan_02_2024 11_18_55.json'
    apath = load_gaia_context_file(context_file)['artifactPath']

    
    df = read_google_sheet(filename='Cell ID - Centralized Ground Truth Log', sheetname='4X Confluence Ground Truth')
    artifact_path = zarr_path_google_sheet(df.iloc[0])
    zarr, darr, artifact_path = read_zarr(context_file=None, artifact_path=artifact_path, is_local=False, zarr_path=None)

    import pdb; pdb.set_trace()