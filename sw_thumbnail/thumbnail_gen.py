
# public package
import os
import numpy as np
import dask.array as da
from PIL import Image
from os.path import join as pj

#cellino package
from stargaze_util.cellino_zarr import CellinoZarr

def read_arr_from_zarr(file_path, t_idx=0, z_slice=0, is_local=True, project=''):
    if is_local:
        cellino_zarr = CellinoZarr(file_path, is_local=True)
    else:
        credential_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        cellino_zarr = CellinoZarr(file_path, is_local=False, project_id=project,  credential_file=credential_file)

    rt = cellino_zarr.get_zarr_root()

    # Scale the zarr to write out to images
    # NOTE: This is a hacky work around for non-standard zarr generation!! Need to make this more of a priority!
    if len(rt['0'].shape) == 3:
        raw_arr = da.from_zarr(rt['0'])[0, slice(None), slice(None)].astype(np.float16).compute()
    else:
        raw_arr = da.from_zarr(rt['0'])[t_idx, 0, z_slice, slice(None), slice(None)].astype(np.float16).compute()

    a_max = np.nanmax(raw_arr[raw_arr != np.inf])
    a_min = np.nanmin(raw_arr[raw_arr != -np.inf])
    if a_max==a_min:
        a_max += 0.0001
    arr = np.nan_to_num((((raw_arr - a_min) / (a_max - a_min)) * 255), neginf=0, posinf=255).astype(np.uint8)
    return arr

def get_scaled_img_from_zarr(file_path, t_idx=0, z_slice=0, is_local=True, project='', scale_factor=0.125):

    arr = read_arr_from_zarr(file_path, t_idx, z_slice=z_slice, is_local=is_local, project=project)
    new_size = (int(arr.shape[-2] * scale_factor), int(arr.shape[-1] * scale_factor))
    img = Image.fromarray(arr).resize(new_size).convert('RGB')

    return img    


if __name__=='__main__':
    zarr_path = '/home/shuhangwang/Documents/Dataset/ground_truth/ttt/tfolder/Processed_CELL-001815_D2_t0'
    full_img_path = '/home/shuhangwang/Documents/Dataset/ground_truth/ttt/test.png'
    img = get_scaled_img_from_zarr(zarr_path, is_local=True)
    img.save(full_img_path)