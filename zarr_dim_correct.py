
import os
from stargaze_util.cellino_zarr import CellinoZarr
from stargaze_util.helpers.zarr_helpers import copy_to_new_zarr
import dask.array as da

src_zarr_path = 'starlight-computing/IMX_Imaging_v2/CELL-001728/4x-confluence-gt_t0.3/Processed_CELL-001728_D2_t0'
# loading the zarr and manipulate the shape
project_id = 'cellino-plate-db'
cellino_zarr_src = CellinoZarr(src_zarr_path, False, credential_file =os.environ['GOOGLE_APPLICATION_CREDENTIALS'], project_id=project_id)
rt = cellino_zarr_src.get_zarr_root()
xx = da.from_zarr(rt['0'])
yy = da.reshape(xx, (1,1)+xx.shape)

# creating local zarr
local_zarr_path = 'test'
cellino_zarr_local = CellinoZarr(local_zarr_path, True)
cellino_zarr_local.load_data_from_dask_array(yy, '0')

# currently make an error since my credential does not have write permission.
# but with the write permission, the following copy  should work. 
remote_zarr_path = 'zarr-dim-corrected/IMX_Imaging_v2/CELL-001728/4x-confluence-gt_t0.3/Processed_CELL-001728_D2_t0-1'
project_id = 'project-ml-storage'
cellino_zarr_local = CellinoZarr(local_zarr_path, True)
cellino_zarr_dst = CellinoZarr(remote_zarr_path, False, credential_file =os.environ['GOOGLE_APPLICATION_CREDENTIALS'], project_id=project_id)
# import pdb; pdb.set_trace()
copy_to_new_zarr(cellino_zarr_local, cellino_zarr_dst)