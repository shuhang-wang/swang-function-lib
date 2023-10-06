'''
Generate visualization for
1) brt 
2) confluence map 
3) confluence confidence map
4) confluence confidence map + patches overlaid
5) ground-truth image
'''

import os
import json
import numpy as np
from datetime import datetime
import glob
import wandb
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation
from functools import wraps
from stargaze_util.cellino_zarr import CellinoZarr
import dask.array as da
from zarrutil import zarr_util as zu
from cellino_tiler import patch
from cellino_tiler.patch_utils import patcher
from PIL import Image, ImageDraw
from cellino_tiler.patch import PatchDataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap


pj = os.path.join

# Setting up the negative patches shared y_artifact_path
y_artifact_path_negative = {
    'project': 'cellino-plate-db',
    'datatype': 'zarr',
    'bucket': 'starlight-computing',
    'blob_path': 'Users/daksel/Processed_BlankZarr_t0',
    'time_slice_index': 0
}

def clip_and_stretch(img_data, lower_percentile=1, upper_percentile=99):

    # Compute the lower and upper percentile values
    lower_bound = np.percentile(img_data, lower_percentile)
    upper_bound = np.percentile(img_data, upper_percentile)

    # Clip and linearly stretch the image data
    img_data = np.clip(img_data, lower_bound, upper_bound)
    img_data = 255 * (img_data - lower_bound) / (upper_bound - lower_bound)
    img_data = img_data.astype(np.uint8)

    return img_data


def generate_color_confidence(arr, brt_arr):

    def power_transform(x, power=0.5):
        """A power function to emphasize the brighter areas."""
        return x ** power

    # Generate a power-based set of values between 0 and 1
    num_points = 256
    x = np.linspace(0, 1, num_points)
    y = power_transform(x, 10.0)

    # Create a colormap based on the power transition
    colors = np.zeros((num_points, 3))
    colors[:, 0] = 0.4 + 0.4*y  # Red channel
    colors[:, 2] = 0.4 + 0.4*y  # Blue channel


    arr = arr/255.0
    cnfi_arr = np.abs(arr-0.5)/0.5
    

    # Create a custom colormap from bright purple to dark purple
    colors = [(0.8, 0, 0.8),  # bright purple
            (0.4, 0, 0.4)]  # dark purple
    cmap_name = 'purple_scale'
    purple_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


    # Map the grayscale values to the custom colormap
    confi_img = (purple_cmap(cnfi_arr) * 255).astype(np.uint8)

    threshold = 0.1  # Set this to your desired threshold value
    # Create a boolean mask where arr values are less than the threshold
    mask = arr < threshold
    # Set the corresponding pixels in confi_img to white using the mask
    confi_img[mask, :3] = brt_arr[mask][:,None]  # RGB value for white

    return confi_img

def draw_save(img, file_dict, file_path, scale_factor, save_directory, t_idx, second_type=''):
    
    for draw_patch in [False, True]:
        if draw_patch:
            width = 3
            outline = (65, 105, 225)
            # Drawing rectangles
            draw = ImageDraw.Draw(img)
            # For each patch that pulls from that file
            for p in file_dict[file_path]['time_indices'][t_idx]:
                scaled_rect = [int(r*scale_factor) for r in p.rect]
                draw.rectangle(((scaled_rect[1], scaled_rect[0]), (scaled_rect[3], scaled_rect[2])), outline=outline,
                            width=width)

        # Save out
        save_name = file_path.replace('/', '_') + '_t' + str(t_idx) + second_type + '_patch' * draw_patch
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        full_img_path = os.path.join(save_directory, save_name)
        print(f'Saving at {full_img_path}.png')
        img.save(full_img_path + '.png')

def thumbnail_save(img, file_path, save_directory, t_idx, second_type=''):
        save_name = file_path.replace('/', '_') + '_t' + str(t_idx) + '_' + second_type
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        full_img_path = os.path.join(save_directory, save_name)
        print(f'Saving at {full_img_path}.png')
        img.save(full_img_path + '.png')

def read_arr(file_dict, file_path, z_slices_dict, credential_file):
    # file_path = list(file_dict.keys())[1]
    if len(file_dict[file_path]['bucket'])>0:
        cellino_zarr = CellinoZarr(file_path, is_local=False, project_id=file_dict[file_path]['project'],  credential_file=credential_file)
    else:
        cellino_zarr = CellinoZarr(file_path, is_local=True, credential_file=credential_file)

    rt = cellino_zarr.get_zarr_root()

    # For each time point from that file
    t_idx = list(file_dict[file_path]['time_indices'].keys())[0]
    # Taking the bottom z-slice for now
    z_slice = z_slices_dict[file_path][t_idx][0]

    # Scale the zarr to write out to images
    # NOTE: This is a hacky work around for non-standard zarr generation!! Need to make this more of a priority!
    if len(rt['0'].shape) == 3:
        raw_arr = da.from_zarr(rt['0'])[0, slice(None), slice(None)].astype(np.float16).compute()
    else:
        raw_arr = da.from_zarr(rt['0'])[t_idx, 0, z_slice, slice(None), slice(None)].astype(np.float16).compute()

    a_max = np.nanmax(raw_arr[raw_arr != np.inf])
    a_min = np.nanmin(raw_arr[raw_arr != -np.inf])
    arr = np.nan_to_num((((raw_arr - a_min) / (a_max - a_min)) * 255), neginf=0, posinf=255).astype(np.uint8)
    return arr

def read_arr_simple(file_path, t_idx=0, z_slice=0, is_local=True, project='', credential_file=''):
    # file_path = list(file_dict.keys())[1]
    if is_local:
        cellino_zarr = CellinoZarr(file_path, is_local=True)
    else:
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
    arr = np.nan_to_num((((raw_arr - a_min) / (a_max - a_min)) * 255), neginf=0, posinf=255).astype(np.uint8)
    return arr

def generate_thumbnail_single_well(dataset: PatchDataset, credential_file: str = None, save_directory='', scale_factor=0.125):

    # If the save_directory doesn't end in a '/' add it
    if len(save_directory) > 0 and not save_directory.endswith('/'):
        save_directory += '/'

    # If a credential file wasn't specified, pull from standard environmental variable 
    if credential_file is None:
        credential_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

    # Get the patches
    patches = dataset.get_patches()
    
    file_dict = {}
    z_slices_dict = {}
    for p in patches:
        for f_type in dataset.get_feature_names():
            artifact_path = p.artifact_infos[f_type]['artifact_path']
            path_str = pj(artifact_path['bucket'], artifact_path['blob_path'])

            # If the path is already in the dict
            if path_str in file_dict.keys():
                # If the time_index is already in the path
                if artifact_path['time_slice_index'] in file_dict[path_str]['time_indices'].keys():
                    file_dict[path_str]['time_indices'][artifact_path['time_slice_index']] = \
                        file_dict[path_str]['time_indices'][artifact_path['time_slice_index']] + [p]
                else:
                    file_dict[path_str]['time_indices'][artifact_path['time_slice_index']] = [p]
                    z_slices_dict[path_str][artifact_path['time_slice_index']] = p.artifact_infos[f_type]['z_slices']
            else:
                file_dict[path_str] = {'time_indices': {artifact_path['time_slice_index']: [p]},
                                       'project': artifact_path['project'], 'bucket':artifact_path['bucket']}
                z_slices_dict[path_str] = {artifact_path['time_slice_index']: p.artifact_infos[f_type]['z_slices']}
    
    # For bright field image
    file_path = list(file_dict.keys())[0]
    t_idx = list(file_dict[file_path]['time_indices'].keys())[0]    
    brt_arr = read_arr(file_dict, file_path, z_slices_dict, credential_file)
    brt_arr = clip_and_stretch(brt_arr, upper_percentile=70, lower_percentile=5)
    new_size = (int(brt_arr.shape[-2] * scale_factor), int(brt_arr.shape[-1] * scale_factor))
    brt_img = Image.fromarray(brt_arr).resize(new_size).convert('RGB')
    draw_save(brt_img, file_dict, file_path, scale_factor, save_directory, t_idx)
    
    # For 'Y' image
    second_type='confidence'
    file_path = list(file_dict.keys())[1]
    t_idx = list(file_dict[file_path]['time_indices'].keys())[0]
    arr = read_arr(file_dict, file_path, z_slices_dict, credential_file)
    if second_type=='confluence':
        arr = (arr>=0.5).astype('float')
        img = Image.fromarray(arr).resize(new_size).convert('RGB')
    elif second_type=='confidence':
        arr = generate_color_confidence(arr, brt_arr)
        img = Image.fromarray(arr).resize(new_size)
    else:
        # ground truth or mask
        pass
    draw_save(img, file_dict, file_path, scale_factor, save_directory, t_idx, second_type)

    return

def generate_thumbnail_zarr(credential_file: str = None, dir_exp='', scale_factor=0.125):

    # If a credential file wasn't specified, pull from standard environmental variable 
    if credential_file is None:
        credential_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    

    import glob
    artifact_path = artifact_path_context(glob.glob(pj(dir_exp, '*.json'))[0])
    file_path = pj(artifact_path['bucket'], artifact_path['blob_path'])
    t_idx = artifact_path['time_slice_index']

    # For bright field image  
    brt_arr = read_arr_simple(file_path, t_idx, z_slice=0, is_local=False, project=artifact_path['project'], credential_file=credential_file)
    brt_arr = clip_and_stretch(brt_arr, upper_percentile=70, lower_percentile=5)
    new_size = (int(brt_arr.shape[-2] * scale_factor), int(brt_arr.shape[-1] * scale_factor))
    brt_img = Image.fromarray(brt_arr).resize(new_size).convert('RGB')
    thumbnail_save(brt_img, file_path, pj(dir_exp, 'visualization'), t_idx, second_type='')
    
    # For 'Y' image
    # second_type='confidence'
    second_type='confluence'
    file_path = '/home/shuhangwang/Documents/Dataset/visualization/differentiation/masks/aims-storage-prod_AIMS_IXM2_001815_A4_standard_well_stitch_v1.1.8_4x_TL-10 4x_4z_c97b458f-eb0d925d'
    t_idx = 0
    arr = read_arr_simple(file_path)
    import pdb; pdb.set_trace()
    if second_type=='confluence':
        arr = (arr>=127).astype('float') * 255
        img = Image.fromarray(arr).resize(new_size).convert('RGB')
    elif second_type=='confidence':
        arr = generate_color_confidence(arr, brt_arr)
        img = Image.fromarray(arr).resize(new_size)
    else:
        # ground truth or mask
        pass
    thumbnail_save(img, file_path, pj(dir_exp, 'visualization'), t_idx, second_type)


def get_edge_mask(darr, patch_side = 256, radius=None):
    dim_y, dim_x = darr.shape

    if radius is None:
        radius = detect_well_circle(darr)

    edge_side = patch_side * 2.5

    # get image center coordinate
    img_center = np.array([dim_y, dim_x])//2
    radius_in = radius - edge_side//2
    radius_out = radius + edge_side//2

    x = np.linspace(0, dim_x - 1, dim_x)
    y = np.linspace(0, dim_y - 1, dim_y)
    mm = np.meshgrid(y, x)

    # distance from img_center
    dist_sq = (mm[0] - img_center[0]) ** 2 + (mm[1] - img_center[1]) ** 2

    # mask_radius = radius_thresh * np.min(img_center)
    mask_out = (dist_sq <= (radius_out * radius_out)) & (dist_sq >= (radius_in * radius_in))
    return mask_out

def artifact_path_context(context_file):
    '''
        read artifact path from context file
    '''
    with open(context_file, 'r') as file:
        data = json.load(file)
    return data['context']['artifactPath']

def get_disk_mask(darr, radius=None):
    dim_y, dim_x = darr.shape

    if radius is None:
        radius = detect_well_circle(darr)
    
    # get image center coordinate
    img_center = np.array([dim_y, dim_x])//2
    x = np.linspace(0, dim_x - 1, dim_x)
    y = np.linspace(0, dim_y - 1, dim_y)
    mm = np.meshgrid(y, x)

    # distance from img_center
    dist_sq = (mm[0] - img_center[0]) ** 2 + (mm[1] - img_center[1]) ** 2

    mask_out = dist_sq <= (radius * radius)
    return mask_out, radius

def get_uncertain_mask(mask, lower_bound=0.25, upper_bound=0.75):
    uncertain_mask = (mask>lower_bound) & (mask<upper_bound)

    structure_element = np.ones((5, 5), dtype=bool)
    # Erosion
    uncertain_mask = binary_erosion(uncertain_mask, structure_element)
    structure_element = np.ones((15, 15), dtype=bool)
    # Dilation
    uncertain_mask = binary_dilation(uncertain_mask, structure_element)
    
    # import matplotlib.pyplot as plt
    # confidence_mask = 2*np.abs(mask-0.5)
    # plt.figure(figsize=(20,20))
    # plt.imshow(confidence_mask, cmap='jet')  # Set the colormap directly in imshow
    # # Displaying the uncertain_mask on top with a different colormap and some transparency
    # plt.imshow(uncertain_mask, cmap='hot', alpha=0.8) 
    # plt.axis('off')
    # # plt.colorbar()
    # # plt.title('Prediction Confidence')
    # timestr = datetime.utcnow().strftime("D%m%d%H%M%S%f")
    # plt.savefig(f'./debug/prediction_confidence-{timestr}.png')

    return uncertain_mask

def detect_well_circle(darr):
    '''
        detect the well circle
        return the radius of the circle
    '''
    dim_y, dim_x = darr.shape
    # get image center coordinate
    img_center = np.array([dim_y, dim_x])//2
    x = np.linspace(0, dim_x - 1, dim_x)
    y = np.linspace(0, dim_y - 1, dim_y)
    mm = np.meshgrid(y, x)

    # distance from img_center
    dist = np.sqrt((mm[0] - img_center[0]) ** 2 + (mm[1] - img_center[1]) ** 2).astype(int)

    value_median = np.median(darr)
    darr[np.isinf(darr)] = value_median
    diff = darr - value_median

    values = np.bincount(dist.reshape(-1), diff.reshape(-1))
    counts = np.bincount(dist.reshape(-1), np.array([1 for _ in range(dist.size)]))
    values = values/(counts+0.01)
    radius_bright = np.argmax(values[1000:]) +1000
    radius_dark = np.argmin(values[1000:]) +1000

    return (radius_bright + radius_dark)//2

def build_save_patchdataset(patches, save_directory='debug'):
    '''
        build and save patch dataset
    '''
    patch_ds = patch.PatchDataset(patches)
    # generate_thumbnails_single_well(patch_ds, save_directory=save_directory, colormap=colormap, draw_patch=draw_patch)
    generate_thumbnails_single_well(patch_ds, save_directory=save_directory)

def read_artifactpaths(context_files):
    artifact_paths = []
    for context_file in context_files:
        artifact_paths.append(artifact_path_context(context_file))
    
    return artifact_paths

def load_patches_wandb(entity = 'cellino-ml-ninjas',
                       project = '4x_conf_retrain', 
                       artifact_names = ['4x_conf_retrain_training_patchdataset:v5', '4x_conf_retrain_validation_patchdataset:v6']):
    
    # https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/2ltztith/artifacts?workspace=user-swang

    # Fetch the specific artifact using WandB API
    api = wandb.Api()
    patches = []
    for artifact_name in artifact_names:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
        
        # Download the artifact
        artifact_dir = artifact.download()
        json_path = glob.glob(pj(artifact_dir, '*.json'))[0]
        print(f"Downloaded {json_path}")
        patches.extend(patch.PatchDataset.load_from_json(json_path).get_patches())

    return patches

def read_zarr(context_file=None, artifact_path=None, is_local=False, zarr_path=None):
    '''
    read zarr data as numpy array
    '''
    if is_local:
        zarr = CellinoZarr(zarr_path, is_local=True)
        time_slice_index = 0
    else:
        if context_file is not None:
            artifact_path = artifact_path_context(context_file)

        zarr = CellinoZarr(pj(artifact_path['bucket'], artifact_path['blob_path']), is_local=False, 
                           project_id=artifact_path['project'], credential_file=os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
        time_slice_index = artifact_path['time_slice_index']
    
    zarr_root = zarr.get_zarr_root()
    
    # the z_slice is set to 0
    darr = da.from_zarr(zarr_root['0'])[time_slice_index, 0, 0, slice(None), slice(None)].astype(np.float16).compute()

    return (zarr, darr, artifact_path,)


def sample_patches(artifact_infos, search_mask, patch_overlap_ratio=(0.5, 0.5), patch_mask_overlap_ratio_thresh=0.5):
    pt = patcher.SequentialPatcher(artifact_infos, (256, 256), credential_file=os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    pt.sample(search_mask=search_mask, patch_overlap_ratio=patch_overlap_ratio, patch_mask_overlap_ratio_thresh=patch_mask_overlap_ratio_thresh)
    patchdataset = pt.create_patchdataset()

    if patchdataset is None:
        return []
    else:
        # Filtering overlaps
        return patchdataset.get_patches()


def patch_decorator(json_filename):
    def decorator_sample_patches(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            dir_exp = args[0]
            force_redo = args[1] if len(args) > 1 else kwargs.get('force_redo', False)
            print(f'******************************{func.__name__}******************************')
            json_path = pj(dir_exp, json_filename)
            if not force_redo and os.path.exists(json_path):
                patches = patch.PatchDataset.load_from_json(json_path).get_patches()
                return patches
            print('force redo...')
            patches = func(dir_exp)
            
            patch_dataset = patch.PatchDataset(patches)
            patch_dataset.save_to_json(filename=json_path)
            
            return patches
        return wrapper
    return decorator_sample_patches

@patch_decorator('uncertain_patches.json')
def sample_uncertain_patches(dir_exp):

    '''    
    Run the current confluence model (v8) on the wells from plate 1859 (column E) that are not marked as "test" confluence
    wells (column U) that are listed on this sheet: https://docs.google.com/spreadsheets/d/1EqEGWOXfiB1oR6FzuAdpJd4fpbQgpLcftNHL7VYfigo/edit#gid=0
    From the confluence predictions, create a mask of "low confidence" areas (aka areas where the prediction is
    close to 0.5). You also need to make a mask that makes sure NOT
    to sample from the edges of these wells (in these wells the florescence is outside of the well, so we will be
    feeding the model lies). A diskmask should work here. Patch from the product of those two masks.'''

    try:
        df = pd.read_csv(pj(dir_exp, 'plate_1859', 'plate1859.csv'))
        project_brt = 'project-aims-prod'
        project_y = 'cellino-plate-db'
    except:
        print('Make sure to run the confluence model v8 on wells from plate 1859, and generate csv sheet for plate 1859, \
              using aims_confluence, ')

    patches = []
    # import pdb; pdb.set_trace()
    # d3
    for idx in range(1, len(df)):
        print(f'---{idx} out of {len(df)}---')

        brt_artifact_path = {   'datatype': 'zarr',
                                'project': project_brt, 
                                'bucket': df['zarr_path'][idx].split('/', 1)[0], 
                                'blob_path': df['zarr_path'][idx].split('/', 1)[1],
                                'time_slice_index': int(df['time_slice_index'][idx])
                                }

        y_artifact_path = {     'datatype': 'zarr',
                                'project': project_y, 
                                'bucket': df['gt_zarr_path'][idx].split('/', 1)[0], 
                                'blob_path': df['gt_zarr_path'][idx].split('/', 1)[1],
                                'time_slice_index': int(df['gt_time_slice_index'][idx])
                                }
        
        cmap_artifact_path = {  'datatype': 'zarr',
                                'project': '', 
                                'bucket': '', 
                                'blob_path': df['mask_path'][idx],
                                'time_slice_index': 0
                            }      
        
        # import pdb; pdb.set_trace()
        
        # artifact_infos = {'X': {'artifact_path': brt_artifact_path, 'z_slices': [0, 1, 2, 3], 'arrayname': '0'},
        #                     'Y': {'artifact_path': y_artifact_path, 'z_slices': [0], 'arrayname': '0'}}

        artifact_infos = {'X': {'artifact_path': brt_artifact_path, 'z_slices': [0, 1, 2, 3], 'arrayname': '0'},
                    'Y': {'artifact_path': cmap_artifact_path, 'z_slices': [0], 'arrayname': '0'}}

        _, brt_darr, _ = read_zarr(context_file=None, artifact_path = brt_artifact_path)
        disk_mask, radius = get_disk_mask(brt_darr)
        edge_mask = get_edge_mask(brt_darr, patch_side=256, radius=radius)

        _, mask_darr, _ = read_zarr(zarr_path=df['mask_path'][idx], is_local=True)
        uncertain_mask = get_uncertain_mask(mask_darr)

        mask = disk_mask & uncertain_mask & (~edge_mask)
        patches_new = sample_patches(artifact_infos=artifact_infos, search_mask=mask, patch_overlap_ratio=(0.75, 0.75), patch_mask_overlap_ratio_thresh=0.02)
        patches.extend(patches_new)

        break

    return patches



def main():
    dir_exp = '/home/shuhangwang/Documents/Dataset/patch_selection_v9/'


    '''
    test two: plate 1859 involving low confidence
    '''
    patches = sample_uncertain_patches(dir_exp, force_redo=True)

    build_save_patchdataset(patches)



if __name__ == '__main__':
    # main()

    dir_exp = '/home/shuhangwang/Documents/Dataset/visualization/differentiation/'

    generate_thumbnail_zarr(dir_exp=dir_exp)