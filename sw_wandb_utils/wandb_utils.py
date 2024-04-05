# public package
import wandb
import glob
import re
import os
from os.path import join as pj

# cellino package
from cellino_tiler import patch
from cellino_tiler.patch_utils.helpers import generate_thumbnails

WANDB_ENTITY = 'cellino-ml-ninjas'

def patch_save(dir_exp, patches, filename='patch', type='patchdataset'):
    '''
    This function generate thumbnails and patchdataset json, and upload them to wandb
    '''

    patch_dataset = patch.PatchDataset(patches)

    # save and upload patch datasets
    patch_json_path = pj(dir_exp, 'patches_json', filename+'.json')
    if not os.path.exists(pj(dir_exp, 'patches_json')):
        os.makedirs(pj(dir_exp, 'patches_json'))
    patch_dataset.save_to_json(filename=patch_json_path)
    artifact = wandb.Artifact(filename, type=type)
    artifact.add_file(patch_json_path)
    wandb.log_artifact(artifact)

    # save and upload table of thumbnails
    thumbnail_folder = pj(dir_exp, 'thumbnails_'+filename)
    generate_thumbnails(patch_dataset, save_directory=thumbnail_folder)
    log_patchdataset_tables(thumbnail_folder,
                            patch.PatchDataset.load_from_json(patch_json_path), prefix=filename)


def log_patchdataset_tables(thumbnail_dir, patches: patch.PatchDataset,  prefix=''):
    # If the thumbnail_dir doesn't end in a '/' add it
    if len(thumbnail_dir) > 0 and not thumbnail_dir.endswith('/'):
        thumbnail_dir += '/'

    # Initializations
    file_dict = dict()
    re_well = re.compile(r'\d{6}_[A-Z]\d+')

    feat_names = patches.get_feature_names()

    for p in patches.get_patches():
        artifact_path = p.artifact_infos[feat_names[0]]['artifact_path']
        path_str = artifact_path['bucket'] + '/' + artifact_path['blob_path']
        file_tup = tuple([p.artifact_infos[f_name]['artifact_path']['bucket'] + '/' +
                          p.artifact_infos[f_name]['artifact_path']['blob_path'] for f_name in feat_names])
        time_tup = tuple([p.artifact_infos[f_name]['artifact_path']['time_slice_index'] for f_name in feat_names])
        # If the file is already in the path
        if path_str in file_dict.keys():
            # And the time index is already in the path
            if time_tup in file_dict[path_str]['time_indices'].keys():
                file_dict[path_str]['time_indices'][time_tup].add(file_tup)
            else:
                file_dict[path_str]['time_indices'][time_tup] = set()
                file_dict[path_str]['time_indices'][time_tup].add(file_tup)
        else:
            file_dict[path_str] = {'time_indices': {time_tup: set()}}
            file_dict[path_str]['time_indices'][time_tup].add(file_tup)

    column_names = ['Well Name'] + [f'{f_name} Image' for f_name in feat_names]
    thumb_table = wandb.Table(columns=column_names)
    # For all the file groups
    for v in file_dict.values():
        # For each timepoint and set of tuples
        for t, g in v['time_indices'].items():
            # For each group of images
            for tup in g:
                stitched_for_well_names = ''.join(tup).replace('/', '_')
                found_well_name = re.search(re_well, stitched_for_well_names).group(0)
                well_name = ''
                if found_well_name:
                    well_name = f'CELL-{found_well_name}_t{t[0]}'
                data_to_add = [well_name] + [wandb.Image(thumbnail_dir + file_path.replace('/', '_') + '_t' +
                                                         str(t[idx]) + '.png') for idx, file_path in enumerate(tup)]
                thumb_table.add_data(*data_to_add)
    wandb.log({f'{prefix}-Thumbnail Table': thumb_table})
    return

# def load_patches_wandb(entity = 'cellino-ml-ninjas',
#                        project = '4x_conf_retrain', 
#                        artifact_names = ['4x_conf_retrain_training_patchdataset:v5', '4x_conf_retrain_validation_patchdataset:v6']):
#     '''
#     Load patchdataset from wandb artifact of json file
#     The default loading is for confluence v8
#     https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/2ltztith/artifacts
#     '''

#     api = wandb.Api()
#     patches = []
#     for artifact_name in artifact_names:
#         artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
        
#         # Download the artifact
#         artifact_dir = artifact.download()
#         json_path = glob.glob(pj(artifact_dir, '*.json'))[0]
#         print(f"Downloaded {json_path}")
#         patches.extend(patch.PatchDataset.load_from_json(json_path).get_patches())

#     return patches


def load_patches_wandb(project, 
                       artifact_names,
                       entity = 'cellino-ml-ninjas', download_root=None):
    '''
    This function loads patches from wandb artifact of json file.
    
    For example, with the following settings, it will load the patchdataset for confluence v8, https://cellinobio.wandb.io/cellino-ml-ninjas/4x_conf_retrain/runs/2ltztith/artifacts
    
    project = '4x_conf_retrain'
    artifact_names = ['4x_conf_retrain_training_patchdataset:v5', '4x_conf_retrain_validation_patchdataset:v6']
    entity = 'cellino-ml-ninjas'
    '''

    api = wandb.Api()
    patches = []
    for artifact_name in artifact_names:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
        
        # Download the artifact
        artifact_dir = artifact.download(root=download_root)
        json_path = glob.glob(pj(artifact_dir, '*.json'))[0]
        print(f"Downloaded {json_path}")
        patches.extend(patch.PatchDataset.load_from_json(json_path).get_patches())

    return patches