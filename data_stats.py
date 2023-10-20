import collections
from openpyxl import Workbook
import pandas as pd
import numpy as np
from cellino_tiler import patch
import os
pj = os.path.join

patch_dt = collections.defaultdict(int)
def update_well_patch_map(patch_json_path):
    json_filename = patch_json_path.split('/')[-1]
    if json_filename in ['patches_for_v8_training.json', 
                         'patches_for_v9_test_edges.json',
                         'patches_for_v9_test_crystals.json',
                         'patches_for_v8_validation.json']:
        return
    patches = patch.PatchDataset.load_from_json(patch_json_path).get_patches()
    for p in patches:
        artifact_path = p.artifact_infos['X']['artifact_path']
        brt_path = pj(artifact_path['bucket'], artifact_path['blob_path'])
        time_index = artifact_path['time_slice_index']
        plate_well_tidx_name = '_'.join(brt_path.split('/')[2:4]) + '_' + str(time_index)
        key = (plate_well_tidx_name, json_filename)
        patch_dt[key] += 1

def get_wells_info_from_patches(patch_json_path):
    '''
    load patches, return the well lists, well number, and patches number
    '''

    brt_path_set = set()
    brt_path_tidx_set = set()
    well_set = set()
    well_tidx_set = set()
    patches = patch.PatchDataset.load_from_json(patch_json_path).get_patches()
    for p in patches:
        artifact_path = p.artifact_infos['X']['artifact_path']
        brt_path = pj(artifact_path['bucket'], artifact_path['blob_path'])
        time_index = artifact_path['time_slice_index']
        brt_path_set.add(brt_path)
        brt_path_tidx_set.add(brt_path + '_' + str(time_index))
        plate_well_name = '_'.join(brt_path.split('/')[2:4])
        well_set.add(plate_well_name)
        plate_well_tidx_name = plate_well_name + '_' + str(time_index)
        well_tidx_set.add(plate_well_tidx_name)

    
    return len(patches), sorted(list(brt_path_set)), sorted(list(brt_path_tidx_set)), sorted(list(well_set)), sorted(list(well_tidx_set))

def get_wells_info_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    brt_path_set = set()
    brt_path_tidx_set = set()
    well_set = set()
    well_tidx_set = set()
    for i in range(len(df)):
        brt_path = df['brt'].iloc[i]
        time_index = df['t_index'].iloc[i]
        brt_path_set.add(brt_path)
        brt_path_tidx_set.add(brt_path + '_' + str(time_index))
        plate_well_name = '_'.join(brt_path.split('/')[2:4])
        well_set.add(plate_well_name)
        well_tidx_set.add(plate_well_name + '_' + str(time_index))

    # well_set = set(df['brt'])
    # well_tidx_set = set(df['brt'] + df['t_index'].astype(str))

    return sorted(list(brt_path_set)), sorted(list(brt_path_tidx_set)), sorted(list(well_set)), sorted(list(well_tidx_set))

def quali_quanti_in_train_val(df, col = 'well_tidx_list'):
    # df row 0: v9_train
    # df row 1: v9_validation
    # df row 2: quali_test
    # df row 3: quanti_test
    test_set = set(df[col].iloc[2] + df[col].iloc[3])
    train_val_set = set(df[col].iloc[1] + df[col].iloc[0])
    print(test_set.intersection(train_val_set))

def confluence_v9_analysis():

    dir_exp = '/home/shuhangwang/Documents/Dataset/conf_v9/'
    conf_v9_data_path = pj(dir_exp, 'conf_v9_data_distribution.csv')
    df = pd.read_csv(conf_v9_data_path)
    for col in ['well_list', 'well_num', 'well_tidx_list', 'well_tidx_num', 'brt_path_list', 'brt_path_num', 'brt_path_tidx_list', 'brt_path_tidx_num']:
        df[col] = np.nan
    
    for i in range(0 ,len(df)):
        print(i, '...')
        # get well info from patches
        if df['patch_json_local_path'].iloc[i] is not np.nan:        
            patch_num, brt_path_list, brt_path_tidx_list, well_list, well_tidx_list = get_wells_info_from_patches(df['patch_json_local_path'].iloc[i])
            df['patch_num'].iloc[i] = patch_num
            update_well_patch_map(df['patch_json_local_path'].iloc[i])
        # get well info from csv file directly
        elif df['csv_local_path'].iloc[i] is not np.nan:
            brt_path_list, brt_path_tidx_list, well_list, well_tidx_list = get_wells_info_from_csv(df['csv_local_path'].iloc[i])

        df['well_list'].iloc[i] = well_list
        df['well_num'].iloc[i] = len(well_list)
        df['well_tidx_list'].iloc[i] = well_tidx_list
        df['well_tidx_num'].iloc[i] = len(well_tidx_list)
        
        df['brt_path_list'].iloc[i] = brt_path_list
        df['brt_path_num'].iloc[i] = len(brt_path_list)
        df['brt_path_tidx_list'].iloc[i] = brt_path_tidx_list
        df['brt_path_tidx_num'].iloc[i] = len(brt_path_tidx_list)

    columns_to_convert = ['well_num', 'well_tidx_num', 'brt_path_num', 'brt_path_tidx_num']
    df[columns_to_convert] = df[columns_to_convert].astype(int)
    # df.to_csv('conf_v9_data_distribution_analysis.csv')

    # Convert the dictionary to a list of tuples
    patch_dt_list = list(patch_dt.items())
    patch_dt_list = [(e0[0], e0[1], e1) for e0, e1 in patch_dt_list]
    # Create a DataFrame from the list
    df_patch = pd.DataFrame(patch_dt_list, columns=['well', 'data set', 'num of patches'])
    df_patch = df_patch.groupby('well').agg({
    'data set': lambda x: ', '.join(x),
    'num of patches': 'sum'}).reset_index()
    
    df_patch = df_patch.sort_values(by='num of patches', ascending=False)
    df_patch.reset_index(drop=True, inplace=True)

    result_path = pj(dir_exp, 'conf_v9_data_distribution_analysis.xlsx')
    if os.path.exists(result_path):
        os.remove(result_path)
    workbook = Workbook()
    workbook.save(result_path)

    with pd.ExcelWriter(result_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name='wells')
        df_patch.to_excel(writer, sheet_name='patches_per_well')

    # check if quali and quanti wells were included in train and val
    quali_quanti_in_train_val(df)

if __name__=='__main__':
    confluence_v9_analysis()