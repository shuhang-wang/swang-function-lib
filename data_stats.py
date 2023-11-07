import collections
from openpyxl import Workbook
import pandas as pd
import numpy as np
from cellino_tiler import patch
import json
import os
import ast
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
        # plate_well_tidx_name = '_'.join(brt_path.split('/')[2:4]) + '_' + str(time_index)
        plate_well_tidx_name =  '-'.join(['CELL'] + brt_path.split('/')[2:4] + ['t'+str(time_index)])
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
        # plate_well_name = '_'.join(brt_path.split('/')[2:4])
        plate_well_name = '-'.join(['CELL'] + brt_path.split('/')[2:4])
        well_set.add(plate_well_name)
        # plate_well_tidx_name = plate_well_name + '_' + str(time_index)
        plate_well_tidx_name = plate_well_name + '-t' + str(time_index)
        well_tidx_set.add(plate_well_tidx_name)

    
    return len(patches), sorted(list(brt_path_set)), sorted(list(brt_path_tidx_set)), sorted(list(well_set)), sorted(list(well_tidx_set))

def get_patch_count_cat(patch_json_path):
    '''
    load patches, get patch count for each category
    '''
    dt = collections.defaultdict(int)
    patches = patch.PatchDataset.load_from_json(patch_json_path).get_patches()
    for p in patches:
        # dt[p.attributes] += 1
        if len(p.attributes)>0:
            stop = True
            import pdb; pdb.set_trace()

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
        # plate_well_name = '_'.join(brt_path.split('/')[2:4])
        plate_well_name = '-'.join(['CELL'] + brt_path.split('/')[2:4])
        well_set.add(plate_well_name)
        # well_tidx_set.add(plate_well_name + '_' + str(time_index))
        well_tidx_set.add(plate_well_name + '-t' + str(time_index))

    # well_set = set(df['brt'])
    # well_tidx_set = set(df['brt'] + df['t_index'].astype(str))

    return sorted(list(brt_path_set)), sorted(list(brt_path_tidx_set)), sorted(list(well_set)), sorted(list(well_tidx_set))

def confluence_v9_patch_histogram():
    '''
    conf_v9_data_distribution.csv was created manually, including jason path and csv for each group of data
    '''

    dir_exp = '/home/shuhangwang/Documents/Dataset/conf_v9/'
    conf_v9_data_path = pj(dir_exp, 'conf_v9_data_distribution.csv')
    df = pd.read_csv(conf_v9_data_path)
    for col in ['well_list', 'well_num', 'well_tidx_list', 'well_tidx_num', 'brt_path_list', 'brt_path_num', 'brt_path_tidx_list', 'brt_path_tidx_num']:
        df[col] = np.nan
    
    for i in range(0 ,len(df)):
        print(i, '...', df['patch_json_local_path'].iloc[i])
        # get well info from patches
        if df['patch_json_local_path'].iloc[i] is not np.nan:        
            get_patch_count_cat(df['patch_json_local_path'].iloc[i])

def confluence_v9_analysis():
    '''
    conf_v9_data_distribution.csv was created manually, including json path and csv for each group of data
    '''

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
    df_patch = pd.DataFrame(patch_dt_list, columns=['well', 'data set', 'patch_count'])
    df_patch = df_patch.groupby('well').agg({
    'data set': lambda x: ', '.join(x),
    'patch_count': 'sum'}).reset_index()
    
    df_patch = df_patch.sort_values(by='patch_count', ascending=False)
    df_patch.reset_index(drop=True, inplace=True)

    result_path = pj(dir_exp, 'conf_v9_data_distribution_analysis.xlsx')
    if os.path.exists(result_path):
        os.remove(result_path)
    workbook = Workbook()
    workbook.save(result_path)

    with pd.ExcelWriter(result_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name='wells')
        df_patch.to_excel(writer, sheet_name='patches_per_well')

def get_wells_info_pluri_csv(dir_exp, tfr_list):
    tfr_list = [f.split('/')[-1] for f in tfr_list]
    df = pd.DataFrame(columns=['well', 'data set', 'PLURI_patch_count', 'DIFF_patch_count', 'EDGE_patch_count'])
    
    for art_name in ['TFRECORD-manual_pluri_ibidi', 'TFRECORD-manual_pluri_ibidi_edge']:
        df_ref  = pd.read_csv(pj(dir_exp, art_name+'_file_table.csv'))
        for file_name in tfr_list:
            print(art_name, file_name, '......')
            new_row = {'well': '-'.join(file_name.split('-')[:-1]), 
                    'data set': art_name, 
                    'PLURI_patch_count': 0,
                    'DIFF_patch_count': 0,
                    'EDGE_patch_count': 0}
            
            tfr_type = file_name.split('-')[-1][:-4]
            if art_name=='TFRECORD-manual_pluri_ibidi' and tfr_type in ['PLURI', 'DIFF']:
                if tfr_type=='PLURI':
                    ridx = df_ref.index[df_ref['PLURI_tfrecord'] == file_name].tolist()[0]
                    new_row['PLURI_patch_count'] = df_ref['PLURI_patch_count'].iloc[ridx]
                else:
                    ridx = df_ref.index[df_ref['DIFF_tfrecord'] == file_name].tolist()[0]
                    new_row['DIFF_patch_count'] = df_ref['DIFF_patch_count'].iloc[ridx]
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            if art_name=='TFRECORD-manual_pluri_ibidi_edge' and tfr_type=='EDGE':
                ridx = df_ref.index[df_ref['EDGE_tfrecord'] == file_name].tolist()[0]
                new_row['EDGE_patch_count'] = df_ref['EDGE_patch_count'].iloc[ridx]
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df

def get_wells_info_pluri(dir_exp, train_tfr_list, val_tfr_list):

    def tfr2well(tfr_list):
        tfr_list = [f.split('/')[-1] for f in tfr_list]        
        tfr_list = ['-'.join(file_name.split('-')[:-1]) for file_name in tfr_list]
        return tfr_list
                    
    # train & val
    train_well_tidx_list = tfr2well(train_tfr_list)
    val_well_tidx_list = tfr2well(val_tfr_list)

    # test
    # this table is from 4x_pluripotency_tf/evaluation_script2.py
    TEST_DATA = [{'plate_barcode': 'CELL-001860', 'well_name': 'B4'}, 
            {'plate_barcode': 'CELL-001860', 'well_name': 'B5'}, 
            {'plate_barcode': 'CELL-001860', 'well_name': 'C5'}, 
            {'plate_barcode': 'CELL-001859', 'well_name': 'A2'}, 
            {'plate_barcode': 'CELL-001831', 'well_name': 'A3'}]
    df_ref = pd.read_csv(pj(dir_exp, 'gt_pluri_4x_manual_file_table.csv'))
    test_well_tidx_list = []
    for data in TEST_DATA:
        idx = df_ref.loc[(df_ref['plate_barcode'] == data['plate_barcode']) & (df_ref['well_name'] == data['well_name'])].index
        plate_well_tidx_name = '-'.join([data['plate_barcode'], data['well_name'], 't'+ str(df_ref['time_slice_index'][idx[0]])])
        test_well_tidx_list.append(plate_well_tidx_name)

    return pd.DataFrame([{'data_sets': 'pluri_train', 'well_tidx_list': train_well_tidx_list},
                         {'data_sets': 'pluri_val', 'well_tidx_list': val_well_tidx_list},
                         {'data_sets': 'pluri_test', 'well_tidx_list': test_well_tidx_list}])

def pluripotency_v1_analysis():
    '''
    The training and validation sets were generated by:
    tr_list, n_tr, val_list, n_val = get_train_val_list(art_names, feat_names, test_wellinfo, nval=1) @ 4x_pluripotency_tf/pluritrainer_refactor_v1.py
    which are from 
    (TFRECORD-manual_pluri_ibidi:v0, file_table.csv) --> TFRECORD-manual_pluri_ibidi_file_table.csv
    (TFRECORD-manual_pluri_ibidi_edge:v0, file_table.csv) --> TFRECORD-manual_pluri_ibidi_edge_file_table.csv

    The test set is from:
    (gt_pluri_4x_manual:v1, file_table.csv) --> gt_pluri_4x_manual_file_table.csv
    '''
    
    dir_exp = '/home/shuhangwang/Documents/Dataset/pluri/train_val_v1/'

    

    with open(pj(dir_exp, 'train_tfr_list.json'), 'r') as file:
        train_list = json.load(file)
    with open(pj(dir_exp, 'val_tfr_list.json'), 'r') as file:
        val_list = json.load(file)

    # well level: train & validation & test
    df_well = get_wells_info_pluri(dir_exp, train_list, val_list)

    # patch level: train & validation
    df_info1 = get_wells_info_pluri_csv(dir_exp, train_list)
    df_info2 = get_wells_info_pluri_csv(dir_exp, val_list)
    df_patch = pd.concat([df_info1, df_info2], ignore_index=True)

    df_patch = df_patch.groupby('well').agg({
        'data set': lambda x: ', '.join(x),
        'PLURI_patch_count': 'sum',
        'DIFF_patch_count': 'sum',
        'EDGE_patch_count': 'sum'}).reset_index()
    df_patch['patch_count'] = df_patch[['PLURI_patch_count', 'DIFF_patch_count', 'EDGE_patch_count']].sum(axis=1)

    df_patch = df_patch.sort_values(by='patch_count', ascending=False)
    df_patch.reset_index(drop=True, inplace=True)

    result_path = pj(dir_exp, 'pluri_v1_data_distribution_analysis.xlsx')
    if os.path.exists(result_path):
        os.remove(result_path)
    workbook = Workbook()
    workbook.save(result_path)

    with pd.ExcelWriter(result_path, engine='openpyxl', mode='a') as writer:
        df_well.to_excel(writer, sheet_name='wells')
        df_patch.to_excel(writer, sheet_name='patches_per_well')

    # import pdb; pdb.set_trace()

def add_cell_ground_truth_flag(df_merged):
    '''
    Cell ID - Centralized Ground Truth Log - 4X Confluence Ground Truth.csv is from:
    https://docs.google.com/spreadsheets/d/1EqEGWOXfiB1oR6FzuAdpJd4fpbQgpLcftNHL7VYfigo/edit#gid=0
    '''
    # add a column to indicate if ground truth of cells exists
    df_gt = pd.read_csv('/home/shuhangwang/Documents/Dataset/ground_truth/Cell ID - Centralized Ground Truth Log - 4X Confluence Ground Truth.csv')
    well_tidx_set = set(df_gt['plate_barcode']+'-'+df_gt['well_name']+'-t'+df_gt['time_slice_index'].astype('str'))
    df_merged['cell_GT'] = df_merged['well'].isin(well_tidx_set)
    df_merged = df_merged.sort_values(by='cell_GT', ascending=False)
    return df_merged

def add_negative_category(df_merged):
    '''
    add a column to show negative categories, for manually curated all negative patches

    4XConfluenceRetrain_NegativePatches_Manual.xlsx is from
    https://docs.google.com/spreadsheets/d/1ERFB9gsq1Y3KVTFv_mDodFZmNTMOLBdwg2_Nn_67Oxc/edit#gid=0
    '''

    
    file_path = '/home/shuhangwang/Documents/Dataset/ground_truth/4XConfluenceRetrain_NegativePatches_Manual.xlsx'
    df_negative = pd.read_excel(file_path, sheet_name='Sheet1',  dtype={'plate_barcode': str, 'brt_timeslice_ind':str})
    df_negative['plate_barcode'] = df_negative['plate_barcode'].str.zfill(6)
    df_negative['well_tidx'] = df_negative.apply(
        lambda row: '-'.join(['CELL', row['plate_barcode'], row['well'], 't'+row['brt_timeslice_ind']]), axis=1)

    print('Wells of negative patches were not included in the train-val sets:', set(df_negative['well_tidx']) - set(df_merged['well']))

    def get_negative_categories(row):
        try:
            well_value = row['well']
            negative_cats = list(set(df_negative[df_negative['well_tidx'] == well_value]['Class']))
            if len(negative_cats)==0:
                return None
            return negative_cats
        except Exception as e:
            print("Error with row:", row)
            print("Exception:", e)
            return None

    df_merged['negative_cats'] = df_merged.apply(get_negative_categories, axis=1)

    return df_merged

def add_positive_category(df_merged):
    '''
    add a column to show positive categories, for manually curated all positive patches

    4XConfluenceRetrain_PositivePatches_Manual.xlsx is from
    https://docs.google.com/spreadsheets/d/1D1xM7YrmBOHgKxs6vRX9XjhHXNDjGvFsoMw0kO_4was/edit#gid=0
    '''

    
    file_path = '/home/shuhangwang/Documents/Dataset/ground_truth/4XConfluenceRetrain_PositivePatches_Manual.xlsx'
    df_positive = pd.read_excel(file_path, sheet_name='Sheet1',  dtype={'plate_barcode': str, 'brt_timeslice_ind':str})
    df_positive['plate_barcode'] = df_positive['plate_barcode'].str.zfill(6)
    df_positive['well_tidx'] = df_positive.apply(
        lambda row: '-'.join(['CELL', row['plate_barcode'], row['well'], 't'+row['brt_timeslice_ind']]), axis=1)

    print('Wells of positive patches were not included in the train-val sets:', set(df_positive['well_tidx']) - set(df_merged['well']))

    def get_positive_categories(row):
        try:
            well_value = row['well']
            positive_cats = list(set(df_positive[df_positive['well_tidx'] == well_value]['Class']))
            if len(positive_cats)==0:
                return None
            return positive_cats
        except Exception as e:
            print("Error with row:", row)
            print("Exception:", e)
            return None

    df_merged['positive_cats'] = df_merged.apply(get_positive_categories, axis=1)

    return df_merged

def patch_level_comparion():
    '''
    confluence v9 & pluripotency v1
    '''

    file_path_conf = '/home/shuhangwang/Documents/Dataset/conf_v9/conf_v9_data_distribution_analysis.xlsx'
    df_conf = pd.read_excel(file_path_conf, sheet_name='patches_per_well')
    

    file_path_pluri = '/home/shuhangwang/Documents/Dataset/pluri_v1/pluri_v1_data_distribution_analysis.xlsx'
    df_pluri = pd.read_excel(file_path_pluri, sheet_name='patches_per_well')


    df_merged = df_conf.merge(df_pluri, on='well', how='outer')
    df_merged = df_merged.rename(columns={
        'Unnamed: 0_x': 'order_conf',
        'data set_x': 'data set_conf',
        'patch_count_x': 'patch_count_conf',
        'Unnamed: 0_y': 'order_pluri',
        'data set_y': 'data set_pluri',
        'patch_count_y': 'patch_count_pluri'
    })

    # Convert all columns with numbers to int
    numeric_cols = df_merged.select_dtypes(include=['number']).columns

    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0).astype(int)

    df_merged = add_cell_ground_truth_flag(df_merged)
    df_merged = add_negative_category(df_merged)
    df_merged = add_positive_category(df_merged)

    df_merged.to_csv('/home/shuhangwang/Documents/Dataset/conf_v10_pluri_v2/patches_per_well_comparision.csv')

def well_level_comparion():
    '''
    confluence v9 & pluripotency v1
    '''

    file_path_conf = '/home/shuhangwang/Documents/Dataset/conf_v9/conf_v9_data_distribution_analysis.xlsx'
    df_conf = pd.read_excel(file_path_conf, sheet_name='wells')
    # df row 0: v9_train
    # df row 1: v9_validation
    # df row 2: quali_test
    # df row 3: quanti_test
    trval_wells_conf = set(ast.literal_eval(df_conf['well_tidx_list'].iloc[0]) + ast.literal_eval(df_conf['well_tidx_list'].iloc[1]))
    quali_wells_conf = set(ast.literal_eval(df_conf['well_tidx_list'].iloc[2]))
    quanti_wells_conf = set(ast.literal_eval(df_conf['well_tidx_list'].iloc[3]))



    file_path_pluri = '/home/shuhangwang/Documents/Dataset/pluri_v1/pluri_v1_data_distribution_analysis.xlsx'
    df_pluri = pd.read_excel(file_path_pluri, sheet_name='wells')
    # df row 0: v1_train
    # df row 1: v1_validation
    # df row 2: v1_test
    trval_wells_pluri = set(ast.literal_eval(df_pluri['well_tidx_list'].iloc[0]) + ast.literal_eval(df_pluri['well_tidx_list'].iloc[1]))
    test_wells_pluri = set(ast.literal_eval(df_pluri['well_tidx_list'].iloc[2]))

    # import pdb; pdb.set_trace()
    
    print('... conf quali-test:', quali_wells_conf)
    print('conf quali-test in conf train-val:', quali_wells_conf.intersection(trval_wells_conf))
    print('conf quali-test in pluri train-val:', quali_wells_conf.intersection(trval_wells_pluri))
    print('conf quali-test in pluri test:', quali_wells_conf.intersection(test_wells_pluri))

    print('... conf quanti-test:', quanti_wells_conf)
    print('conf quanti-test in conf train-val:', quanti_wells_conf.intersection(trval_wells_conf))
    print('conf quanti-test in pluri train-val:', quanti_wells_conf.intersection(trval_wells_pluri))
    print('conf quanti-test in pluri test:', quanti_wells_conf.intersection(test_wells_pluri))

    print('... pluri test:', test_wells_pluri)
    print('pluri test in conf train-val:', test_wells_pluri.intersection(trval_wells_conf))
    print('pluri test in pluri train-val:', test_wells_pluri.intersection(trval_wells_pluri))

def consistency_check():
    '''
    check the consistency within each or between confluence and pluripotency data distribution
    '''
    
    # well_level_comparion()

    patch_level_comparion()

if __name__=='__main__':
    # confluence_v9_analysis()
    # pluripotency_v1_analysis()
    # consistency_check()

    confluence_v9_patch_histogram()
    

