import os
import glob
import json
from PIL import Image
from os.path import join as pj

def rename_contextfile_platewell(folder):
    '''
        This function is to rename the gaia context file by plate & well name & time_slice_index
    '''
    # import pdb; pdb.set_trace()
    filenames = glob.glob(pj(folder, '*.json'))
    for fn in filenames:
        with open(fn, 'r') as file:
            data = json.load(file)
        plate = data['context']['plate']['name']
        well = data['context']['well']['position']
        time_slice_index = data['context']['artifactPath']['time_slice_index']
        fn_new = pj(folder, f'{plate}-{well}-{time_slice_index}.json')
        os.rename(fn, fn_new)

def save_arr_scaled_image(arr, img_path, scale_factor=0.125):
    '''
        This function saves numpy array to a scaled image
    '''
    new_size = (int(arr.shape[-2] * scale_factor), int(arr.shape[-1] * scale_factor))
    img = Image.fromarray(arr).resize(new_size).convert('RGB')
    img.save(img_path)


if __name__=='__main__':
    # folder = '/home/shuhangwang/Documents/Dataset/anomaly/contamination/gaia-context'
    # folder = '/home/shuhangwang/Documents/Dataset/conf_v10/bumpy_contexts'
    folder = '/home/shuhangwang/Documents/Dataset/conf_v10/test_contexts'

    rename_contextfile_platewell(folder)