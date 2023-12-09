import pickle
import numpy as np
import os.path as op


def get_object_pos(dataset, model, img_name, noun_index):
    module_path = op.dirname(__file__)
    flags = pickle.load(open(f'{module_path}/flag/{dataset}/{model}_flag.pkl', 'rb'))
    method = flags[f'{img_name}_{noun_index}']
    if method == 'OD':
        pos_img = np.load(f'{module_path}/result/{dataset}/{model}/OD_result/{img_name}_{noun_index}.npy')
    else:
        pos_img = np.load(f'{module_path}/result/{dataset}/{model}/occlusion_result/{img_name}_{noun_index}.npy')
    return method, pos_img


def get_OD_pos(dataset, model, img_name, noun_index):
    module_path = op.dirname(__file__)
    flags = pickle.load(open(f'{module_path}/flag/{dataset}/{model}_flag.pkl', 'rb'))
    method = flags[f'{img_name}_{noun_index}']
    if method == 'OD':
        return True, np.load(f'{module_path}/result/{dataset}/{model}/OD_result/{img_name}_{noun_index}.npy')
    else:
        return False, ''


def get_occlusion_pos(dataset, model, img_name, noun_index):
    module_path = op.dirname(__file__)
    return np.load(f'{module_path}/result/{dataset}/{model}/occlusion_result/{img_name}_{noun_index}.npy')