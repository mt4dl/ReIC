import numpy as np
import os
case_generation_dir = os.path.join(os.getcwd(), '..', 'Follow_up_Test_Cases_Generation')


def get_mask_img_diverse(dataset, model, mr, img_name, batch):
    return np.load(f'{case_generation_dir}/dynamic_{mr}_mask/{dataset}/dynamic_{mr}_{model}_mask/{batch}/{img_name}_mask.npy')


def get_mask_img_without_diverse(dataset, model, mr, img_name, batch):
    return np.load(f'{case_generation_dir}/dynamic_{mr}_mask/{dataset}/dynamic_{mr}_{model}_without_diverse_mask/{batch}/{img_name}_mask.npy')


def get_mask_img_without_score(dataset, model, mr, img_name, batch):
    return np.load(f'{case_generation_dir}/dynamic_{mr}_mask/{dataset}/dynamic_{mr}_{model}_without_score_mask/{batch}/{img_name}_mask.npy')

