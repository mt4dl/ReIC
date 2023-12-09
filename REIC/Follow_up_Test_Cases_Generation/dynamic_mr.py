import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse
import json
import math
import os.path as op

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dynamic_mr_function import calculate_crop_average, calculate_score, choose_next
from functions.rotate_function import img_rotate, get_mask_img as rotate_mask, get_largest_rec
from Visual_Caption_Alignment.get_result import get_object_pos

import random
random.seed(10)
np.random.seed(10)


'''
This is the code for dynamic crop
'''
def get_crop_start_point(shape, crop_threshold, c):
    i_shape = int(shape[0] * (1 - crop_threshold[0]))
    j_shape = int(shape[1] * (1 - crop_threshold[0]))
    order_len = math.ceil(i_shape * j_shape / c)
    for i in range(0, i_shape * j_shape, order_len):
        index = min(i + int(order_len / 2), i_shape * j_shape - 1)
        yield divmod(index, j_shape)


def dynamic_crop(dataset, model, threshold1=0.2, threshold2=0.9, c=10, k=3):
    dynamic_crop_dir = op.join('follow_up_images', dataset)
    dynamic_crop_mask_dir = op.join('dynamic_crop_mask', dataset, f'dynamic_crop_{model}_mask')
    os.makedirs(dynamic_crop_dir, exist_ok=True)
    os.makedirs(dynamic_crop_mask_dir, exist_ok=True)

    for batch_index in range(k):
        os.makedirs(op.join(dynamic_crop_dir, f'dynamic_crop_{model}_{batch_index}'), exist_ok=True)
        os.makedirs(op.join(dynamic_crop_mask_dir, f'{batch_index}'), exist_ok=True)

    # load image features
    features = pd.read_csv(f'object_areas/{model}_{dataset}.csv', index_col=0, converters={'feature': json.loads})
    for index, value in tqdm(features.iterrows(), total=features.shape[0]):
        img_name = value[0]
        feature = value[1]

        # get image and mask
        image_path = op.join('..', 'data', 'images', dataset, f'{img_name}.jpg')
        image = cv2.imread(image_path)
        shape = (image.shape[0], image.shape[1])

        # calculate crop threshold
        min_crop_rate = calculate_crop_average(shape, feature)
        crop_threshold = (min_crop_rate, 1.0)

        # get noun positions of this image
        noun_positions = []
        for noun in feature:
            noun_index = noun['index']
            _, noun_position = get_object_pos(dataset, model, img_name, noun_index)
            noun_positions.append(noun_position)

        # get candidate masks
        candidate_mask = []
        for start_i, start_j in get_crop_start_point(shape, crop_threshold, c):
            for end_i in range(start_i + int(shape[0] * crop_threshold[0]),
                               min(shape[0], start_i + int(shape[0] * crop_threshold[1]))):
                end_j = int((end_i - start_i) / shape[0] * shape[1] + start_j)
                if end_j >= shape[1]:
                    break
                candidate_mask.append((start_i, start_j, end_i, end_j))
        random.shuffle(candidate_mask)

        # choose k masks
        current_pos = None
        current_index = 0
        min_score = 10000
        max_score = -10000

        # calculate scores and choose first pos
        mask_scores = []
        diverse_scores = []
        for mask_index in range(len(candidate_mask)):
            pos = candidate_mask[mask_index]
            mask_img = np.zeros(shape, dtype=np.int)
            mask_img[pos[0]:pos[2], pos[1]:pos[3]] = 1
            score = calculate_score(noun_positions, mask_img, threshold1, threshold2)
            mask_scores.append(score)
            if score >= max_score:
                max_score = score
                current_pos = pos
                current_index = mask_index
            if score <= min_score:
                min_score = score
            diverse_scores.append(1.0)

        candidates = pd.DataFrame({'pos': candidate_mask, 'mask_score': mask_scores, 'diverse_score': diverse_scores})
        candidates['mask_score'] = candidates['mask_score'].map(
            lambda x: 0.5 if max_score == min_score else (x - min_score) / (max_score - min_score))

        choose_indexes = [current_index]
        choose_positions = [current_pos]

        for _ in range(1, k):
            current_index, current_pos = choose_next(candidates, choose_indexes, current_pos, 'crop')
            choose_indexes.append(current_index)
            choose_positions.append(current_pos)

        # save image and mask
        for batch_index in range(k):
            pos = choose_positions[batch_index]
            mask_img = np.zeros(shape, dtype=np.int)
            mask_img[pos[0]:pos[2], pos[1]:pos[3]] = 1

            np.save(op.join(dynamic_crop_mask_dir, f'{batch_index}', f'{img_name}_mask.npy'), mask_img)
            dynamic_crop_image = image[pos[0]:pos[2], pos[1]:pos[3]]
            cv2.imwrite(op.join(dynamic_crop_dir, f'dynamic_crop_{model}_{batch_index}', f'{img_name}.jpg'), dynamic_crop_image)


'''
This is the code for dynamic stretch
'''
def get_stretch_start(shape, stretch_threshold, c):
    j_shape = int(shape[1] * (1 - stretch_threshold))
    order_len = math.ceil(j_shape / c)
    for i in range(0, j_shape, order_len):
        index = min(i + int(order_len / 2), j_shape - 1)
        yield index


def dynamic_stretch(dataset, model, threshold1=0.2, threshold2=0.9, c=10, k=3):
    dynamic_stretch_dir = op.join('follow_up_images', dataset)
    dynamic_stretch_mask_dir = op.join('dynamic_stretch_mask', dataset, f'dynamic_stretch_{model}_mask')
    os.makedirs(dynamic_stretch_dir, exist_ok=True)
    os.makedirs(dynamic_stretch_mask_dir, exist_ok=True)

    for batch_index in range(k):
        os.makedirs(op.join(dynamic_stretch_dir, f'dynamic_stretch_{model}_{batch_index}'), exist_ok=True)
        os.makedirs(op.join(dynamic_stretch_mask_dir, f'{batch_index}'), exist_ok=True)


    features = pd.read_csv(f'object_areas/{model}_{dataset}.csv', index_col=0, converters={'feature': json.loads})
    for index, value in tqdm(features.iterrows(), total=features.shape[0]):
        img_name = value[0]
        feature = value[1]

        # get image and mask
        image_path = op.join('..', 'data', 'images', dataset,  f'{img_name}.jpg')
        image = cv2.imread(image_path)
        shape = (image.shape[0], image.shape[1])

        stretch_threshold = 0.6

        # get noun positions of this image
        noun_positions = []
        for noun in feature:
            noun_index = noun['index']
            _, noun_position = get_object_pos(dataset, model, img_name, noun_index)
            noun_positions.append(noun_position)

        candidate_mask = []
        for start in get_stretch_start(shape, stretch_threshold, c):
            for end in range(start + int(shape[0] * stretch_threshold), shape[0]):
                candidate_mask.append((start, end))

        random.shuffle(candidate_mask)

        # choose k masks
        current_pos = None
        current_index = 0
        min_score = 10000
        max_score = -10000

        # calculate scores and choose first pos
        mask_scores = []
        diverse_scores = []
        for mask_index in range(len(candidate_mask)):
            pos = candidate_mask[mask_index]
            mask_img = np.zeros(shape, dtype=np.int)
            mask_img[pos[0]:pos[1], :] = 1
            score = calculate_score(noun_positions, mask_img, threshold1, threshold2)
            mask_scores.append(score)
            if score >= max_score:
                max_score = score
                current_pos = pos
                current_index = mask_index
            if score <= min_score:
                min_score = score
            diverse_scores.append(1.0)

        candidates = pd.DataFrame({'pos': candidate_mask, 'mask_score': mask_scores, 'diverse_score': diverse_scores})
        candidates['mask_score'] = candidates['mask_score'].map(
            lambda x: 0.5 if max_score == min_score else (x - min_score) / (max_score - min_score))

        choose_indexes = [current_index]
        choose_positions = [current_pos]

        for _ in range(1, k):
            current_index, current_pos = choose_next(candidates, choose_indexes, current_pos, 'stretch')
            choose_indexes.append(current_index)
            choose_positions.append(current_pos)

        # save image and mask
        for batch_index in range(k):
            pos = choose_positions[batch_index]
            mask_img = np.zeros(shape, dtype=np.int)
            mask_img[pos[0]:pos[1], :] = 1

            np.save(op.join(dynamic_stretch_mask_dir, f'{batch_index}', f'{img_name}_mask.npy'), mask_img)
            dynamic_stretch_image = image[pos[0]:pos[1], :]
            dynamic_stretch_image = cv2.resize(dynamic_stretch_image, (image.shape[1], image.shape[0]))
            cv2.imwrite(op.join(dynamic_stretch_dir, f'dynamic_stretch_{model}_{batch_index}', f'{img_name}.jpg'), dynamic_stretch_image)



'''
This is the code for dynamic rotate method
'''
def dynamic_rotate(dataset, model, threshold1=0.2, threshold2=0.9, c=30, k=3):
    dynamic_rotate_dir = op.join('follow_up_images', dataset)
    dynamic_rotate_mask_dir = op.join('dynamic_rotate_mask', dataset, f'dynamic_rotate_{model}_mask')
    os.makedirs(dynamic_rotate_dir, exist_ok=True)
    os.makedirs(dynamic_rotate_mask_dir, exist_ok=True)

    for batch_index in range(k):
        os.makedirs(op.join(dynamic_rotate_dir, f'dynamic_rotate_{model}_{batch_index}'), exist_ok=True)
        os.makedirs(op.join(dynamic_rotate_mask_dir, f'{batch_index}'), exist_ok=True)


    features = pd.read_csv(f'object_areas/{model}_{dataset}.csv', index_col=0, converters={'feature': json.loads})
    for index, value in tqdm(features.iterrows(), total=features.shape[0]):
        img_name = value[0]
        feature = value[1]

        # get image and mask
        image_path = op.join('..', 'data', 'images', dataset, f'{img_name}.jpg')
        image = cv2.imread(image_path)
        shape = (image.shape[0], image.shape[1])

        # get noun positions of this image
        noun_positions = []
        for noun in feature:
            noun_index = noun['index']
            _, noun_position = get_object_pos(dataset, model, img_name, noun_index)
            noun_positions.append(noun_position)

        candidate_angles = range(-c, c+1)

        # choose k angles
        current_angle = None
        current_index = 0
        min_score = 10000
        max_score = -10000

        # calculate scores and choose first pos
        mask_scores = []
        diverse_scores = []
        for index in range(len(candidate_angles)):
            angle = candidate_angles[index]
            mask_img = rotate_mask(shape, angle)
            score = calculate_score(noun_positions, mask_img, threshold1, threshold2)
            mask_scores.append(score)
            if score >= max_score:
                max_score = score
                current_angle = angle
                current_index = index
            if score <= min_score:
                min_score = score
            diverse_scores.append(1.0)

        candidates = pd.DataFrame({'pos': candidate_angles, 'mask_score': mask_scores, 'diverse_score': diverse_scores})
        candidates['mask_score'] = candidates['mask_score'].map(
            lambda x: 0.5 if max_score == min_score else (x - min_score) / (max_score - min_score))

        choose_angles = [current_angle]
        choose_indexes = [current_index]

        for _ in range(1, k):
            current_index, current_angle = choose_next(candidates, choose_indexes, current_angle, 'rotate')
            choose_indexes.append(current_index)
            choose_angles.append(current_angle)

        # save image and mask
        for batch_index in range(k):
            angle = choose_angles[batch_index]
            mask_img = rotate_mask(shape, angle)

            np.save(op.join(dynamic_rotate_mask_dir, f'{batch_index}', f'{img_name}_mask.npy'), mask_img)
            dynamic_rotate_image = img_rotate(image, angle)
            index_i, index_j = get_largest_rec((image.shape[0], image.shape[1]), angle)
            dynamic_rotate_image = dynamic_rotate_image[index_i:image.shape[0] - index_i, index_j:image.shape[1] - index_j]
            cv2.imwrite(op.join(dynamic_rotate_dir, f'dynamic_rotate_{model}_{batch_index}', f'{img_name}.jpg'), dynamic_rotate_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--model', '-m')
    parser.add_argument('--transformation', '-t')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    transformation = args.transformation

    if transformation == 'crop':
        dynamic_crop(dataset, model)
    elif transformation == 'stretch':
        dynamic_stretch(dataset, model)
    elif transformation == 'rotate':
        dynamic_rotate(dataset, model)
    else:
        raise NotImplemented
