import os
import numpy as np
import pandas as pd
import json
import os.path as op
import cv2

import argparse

from tqdm import tqdm


def get_raw_data_and_objects(dataset):
    sgg_results = pd.read_csv(f'../image_features_{dataset}.tsv', sep='\t', header=None, converters={1: json.loads})
    data_objects = []
    for res in zip(sgg_results[0], sgg_results[1]):
        image_path = op.join('..', '..', 'data', 'images', dataset, f'{res[0]}.jpg')
        cv_image = cv2.imread(image_path)
        data_object = [res[0], cv_image]
        rects = []
        classes = []
        for item in res[1]:
            rects.append([int(x) for x in item['rect']])
            classes.append(item['class'])
        data_object.append(rects)
        data_object.append(classes)
        data_objects.append(data_object)
    return data_objects


def blur_rec(img, rec, k_size=(52, 52)):
    masked_img = img.copy()
    blur_img = cv2.blur(img, k_size)
    rec = [int(x) for x in rec]
    masked_img[rec[1]:rec[3], rec[0]:rec[2]] = blur_img[rec[1]:rec[3], rec[0]:rec[2]]
    return masked_img


# generate blurring images and black filling images
def generate_occlude_images(save_dir, dataset):
    data_objects = get_raw_data_and_objects(dataset=dataset)
    meta_data = []
    for name, image, rects, classes in tqdm(data_objects):
        o_index = 0
        for rect, class_name in zip(rects, classes):
            meta_data.append([name, f'{name}_{o_index}', rect, class_name])
            blur_img = blur_rec(image, rect)
            black_filling_img = image.copy()
            black_filling_img[rect[1]: rect[3], rect[0]: rect[2]] = 0
            cv2.imwrite(f'{save_dir}/{dataset}/blur/{name}_{o_index}.jpg', blur_img)
            cv2.imwrite(f'{save_dir}/{dataset}/black/{name}_{o_index}.jpg', black_filling_img)
            o_index += 1
    pd.DataFrame(meta_data).to_csv(f'occlusion_{dataset}.csv')


# generate inpainting prepare images
def generate_inpainting_image_and_mask(lama_dir, dataset):
    mask_dir = op.join(lama_dir, dataset)
    data_objects = get_raw_data_and_objects(dataset=dataset)
    for name, image, rects, classes in data_objects:
        o_index = 0
        cv2.imwrite(op.join(mask_dir, f'{name}.png'), image)
        for rect, class_name in zip(rects, classes):
            masked_img = np.zeros(image.shape, dtype=np.int)
            masked_img[rect[1]: rect[3], rect[0]: rect[2]] = 255
            cv2.imwrite(op.join(mask_dir, f'{name}_mask{o_index}.png'), masked_img)
            o_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', '-s')
    parser.add_argument('--dataset', '-d')
    args = parser.parse_args()

    save_dir = args.save_dir
    dataset = args.dataset
    # save_dir = 'test'
    # dataset = 'mscoco'

    lama_dir = 'lama'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(op.join(save_dir, dataset, 'blur'), exist_ok=True)
    os.makedirs(op.join(save_dir, dataset, 'black'), exist_ok=True)
    os.makedirs(op.join(save_dir, dataset, 'inpainting'), exist_ok=True)

    os.makedirs(op.join(lama_dir, dataset), exist_ok=True)

    generate_occlude_images(save_dir, dataset)
    generate_inpainting_image_and_mask(lama_dir, dataset)
