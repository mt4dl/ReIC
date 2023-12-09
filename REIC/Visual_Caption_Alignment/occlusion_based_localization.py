import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse
import numpy as np
import pandas as pd
from functions.extract_objects import get_objects_in_caption
from functions.semantic_similarity import get_similarity
from functions.lem import lem
import cv2


def is_in(word, set, st):
    for w in set:
        similarity = get_similarity(word, w, st)
        if similarity >= st:
            return True
    return False


def is_rect_include(rect1, rect2):
    if rect1[0] <= rect2[0] and rect1[1] <= rect2[1] and rect1[2] >= rect2[2] and rect1[3] >= rect2[3]:
        return 2
    if rect1[0] >= rect2[0] and rect1[1] >= rect2[1] and rect1[2] <= rect2[2] and rect1[3] <= rect2[3]:
        return 1
    return 0


def trim_set(rects):
    new_rects = []
    flag = [1 for _ in range(len(rects))]
    for i in range(len(rects) - 1):
        for j in range(i + 1, len(rects)):
            is_include = is_rect_include(rects[i], rects[j])
            if is_include == 1:
                flag[j] = 0
            if is_include == 2:
                flag[i] = 0
    for index in range(len(rects)):
        if flag[index] == 1:
            new_rects.append(rects[index])
    return new_rects


def occlusion_localization(model, dataset):
    raw_captions = pd.read_csv(f'model_captions/{dataset}/{model}/raw.tsv', index_col=0)
    masked_captions1 = pd.read_csv(f'model_captions/{dataset}/{model}/blur.tsv', index_col=0)
    masked_captions2 = pd.read_csv(f'model_captions/{dataset}/{model}/black.tsv', index_col=0)
    masked_captions3 = pd.read_csv(f'model_captions/{dataset}/{model}/inpainting.tsv', index_col=0)
    meta_data = pd.read_csv(f'image_occlusion/occlusion_{dataset}.csv', index_col=0)

    result_dir = f'result/{dataset}/{model}/occlusion_result'
    os.makedirs(result_dir, exist_ok=True)

    for index in range(0, 300):
        image_path = os.path.join('..', 'data', 'images', dataset, f'{index}.jpg')
        image = cv2.imread(image_path)
        shape = (image.shape[0], image.shape[1])

        raw_caption = raw_captions.iloc[index, 0]
        nouns, noun_indexes = get_objects_in_caption(raw_caption)
        nouns = [lem(n) for n in nouns]
        noun_index_zip = zip(noun_indexes, nouns)
        masked_images = meta_data[meta_data['0'] == index]

        occlusion_rects = dict()
        for noun_index, _ in noun_index_zip:
            occlusion_rects[noun_index] = []
        for _, item in masked_images.iterrows():
            masked_index = item.iloc[1]
            rect = [int(x) for x in item.iloc[2].strip('[').strip(']').split(', ')]
            masked_caption1 = masked_captions1.loc[masked_index, '1']
            masked_caption2 = masked_captions2.loc[masked_index, '1']
            masked_caption3 = masked_captions3.loc[masked_index, '1']
            if pd.isna(masked_caption3):
                masked_caption3 = '.'
            mask_nouns1, _ = get_objects_in_caption(masked_caption1)
            mask_nouns2, _ = get_objects_in_caption(masked_caption2)
            mask_nouns3, _ = get_objects_in_caption(masked_caption3)
            mask_nouns = mask_nouns1 + mask_nouns2 + mask_nouns3
            mask_nouns = [lem(n) for n in mask_nouns]
            noun_index_zip = zip(noun_indexes, nouns)
            for noun_index, noun in noun_index_zip:
                if not is_in(noun, mask_nouns, word_similarity_threshold):
                    occlusion_rects[noun_index].append(rect)
        for key in occlusion_rects:
            localization_result = np.zeros(shape, dtype=np.int)
            for rect in trim_set(occlusion_rects[key]):
                localization_result[rect[1]:rect[3], rect[0]:rect[2]] += 1
            avg_v = np.average(localization_result)
            localization_result = np.where(localization_result > avg_v, localization_result, 0)
            with open(f'{result_dir}/{index}_{key}.npy', 'wb') as f:
                np.save(f, localization_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--model', '-m')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model

    word_similarity_threshold = 0.7
    occlusion_localization(model, dataset)

