import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

import json
import cv2
import numpy as np
import os.path as op
import argparse

import pandas as pd
from functions.semantic_similarity import get_similarity
import pickle


def OD_position(model, dataset):
    # get image features
    sgg_results = pd.read_csv(f'image_features_{dataset}.tsv', sep='\t', header=None, converters={1: json.loads})
    # get caption nouns
    caption_nouns = pd.read_csv(f'caption_objects/{model}_{dataset}.csv', converters={'1': json.loads}, index_col=0)

    os.makedirs(f'result/{dataset}/{model}/OD_result', exist_ok=True)
    os.makedirs(f'flag/{dataset}', exist_ok=True)

    flags = dict()
    img_indexes = sgg_results.index
    for img_index in img_indexes:
        img_name = sgg_results.at[img_index, 0]
        features = sgg_results.at[img_index, 1]
        nouns = caption_nouns.at[img_index, '1']
        # get image and mask

        image_path = op.join('..', 'data', 'images', dataset, f'{img_name}.jpg')
        image = cv2.imread(image_path)
        shape = (image.shape[0], image.shape[1])

        for noun_index in nouns:
            noun = nouns[noun_index]
            noun_position = np.zeros(shape, dtype=np.int)
            OD_flag = False
            for feature in features:
                class_name = feature['class'].lower()
                if get_similarity(noun, class_name, word_similarity_threshold) < word_similarity_threshold:
                    continue
                rect = feature['rect']
                noun_position[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])] += 1
                OD_flag = True
            if OD_flag:
                # save position
                noun_position = np.minimum(noun_position, 1)
                np.save(f'result/{dataset}/{model}/OD_result/{img_name}_{noun_index}.npy', noun_position)
                flags[f'{img_name}_{noun_index}'] = 'OD'
            else:
                flags[f'{img_name}_{noun_index}'] = 'occ'
    with open(f'flag/{dataset}/{model}_flag.pkl', 'wb') as f:
        pickle.dump(flags, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--model', '-m')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model

    word_similarity_threshold = 0.7
    # models = ['show_and_tell', 'oscar', 'vinvl', 'ofa', 'Azure']
    # for model in models:
    OD_position(model, dataset)