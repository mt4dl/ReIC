import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse
import json
import numpy as np
import pandas as pd
from Visual_Caption_Alignment.get_result import get_object_pos


def get_noun_area(dataset, model):
    os.makedirs('object_areas', exist_ok=True)
    # get caption nouns
    caption_nouns = pd.read_csv(f'../Visual_Caption_Alignment/caption_objects/{model}_{dataset}.csv', converters={'1': json.loads}, index_col=0)
    images = []
    for _, value in caption_nouns.iterrows():
        img_name = value[0]
        img_nouns = value[1]

        image_features = {
            "image_name": value[0],
            "feature": []
        }
        feature = []
        for noun_index in img_nouns:
            _, pos_img = get_object_pos(dataset, model, img_name, noun_index)
            if not np.any(pos_img):
                continue
            def get_bbox(img):
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                return [int(xmin), int(ymin), int(xmax) + 1, int(ymax) + 1]

            noun_area = get_bbox(pos_img)
            feature.append({'index': noun_index, 'noun': f'{img_nouns[noun_index]}', 'rect': noun_area})
            image_features['feature'] = json.dumps(feature)
        images.append(image_features)
    pd.DataFrame(images).to_csv(f'object_areas/{model}_{dataset}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--model', '-m')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model

    get_noun_area(dataset, model)