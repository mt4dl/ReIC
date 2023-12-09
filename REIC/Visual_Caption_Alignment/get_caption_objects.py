import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

import json
import argparse
import pandas as pd

from functions.extract_objects import get_objects_in_caption
from functions.lem import lem


def get_raw_caption_objects(model, dataset):
    # raw captions
    raw_captions = pd.read_csv(f'model_captions/{dataset}/{model}/raw.tsv', index_col=0)
    indexes = raw_captions.index

    results = []
    for index in indexes:
        # get caption
        index_nouns = dict()
        raw_predict_caption = raw_captions.at[index, '1']
        raw_nouns, raw_indexes = get_objects_in_caption(raw_predict_caption)
        raw_nouns = [lem(n) for n in raw_nouns]
        for noun_indexes, nouns in zip(raw_indexes, raw_nouns):
            index_nouns[noun_indexes] = nouns
        results.append([index, json.dumps(index_nouns)])
    pd.DataFrame(results).to_csv(f'caption_objects/{model}_{dataset}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--model', '-m')
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model

    os.makedirs('caption_objects', exist_ok=True)
    get_raw_caption_objects(model, dataset)