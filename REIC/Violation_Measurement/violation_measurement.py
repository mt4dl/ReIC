import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse
import numpy as np
import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from functions.semantic_similarity import get_similarity
from functions.extract_objects import get_objects_in_caption, get_objects_in_followup_caption
from functions.lem import lem
from Visual_Caption_Alignment.get_result import get_object_pos
from functions.get_mask import get_mask_img_diverse


def is_in(word, set):
    for w in set:
        similarity = get_similarity(word, w, word_similarity_threshold)
        if similarity >= word_similarity_threshold:
            return True
    return False


def is_violation(dataset, model, mr, batch, img_name, raw_index_nouns, mutate_nouns, mutate_nouns_of):
    object_results = []
    case_condition = 0
    raw_nouns = []
    crop_flags = False
    mask_img = get_mask_img_diverse(dataset, model, mr, img_name, batch)
    for raw_noun_index, raw_n in raw_index_nouns:
        raw_nouns.append(raw_n)
        object_result = [img_name, raw_n]
        obj_condition = False
        method_flag, noun_pos = get_object_pos(dataset, model, img_name, raw_noun_index)
        crop_rate = 1 - np.sum(np.multiply(noun_pos, mask_img)) / np.sum(noun_pos)
        if crop_rate >= threshold2:
            crop_flag = 2
            if is_in(raw_n, mutate_nouns):
                obj_condition = True
        elif threshold1 < crop_rate < threshold2:
            crop_flag = 1
        else:
            crop_flag = 0
            if not is_in(raw_n, mutate_nouns):
                obj_condition = True
        if obj_condition:
            object_result.append(1)
            case_condition = 1
        else:
            object_result.append(0)
        object_result.append(method_flag)
        if crop_flag == 2:
            crop_flags = True
            object_result.append('disappear')
        elif crop_flag == 0:
            object_result.append('retain')
        elif crop_flag == 1:
            crop_flags = True
            object_result.append('ambiguous')
        object_results.append(object_result)
    if not crop_flags:
        for mutate_n in mutate_nouns_of:
            object_result = [img_name, mutate_n]
            obj_condition = False
            if not is_in(mutate_n, raw_nouns):
                obj_condition = True
            if obj_condition:
                object_result.append(1)
                case_condition = 1
            else:
                object_result.append(0)
            object_results.append(object_result)
    return case_condition, object_results


def main(dataset, model, mr, batch):
    '''
    Args:
        mr: metamorphic relation
        t1: low threshold for crop analysis
        t2: high threshold for crop analysis
    Returns:
    '''
    obj_result = []
    case_result = []

    # raw captions
    raw_captions = pd.read_csv(f'../data/captions/{dataset}/{model}.tsv', index_col=0)
    # mutate captions
    mutate_captions = pd.read_csv(f'follow_up_captions/{dataset}/dynamic_{mr}_{model}_{batch}.tsv', index_col=0)

    indexes = raw_captions.index

    for index in indexes:
        # get caption
        raw_predict_caption = raw_captions.at[index, '1']
        mutate_predict_caption = mutate_captions.at[index, '1']

        raw_nouns, raw_indexes = get_objects_in_caption(raw_predict_caption)
        raw_nouns = [lem(n) for n in raw_nouns]
        mutate_nouns, mutate_indexes = get_objects_in_followup_caption(mutate_predict_caption)
        mutate_nouns = [lem(n) for n in mutate_nouns]

        mutate_nouns_of, _ = get_objects_in_caption(mutate_predict_caption)
        mutate_nouns_of = [lem(n) for n in mutate_nouns_of]

        raw_noun_indexes = zip(raw_indexes, raw_nouns)
        violation = is_violation(dataset, model, mr, batch, index, raw_noun_indexes, mutate_nouns, mutate_nouns_of)
        obj_result.extend(violation[1])
        case_result.append(violation[0])
    os.makedirs(f'object_result/{dataset}_{model}', exist_ok=True)
    pd.DataFrame(obj_result).to_csv(f'object_result/{dataset}_{model}/{mr}_{batch}.csv')

    os.makedirs(f'case_result/{dataset}_{model}', exist_ok=True)
    pd.DataFrame(case_result).to_csv(f'case_result/{dataset}_{model}/{mr}_{batch}.csv')


if __name__ == '__main__':
    # get input
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m')
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--mr', '-r')
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    mr = args.mr

    word_similarity_threshold = 0.7
    threshold1 = 0.2
    threshold2 = 0.9

    for batch in [0, 1, 2]:
        main(dataset, model, mr, batch)
