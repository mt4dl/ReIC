import math
import numpy as np


def calculate_score(noun_positions, mask_image, threshold1, threshold2):
    crop_situation = [0, 0, 0]
    for noun_position in noun_positions:
        crop_rate = 1 - np.sum(np.multiply(noun_position, mask_image)) / np.sum(noun_position)
        if crop_rate >= threshold2:
            crop_situation[2] += 1
        elif threshold1 < crop_rate < threshold2:
            crop_situation[1] += 1
        else:
            crop_situation[0] += 1
    return crop_situation[2] - crop_situation[1]


def calculate_crop_average(shape, feature):
    crop_rate_sum = 0.0
    if not feature:
        return 0.9
    for obj in feature:
        rect = obj['rect']
        crop_rate = math.sqrt((rect[2] - rect[0]) * (rect[3] - rect[1]) / (shape[0] * shape[1]))
        crop_rate_sum += crop_rate
    crop_average = crop_rate_sum / len(feature)
    if int(shape[0] * (1 - crop_average)) < 3 or int(shape[1] * (1 - crop_average)) < 3:
        return 0.9
    return crop_average


def calculate_diversity(method, param1, param2):
    if method == 'crop':
        x1 = param1[0]
        y1 = param1[1]
        x2 = param1[2]
        y2 = param1[3]

        x3 = param2[0]
        y3 = param2[1]
        x4 = param2[2]
        y4 = param2[3]

        # interset rectange point
        a1 = max(x1, x3)
        b1 = max(y1, y3)
        a2 = min(x2, x4)
        b2 = min(y2, y4)
        if a1 >= a2 or b1 >= b2:
            diverse_score = 1.0
        else:
            intersect_area = (a2 - a1) * (b2 - b1)
            unintersect_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersect_area
            diverse_score = 1 - intersect_area / unintersect_area
        return diverse_score
    elif method == 'stretch':
        y1 = param1[0]
        y2 = param1[1]
        y3 = param2[0]
        y4 = param2[1]

        a1 = max(y1, y3)
        a2 = min(y2, y4)
        if a1 >= a2:
            diverse_score = 1.0
        else:
            intersect_area = a2 - a1
            unintersect_area = (y2 - y1) + (y4 - y3) - intersect_area
            diverse_score = 1 - intersect_area / unintersect_area
        return diverse_score
    elif method == 'rotate':
        return abs(param1 - param2) / 100.0
    else:
        raise NotImplementedError


def choose_next(candidates, choose_indexes, seed_pos, method):
    next_index = 0
    next_score = 0
    next_pos = None

    max_diverse_score = -1000
    min_diverse_score = 1000
    for index, value in candidates.iterrows():
        if index in choose_indexes:
            continue
        candidate_pos = value['pos']
        diverse_score = calculate_diversity(method, seed_pos, candidate_pos)

        # update diverse score
        diverse_score = min(diverse_score, candidates.loc[index, 'diverse_score'])
        candidates.loc[index, 'diverse_score'] = diverse_score

        if diverse_score > max_diverse_score:
            max_diverse_score = diverse_score
        if diverse_score < min_diverse_score:
            min_diverse_score = diverse_score

    diverse_scores = candidates['diverse_score'].map(
        lambda x: 1.0 if max_diverse_score == min_diverse_score else (x - min_diverse_score) / (max_diverse_score - min_diverse_score))

    for index, value in candidates.iterrows():
        if index in choose_indexes:
            continue
        candidate_pos = value['pos']
        mask_score = value['mask_score']
        diverse_score = diverse_scores[index]
        mix_score = diverse_score + mask_score
        if mix_score >= next_score:
            next_score = mix_score
            next_index = index
            next_pos = candidate_pos
    return next_index, next_pos


def choose_next_without_diverse(candidates, choose_indexes, seed_pos, method):
    next_index = 0
    next_score = 0
    next_pos = None

    for index, value in candidates.iterrows():
        if index in choose_indexes:
            continue
        candidate_pos = value['pos']
        mask_score = value['mask_score']
        mix_score = mask_score
        if mix_score >= next_score:
            next_score = mix_score
            next_index = index
            next_pos = candidate_pos
    return next_index, next_pos


def choose_next_without_score(candidates, choose_indexes, seed_pos, method):
    next_index = 0
    next_score = 0
    next_pos = None

    max_diverse_score = -1000
    min_diverse_score = 1000
    for index, value in candidates.iterrows():
        if index in choose_indexes:
            continue
        candidate_pos = value['pos']
        diverse_score = calculate_diversity(method, seed_pos, candidate_pos)

        # update diverse score
        diverse_score = min(diverse_score, candidates.loc[index, 'diverse_score'])
        candidates.loc[index, 'diverse_score'] = diverse_score

        if diverse_score > max_diverse_score:
            max_diverse_score = diverse_score
        if diverse_score < min_diverse_score:
            min_diverse_score = diverse_score

    diverse_scores = candidates['diverse_score'].map(
        lambda x: 1.0 if max_diverse_score == min_diverse_score else (x - min_diverse_score) / (max_diverse_score - min_diverse_score))

    for index, value in candidates.iterrows():
        if index in choose_indexes:
            continue
        candidate_pos = value['pos']
        diverse_score = diverse_scores[index]
        mix_score = diverse_score
        if mix_score >= next_score:
            next_score = mix_score
            next_index = index
            next_pos = candidate_pos
    return next_index, next_pos