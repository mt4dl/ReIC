import numpy as np
import cv2


def img_rotate(img, angle):
    height = img.shape[0]
    width = img.shape[1]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (width, height))


def get_largest_rec(shape, angle):
    height = shape[0]
    width = shape[1]
    rotate_img = np.ones(shape)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotate_img = cv2.warpAffine(rotate_img, M, (width, height))
    index_i = 0
    index_j = 0
    for i in range(0, min(height, width) // 2):
        j = int(i * width / height)
        tem = rotate_img[i:height - i, j:width - j]
        if np.all(tem != 0):
            index_i = i
            index_j = j
            break
    return index_i, index_j


def get_raw_position(shape, angle, box):
    '''
    :param shape:  image shape
    :param angle:  rotate angle
    :param box:  box position after rotation
    :return:  box position before rotation
    '''
    height = shape[0]
    width = shape[1]
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    reverseMatRotation = cv2.invertAffineTransform(matRotation)
    pt1 = np.dot(reverseMatRotation, np.array([[box[0]], [box[1]], [1]])).astype(np.int)
    pt2 = np.dot(reverseMatRotation, np.array([[box[2]], [box[3]], [1]])).astype(np.int)
    pt3 = np.dot(reverseMatRotation, np.array([[box[4]], [box[5]], [1]])).astype(np.int)
    pt4 = np.dot(reverseMatRotation, np.array([[box[6]], [box[7]], [1]])).astype(np.int)
    return [[pt1[0][0], pt1[1][0]], [pt2[0][0], pt2[1][0]], [pt3[0][0], pt3[1][0]], [pt4[0][0], pt4[1][0]]]


def get_mask_img(shape, angle):
    height = shape[0]
    width = shape[1]
    index_i, index_j = get_largest_rec(shape, angle)
    box_after = [index_j, index_i, index_j, height - index_i, width - index_j, height - index_i, width - index_j, index_i]
    box_before = get_raw_position(shape, angle, box_after)
    mask_img = np.zeros(shape)
    return cv2.fillConvexPoly(mask_img, np.array(box_before), 1)
