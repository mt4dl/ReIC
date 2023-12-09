import cv2
import os
import argparse


def move(lama_dir, save_dir, dataset):
    output_dir = f'{lama_dir}/{dataset}_output'
    for file_name in os.listdir(output_dir):
        new_file_name = file_name.replace('mask', '').replace('png', 'jpg')
        image = cv2.imread(f'{output_dir}/{file_name}')
        cv2.imwrite(f'{save_dir}/{dataset}/inpainting/{new_file_name}', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', '-s')
    parser.add_argument('--dataset', '-d')
    args = parser.parse_args()

    save_dir = args.save_dir
    dataset = args.dataset

    lama_dir = 'lama'
    move(lama_dir, save_dir, dataset)