import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import shutil
from PIL import Image as im
import os


def save_images(df, imgs_id, cat_map, all_classes):
    for img_id in imgs_id:
        src_dir = './dataset/imaterialist-fashion-2020-fgvc7/train/'
        dst_dir = './dataset/images/'
        image = mpimg.imread(src_dir + img_id + '.jpg')
        encoded_pixels = df[df['ImageId'] == img_id]['EncodedPixels']
        class_ids = df[df['ImageId'] == img_id]['ClassId']

        # Check if there is any new class ids that it is not seen until now in the previous images
        ''' res = any(item in all_classes for item in class_ids)
        if res is False:
            continue'''
        # Work with masks
        height, width = image.shape[:2]
        masks = []
        f_masks = None
        for pixels, class_id in zip(encoded_pixels, class_ids):
            mask = np.zeros((height, width)).reshape(-1)
            pixels_split = list(map(int, pixels.split()))
            pixel_starts = pixels_split[::2]
            run_lengths = pixels_split[1::2]
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                mask[pixel_start:pixel_start + run_length] = 255 - class_id * 4
            mask = mask.reshape(height, width, order='F')
            f_masks = np.where(mask != 0, mask, f_masks if f_masks is not None else mask)
            masks.append(mask)
            # Remove actual class_id from the list of classes
            '''if class_id in all_classes:
                all_classes.remove(class_id)'''

        # Save image in greyscale with all masks
        im.fromarray(f_masks).convert('RGB').save(dst_dir + img_id + '_MASKS.png')
        print(image.shape, f_masks.shape)
        test_shape = mpimg.imread(dst_dir + img_id + '_MASKS.png')
        print(test_shape.shape)

        # Save every single mask with class ID and class Name in the name of file saved
        '''for m, class_id in zip(masks, class_ids):
            im.fromarray(m).convert('L') \
                .save(dst_dir + img_id + '_mask_' + str(class_id) + cat_map[class_id] + '.png')'''

        # Save original image
        # shutil.copy(src_dir + img_id + '.jpg', dst_dir + img_id + '.jpg')

        # Check if the list of classes is empty
        '''print(all_classes)
        if len(all_classes) == 0:
            break'''


def create_mask(df, img_id):
    src_dir = './dataset/imaterialist-fashion-2020-fgvc7/train/'
    image = mpimg.imread(src_dir + img_id + '.jpg')
    encoded_pixels = df[df['ImageId'] == img_id]['EncodedPixels']
    class_ids = df[df['ImageId'] == img_id]['ClassId']

    # Work with masks
    height, width = image.shape[:2]
    f_masks = None
    for pixels, class_id in zip(encoded_pixels, class_ids):
        mask = np.zeros((height, width)).reshape(-1)
        pixels_split = list(map(int, pixels.split()))
        pixel_starts = pixels_split[::2]
        run_lengths = pixels_split[1::2]
        for pixel_start, run_length in zip(pixel_starts, run_lengths):
            mask[pixel_start:pixel_start + run_length] = class_id + 1
        mask = mask.reshape(height, width, order='F')
        f_masks = np.where(mask != 0, mask, f_masks if f_masks is not None else mask)
    return f_masks


def create_dataset(df, imgs_id):
    # split train, val and test set
    train_len = int(len(imgs_id) * 0.7)
    val_len = int(len(imgs_id) * 0.2)
    train_imgs_id = imgs_id[: train_len]
    val_imgs_id = imgs_id[train_len:train_len + val_len]
    test_imgs_id = imgs_id[train_len + val_len:len(imgs_id)]
    # check dimension
    print(len(imgs_id))
    print(len(train_imgs_id))
    print(len(val_imgs_id))
    print(len(test_imgs_id))
    print(len(train_imgs_id) + len(val_imgs_id) + len(test_imgs_id))

    # create and save masks
    src_dir = './dataset/imaterialist-fashion-2020-fgvc7/train/'
    img_train_dir = './dataset/img_dir/train/'
    ann_train_dir = './dataset/ann_dir/train/'
    if not os.path.isdir(img_train_dir):
        os.makedirs(img_train_dir)
    if not os.path.isdir(ann_train_dir):
        os.makedirs(ann_train_dir)
    for img_id in train_imgs_id:
        train_f_masks = create_mask(df, img_id)
        # Save image in ann_dir
        im.fromarray(train_f_masks).convert('L').save(ann_train_dir + img_id + '.png')
        # Save original image in img_dir
        # shutil.copy(src_dir + img_id + '.jpg', img_train_dir + img_id + '.jpg')

    img_val_dir = './dataset/img_dir/val/'
    ann_val_dir = './dataset/ann_dir/val/'
    if not os.path.isdir(img_val_dir):
        os.makedirs(img_val_dir)
    if not os.path.isdir(ann_val_dir):
        os.makedirs(ann_val_dir)
    for img_id in val_imgs_id:
        val_f_masks = create_mask(df, img_id)
        im.fromarray(val_f_masks).convert('L').save(ann_val_dir + img_id + '.png')
        # shutil.copy(src_dir + img_id + '.jpg', img_val_dir + img_id + '.jpg')

    img_test_dir = './dataset/img_dir/test/'
    ann_test_dir = './dataset/ann_dir/test/'
    if not os.path.isdir(img_test_dir):
        os.makedirs(img_test_dir)
    if not os.path.isdir(ann_test_dir):
        os.makedirs(ann_test_dir)
    for img_id in test_imgs_id:
        test_f_masks = create_mask(df, img_id)
        im.fromarray(test_f_masks).convert('L').save(ann_test_dir + img_id + '.png')
        # shutil.copy(src_dir + img_id + '.jpg', img_test_dir + img_id + '.jpg')

    print(len([name for name in os.listdir(ann_train_dir)]))
    print(len([name for name in os.listdir(ann_val_dir)]))
    print(len([name for name in os.listdir(ann_test_dir)]))


def test_data():
    # Training dataset
    train_df = pd.read_csv('./dataset/imaterialist-fashion-2020-fgvc7/train.csv')
    # remove useless classes from dataset
    train_df.drop(train_df.index[train_df['ClassId'] >= 27], inplace=True)
    images_id = train_df['ImageId'].unique()

    ''' Get label file
    with open('./dataset/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file:
        label_d = json.load(file)'''

    # save_images(train_df, images_id, 0, 0)
    create_dataset(train_df, images_id)
