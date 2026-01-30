# DRPI-DEIM
# Description
We propose an underwater object detection algorithm based on multi-scale feature fusion in spatial and frequency domains, using DEIM, specifically for autonomous underwater vehicles to detect marine debris.
# **Experimental** **setup**
All experiments were conducted on a server running Ubuntu 22.04 operating system, equipped with an NVIDIA RTX 4090D GPU, 24GB memory and 60GB system memory. The experiments were implemented using Python 3.10 and based on the PyTorch 2.3.0 framework, with identical hyper-parameters for  training and validation. 
# Download
The dataset and models have been released. [Data&Model](https://pan.baidu.com/s/1luMDDhD-ladaSzFQ1Jq0Ig?pwd=2u7f)
- These experiments utilized datasets including: UDD, UODD, DUO, RUOD.
- This model includes the design of module in the backbone of the DEIM model, the design of the encoder structure, and the improvement of the feature extraction module, and proposes a Frequency-Spatial Transformer Block for solving image degradation problems. 

### Dataset Directory Structure
~~~
dataset/
├── train
│   ├── annotations
    └── images
    └── labels
├── val
│   ├── annotations
    └── images
    └── labels
├── test
│   ├── annotations
    └── images
    └── labels

-> data/dataset.yml
~~~
### COCO_txt
~~~
import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./dataset/valid',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str,default='./valid.json', help="if not split the dataset, give a path to a json file")
parser.add_argument('--random_split', action='store_true', help="random split the dataset, default ratio is 8:1:1")
parser.add_argument('--split_by_file', action='store_true', help="define how to split the dataset, include ./train.txt ./val.txt ./test.txt ")

arg = parser.parse_args()

def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ",root_path)

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, 'labels')                                        
    originImagesDir = os.path.join(root_path, 'images')
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
    # images dir name
    indexes = os.listdir(originImagesDir)

    if arg.random_split or arg.split_by_file:
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            
        if arg.random_split:
            print("spliting mode: random split")
            train_img, val_img, test_img = train_test_val_split_random(indexes,0.8,0.1,0.1)
        elif arg.split_by_file:
            print("spliting mode: split by files")
            train_img, val_img, test_img = train_test_val_split_by_files(indexes, root_path)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        txtFile = index.replace('images','txt').replace('.jpg','.txt').replace('.png','.txt')
        im = cv2.imread(os.path.join(root_path, 'images/') + index)
        height, width, _ = im.shape
        if arg.random_split or arg.split_by_file:
                if index in train_img:
                    dataset = train_dataset
                elif index in val_img:
                    dataset = val_dataset
                elif index in test_img:
                    dataset = test_dataset
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            continue
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H

                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if arg.random_split or arg.split_by_file:
        for phase in ['train','val','test']:
            json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'val':
                    json.dump(val_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_path, 'annotations/{}'.format(arg.save_path))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":
    yolo2coco(arg)
~~~
