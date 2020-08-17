from pathlib import Path
import json
from typing import List
import csv

def load_labels(label_path: str, class_names):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    f.close()
    coco_labels = []
    for line in lines:
        kitti_label = line.strip().split(' ')
        kitti_label = [kitti_label[0], *list(map(float, kitti_label[1:]))]
        cls = class_names.index(kitti_label[0])
        width = kitti_label[6] - kitti_label[4]
        height = kitti_label[7] - kitti_label[5]
        x_center = (kitti_label[4] + width/2) / 3024
        y_center = (kitti_label[5] + height/2) / 4032
        coco_label = [cls, x_center, y_center, width/3024, height/4032]
        coco_labels.append(coco_label)
    return coco_labels

def save_labels(save_path: str, labels: List):
    with open(save_path, 'w') as fl:
        csv.writer(fl,delimiter=" ").writerows(labels)

if __name__ == "__main__":
    load_dir = '../../Documents/obj_samples/Linz/unwarpped_txt_p1'
    save_dir = Path(load_dir).parent.joinpath('yolov5_labels')
    class_names = ['yes', 'no', 'both', 'empty']
    save_dir.mkdir(exist_ok=True)
    label_path_list = Path(load_dir).glob('*txt')
    for label_path in label_path_list:
        labels = load_labels(label_path, class_names)
        save_path = save_dir.joinpath(label_path.name)
        save_labels(save_path, labels)


