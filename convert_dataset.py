from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset

import pandas as pd
import os
import json

from PIL import Image

import torch


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def gen_test_annotation(test_data_path, annotation_path):
    test_anno_list = []
    for img in os.listdir(test_data_path):
        if img.endswith('jpg'):
            img_info = {}
            img_info['filename'] = img
            img_size = Image.open(os.path.join(test_data_path, img)).size
            img_info['width'] = img_size[0]
            img_info['height'] = img_size[1]
            test_anno_list.append(img_info)
    with open(annotation_path, 'w+') as f:
        json.dump(test_anno_list, f)

if __name__=="__main__":
    DIR_INPUT = '/kaggle/working/mmdetection/data/Wheatdetection'
    DIR_TEST = f'{DIR_INPUT}/test'
    DIR_ANNO = f'{DIR_INPUT}/annotations'

    DIR_WEIGHTS = '/kaggle/input/mmdetfasterrcnn'
    WEIGHTS_FILE = f'{DIR_WEIGHTS}/epoch_50.pth'

    test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

    # prepare test data annotations
    gen_test_annotation(DIR_TEST, DIR_ANNO + '/detection_test.json')