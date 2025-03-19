# Copyright (c) Facebook, Inc. and its affiliates.
import os
from collections import abc
import cv2
import numpy as np
import tqdm
from PIL import Image
from detectron2.config import LazyConfig, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo
from decord import VideoReader, cpu

from qwen_vl_utils.vision_process import _read_video_decord_numpy

import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

def setup_cfg():
    # load config from file and command-line arguments
    config_file = "/home/user/wangxd/github/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
    opts = ['train.init_checkpoint=/data2/wangxd/models/ape_checkpoint/model_final.pth', 'model.model_language.cache_dir=', 'model.model_vision.select_box_nums_for_evaluation=500', 'model.model_vision.text_feature_bank_reset=True', 'model.model_vision.backbone.net.xattn=False']
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    confidence_threshold = 0.3

    if "output_dir" in cfg.model:
        cfg.model.output_dir = cfg.train.output_dir
    if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
        cfg.model.model_vision.output_dir = cfg.train.output_dir
    if "train" in cfg.dataloader:
        if isinstance(cfg.dataloader.train, abc.MutableSequence):
            for i in range(len(cfg.dataloader.train)):
                if "output_dir" in cfg.dataloader.train[i].mapper:
                    cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
        else:
            if "output_dir" in cfg.dataloader.train.mapper:
                cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = confidence_threshold
    else:
        cfg.model.test_score_thresh = confidence_threshold

    setup_logger(name="ape")
    setup_logger(name="timm")

    return cfg

def ape_inference(paths, text_prompt, demo):

    res_list = []

    # if input is mp4
    if ".mp4" in paths:
        input, _ = _read_video_decord_numpy({"video":paths})
    else:
        input = paths

    for item in tqdm.tqdm(input):
        path = None
        # use PIL, to be consistent with evaluation
        try:
            if type(item) == str:
                path = item
                img = read_image(path, format="BGR")
            else:
                img = item
                path = paths
        except Exception as e:
            continue
        
        predictions, visualized_output, visualized_outputs, metadata = demo.run_on_image(
            img,
            text_prompt=text_prompt,
            with_box=True,
            with_mask=False,
            with_sseg=False,
        )

        # import time
        # visualized_output.save(str(int(time.time()*1000000))+'.jpg')
        
        res = ""
        with_box = True
        categories = []
        if "instances" in predictions:
            results = instances_to_coco_json(
                predictions["instances"].to(demo.cpu_device), path
            )
            # if with_box:
            #     for result in results:
            #         res += metadata.thing_classes[result["category_id"]] + ": ["
            #         for idx, box in enumerate(result['bbox']):
            #             if idx != 3:
            #                 res += str(int(box)) + ", "
            #             else:
            #                 res += str(int(box))
            #         res += "]; "
            # else:
            #     for result in results:
            #         res += metadata.thing_classes[result["category_id"]] + ", "
            for json_result in results:
                json_result["category_name"] = metadata.thing_classes[json_result["category_id"]]
                del json_result["image_id"]
            
            categories = [json_result["category_name"] for json_result in results]
    
            # print(f"categories: {categories}")

        res_list.append(categories)
        
        # if len(res) > 0:
        #     if with_box:
        #         res_list.append(res[:-2])
        #     else:
        #         res_list.append(res)
        # else:
        #     res_list.append("")

    return res_list

if __name__ == "__main__":
    
    print("")
    # cfg = setup_cfg()
    
    # demo = VisualizationDemo(cfg, args=None)
    

    # text_prompt = "Apples,Candles,Berries"
    # res = ape_inference(input, text_prompt, demo)
    # print(res)


        
