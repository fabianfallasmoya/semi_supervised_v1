import os
import argparse
from typing import List

import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.data import (
    build_detection_test_loader,
    DatasetMapper,
)
from detectron2.evaluation import COCOEvaluator
from my_hooks import LossEvalHook
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# my trainer
from trainers import MyTrainer

        

def train_setup(args):
    """
    Uses args to config the model.
    """
    setup_logger()

    # register ds names and location of files
    register_coco_instances(args.ds_train, {}, args.ds_train_json, args.ds_train_imgs)
    register_coco_instances(args.ds_val, {}, args.ds_train_json, args.ds_train_imgs)

    # config model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.yaml_config_file))

    # set params
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.weights)
    cfg.DATASETS.TRAIN = (args.ds_train,)
    cfg.DATASETS.TEST = (args.ds_val,)
    cfg.MODEL.RETINANET.NUM_CLASSES = args.num_classes
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.OUTPUT_DIR = args.out_folder_val
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)   

    # additional parameters we can use in retinanet
    #cfg.SOLVER.IMS_PER_BATCH = 4
    #cfg.SOLVER.BASE_LR = 0.001
    #cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.SOLVER.WARMUP_ITERS = 500
    #cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
    #cfg.SOLVER.STEPS = []#(1000, 1500) ## do not decay learning rate
    #cfg.SOLVER.GAMMA = 0.05
    return cfg


def train(args):
    """
    Start training
    """
    # get complete configuration of the model
    cfg = train_setup(args)

    # train the model
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def parse_arguments():
    """
    Get parameter to train.
    """
    parser = argparse.ArgumentParser(description="Training / Evaluating RetinaNet")
    parser.add_argument("--ds_train", type=str, required=True, help="name of training dataset in the system")
    parser.add_argument("--ds_train_json", type=str, required=True, help="json annotations of training data")
    parser.add_argument("--ds_train_imgs", type=str, required=True, help="training images")
    parser.add_argument("--ds_val", type=str, required=True, help="name of val dataset in the system")
    parser.add_argument("--ds_val_json", type=str, required=True, help="json annotations of val data")
    parser.add_argument("--ds_val_imgs", type=str, required=True, help="validation images")
    parser.add_argument("--out_folder_val", type=str, required=True, help="output folder for val results")
    parser.add_argument("--yaml_config_file", type=str, required=True, help="config file of the model to use")
    parser.add_argument("--weights", type=str, default=None, help="load weights")
    parser.add_argument("--num_classes", type=int, required=True, help="number of classes in the dataset")
    parser.add_argument("--eval_period", type=int, default=100, help="periodicity of eval calculation")

    return parser.parse_args()


if __name__ == "__main__":
    print('bingo')
    args = parse_arguments()
    train(args)