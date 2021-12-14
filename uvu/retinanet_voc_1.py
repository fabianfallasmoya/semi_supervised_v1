import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
#robfrom google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import MyTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog


#Step 1: Register dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("voc2007_train", {}, 
                        "../../../../share/semi_supervised/VOC2007_COCO_Full/annotations/_train.json", 
                        "../../../../share/semi_supervised/VOC2007_COCO_Full/trainval")

register_coco_instances("voc2007_val", {}, 
                        "../../../../share/semi_supervised/VOC2007_COCO_Full/annotations/_val.json", 
                        "../../../../share/semi_supervised/VOC2007_COCO_Full/trainval")

register_coco_instances("voc2007_test", {}, 
                        "../../../../share/semi_supervised/VOC2007_COCO_Full/annotations/_test.json", 
                        "../../../../share/semi_supervised/VOC2007_COCO_Full/test")

MetadataCatalog.get("voc2007_train").thing_classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog",
                                                        "horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

MetadataCatalog.get("voc2007_val").thing_classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog",
                                                        "horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

MetadataCatalog.get("voc2007_test").thing_classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog",
                                                        "horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

from detectron2.engine.MyTrainer import MyTrainer

if False:
    #from .detectron2.tools.train_net import Trainer
    #from detectron2.engine import DefaultTrainer
    # select from modelzoo here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines

    from detectron2.config import get_cfg
    #from detectron2.evaluation.coco_evaluation import COCOEvaluator

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("voc2007_train",)
    cfg.DATASETS.TEST = ("voc2007_val",)
    cfg.OUTPUT_DIR = "output/voc_full"

    #cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    #cfg.SOLVER.IMS_PER_BATCH = 4
    #cfg.SOLVER.BASE_LR = 0.001

    #cfg.SOLVER.WARMUP_ITERS = 500
    #cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
    #cfg.SOLVER.STEPS = []#(1000, 1500) ## do not decay learning rate
    #cfg.SOLVER.GAMMA = 0.05

    cfg.MODEL.RETINANET.NUM_CLASSES = 20 
    cfg.TEST.EVAL_PERIOD = 100
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if True:
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("voc2007_train",)
    cfg.DATASETS.TEST = ("voc2007_test",)
    cfg.OUTPUT_DIR = "output/voc_full"

    #cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.RETINANET.NUM_CLASSES = 20 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0014999.pth")

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator("voc2007_test", cfg, False, output_dir="./output/voc_full/testing_results")
    val_loader = build_detection_test_loader(cfg, "voc2007_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)