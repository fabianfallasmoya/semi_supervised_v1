from typing import List
import argparse

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader

from trainers import MyTrainer





def test(args):
    register_coco_instances(args.ds_test, {}, args.ds_test_json, args.ds_test_imgs)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.yaml_config_file))

    # set parameters
    cfg.DATASETS.TEST = (args.ds_test,)
    cfg.DATASETS.TRAIN = (args.ds_test,)
    cfg.OUTPUT_DIR = args.out_folder_val
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.RETINANET.NUM_CLASSES = args.num_classes
    
    # perform inference
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    evaluator = COCOEvaluator(args.ds_test, cfg, False, output_dir=args.out_folder_val)
    val_loader = build_detection_test_loader(cfg, args.ds_test)
    inference_on_dataset(trainer.model, val_loader, evaluator)
    

def parse_arguments():
    """
    Get parameter to train.
    """
    parser = argparse.ArgumentParser(description="Training / Evaluating RetinaNet")
    parser.add_argument("--ds_test", type=str, required=True)
    parser.add_argument("--ds_test_json", type=str, required=True)
    parser.add_argument("--ds_test_imgs", type=str, required=True)
    parser.add_argument("--out_folder_val", type=str, required=True, help="output folder for val results")
    parser.add_argument("--yaml_config_file", type=str, required=True, help="config file of the model to use")
    parser.add_argument("--weights", type=str, default=None, help="load weights")
    parser.add_argument("--num_classes", type=int, required=True, help="number of classes in the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    test(args)