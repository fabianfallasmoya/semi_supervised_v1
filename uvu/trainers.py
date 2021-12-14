from typing import List
import os

import detectron2
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from my_hooks import LossEvalHook



class MyTrainer(DefaultTrainer):
    """
    Custom trainer with new evaluator and adds a new hook to have eval loss.
    """
    def __init__(self, cfg: detectron2.config.CfgNode):
        self.cfg = cfg
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, 
                        cfg: detectron2.config.CfgNode, 
                        dataset_name: str, 
                        output_folder: str = None) -> COCOEvaluator:
        """
        We just indicate that during training the evaluation results
        will go to a specific folder.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    
    def build_hooks(self) -> List[detectron2.engine.HookBase]:
        """
        Every training step has hooks. This methods adds a hook
        at the end of the list of hooks that calculates the Loss
        over the Eval dataset.
        """
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks