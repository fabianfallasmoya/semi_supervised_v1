DS_TRAIN="voc2007_train"
DS_TRAIN_JSON="../../../../share/semi_supervised/VOC2007_COCO_Full/annotations/_train.json"
DS_TRAIN_IMGS="../../../../share/semi_supervised/VOC2007_COCO_Full/trainval"

DS_VAL="voc2007_val"
DS_VAL_JSON="../../../../share/semi_supervised/VOC2007_COCO_Full/annotations/_val.json"
DS_VAL_IMGS="../../../../share/semi_supervised/VOC2007_COCO_Full/trainval"

OUT_FOLDER_VAL="fabian_delete"
YAML_CONFIG_FILE="COCO-Detection/retinanet_R_50_FPN_1x.yaml"
WEIGHTS="COCO-Detection/retinanet_R_50_FPN_1x.yaml"

NUM_CLASSES="20"
EVAL_PERIOD="100"

CUDA_VISIBLE_DEVICES=3 python train_retinanet.py --ds_train $DS_TRAIN  --ds_train_json $DS_TRAIN_JSON  --ds_train_imgs $DS_TRAIN_IMGS --ds_val $DS_VAL  --ds_val_json $DS_VAL_JSON  --ds_val_imgs $DS_VAL_IMGS --out_folder_val $OUT_FOLDER_VAL --yaml_config_file $YAML_CONFIG_FILE --weights $WEIGHTS --num_classes $NUM_CLASSES --eval_period $EVAL_PERIOD