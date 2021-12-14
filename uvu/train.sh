DS_TRAIN="voc2007_train"
DS_TRAIN_JSON="../../../../share/semi_supervised/VOC2007_car_80_20/annotations/train.json"
DS_TRAIN_IMGS="../../../../share/semi_supervised/VOC2007_Original/VOC_trainval/VOCdevkit/VOC2007/JPEGImages"

DS_VAL="voc2007_val"
DS_VAL_JSON="../../../../share/semi_supervised/VOC2007_car_80_20/annotations/val.json"
DS_VAL_IMGS="../../../../share/semi_supervised/VOC2007_Original/VOC_trainval/VOCdevkit/VOC2007/JPEGImages"

OUT_FOLDER_VAL="output/car"
YAML_CONFIG_FILE="COCO-Detection/retinanet_R_50_FPN_1x.yaml"
WEIGHTS="COCO-Detection/retinanet_R_50_FPN_1x.yaml"

NUM_CLASSES="1"
EVAL_PERIOD="100"

CUDA_VISIBLE_DEVICES=1 python train_retinanet.py --ds_train $DS_TRAIN  --ds_train_json $DS_TRAIN_JSON  --ds_train_imgs $DS_TRAIN_IMGS --ds_val $DS_VAL  --ds_val_json $DS_VAL_JSON  --ds_val_imgs $DS_VAL_IMGS --out_folder_val $OUT_FOLDER_VAL --yaml_config_file $YAML_CONFIG_FILE --weights $WEIGHTS --num_classes $NUM_CLASSES --eval_period $EVAL_PERIOD