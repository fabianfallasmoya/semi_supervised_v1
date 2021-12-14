DS_TEST="voc2007_test"
DS_TEST_JSON="../../../../share/semi_supervised/VOC2007_car_80_20/annotations/val.json"
DS_TEST_IMGS="../../../../share/semi_supervised/VOC2007_Original/VOC_trainval/VOCdevkit/VOC2007/JPEGImages"

YAML_CONFIG_FILE="COCO-Detection/retinanet_R_50_FPN_1x.yaml"
WEIGHTS="output/car_2080/model_0004999.pth"

OUT_FOLDER_VAL="output/car_test_2080"
NUM_CLASSES="1"
EVAL_PERIOD="100"

CUDA_VISIBLE_DEVICES=2 python test_retinanet.py --ds_test $DS_TEST  --ds_test_json $DS_TEST_JSON  --ds_test_imgs $DS_TEST_IMGS --yaml_config_file $YAML_CONFIG_FILE --weights $WEIGHTS --out_folder_val $OUT_FOLDER_VAL --num_classes $NUM_CLASSES