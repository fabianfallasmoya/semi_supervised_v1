Semi-Supervised Learning Research using [Detectron](https://github.com/facebookresearch/detectron2).

The main code is contained in the folder **uvu**.


## Installation

1. Create an environment python >3.6

2. Install pytorch. Using pip, and cuda 11.1. See https://pytorch.org/
	
3. Install newer gcc compiler (reference: https://seanlaw.github.io/2019/01/17/pip-installing-wheels-with-conda-gcc/). Commands:
  * conda install gcc_linux-64
	* conda install gxx_linux-64

	After installation of gcc, just to make sure everything is correct, run the commands: 
	* echo $CC (You should see the output: /home/builder/anaconda3/envs/cc_env/bin/x86_64-conda_cos6-linux-gnu-cc).
	* echo $CPP (You should see the output: /home/builder/anaconda3/envs/cc_env/bin/x86_64-conda_cos6-linux-gnu-cpp).

4. Install detectron. Commands:
  * git clone https://github.com/facebookresearch/detectron2.git
	* python -m pip install -e detectron2

5. Install OpenCV. Command:
  * pip install opencv-python



## TODO: HOW TO GENERATE DATASETS

Details on how to generate datasets.

## Train

For training you can run train_retinanet.py. There are a lot of parameters that have to be sent in order to train. To keep things easy there is a .sh file that has an example on how to train a model. This is the content of train.sh

```
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

CUDA_VISIBLE_DEVICES=1 python train_retinanet.py 				\\
			--ds_train $DS_TRAIN  					\\
			--ds_train_json $DS_TRAIN_JSON  			\\
			--ds_train_imgs $DS_TRAIN_IMGS 				\\
			--ds_val $DS_VAL  --ds_val_json $DS_VAL_JSON  		\\
			--ds_val_imgs $DS_VAL_IMGS 				\\
			--out_folder_val $OUT_FOLDER_VAL 			\\
			--yaml_config_file $YAML_CONFIG_FILE 			\\
			--weights $WEIGHTS 					\\
			--num_classes $NUM_CLASSES 				\\
			--eval_period $EVAL_PERIOD				\\
```
This example is already working with a dataset contained in the shared folder.

## Test

Similar to training, the testing code is contained in the file train_retinanet.py. Here is the train.sh file that sends the parameters:

```
DS_TEST="voc2007_test"
DS_TEST_JSON="../../../../share/semi_supervised/VOC2007_car_80_20/annotations/val.json"
DS_TEST_IMGS="../../../../share/semi_supervised/VOC2007_Original/VOC_trainval/VOCdevkit/VOC2007/JPEGImages"

YAML_CONFIG_FILE="COCO-Detection/retinanet_R_50_FPN_1x.yaml"
WEIGHTS="output/car_2080/model_0004999.pth"

OUT_FOLDER_VAL="output/car_test_2080"
NUM_CLASSES="1"
EVAL_PERIOD="100"

CUDA_VISIBLE_DEVICES=2 python test_retinanet.py 		\\
			--ds_test $DS_TEST  			\\
			--ds_test_json $DS_TEST_JSON  		\\
			--ds_test_imgs $DS_TEST_IMGS 		\\
			--yaml_config_file $YAML_CONFIG_FILE 	\\
			--weights $WEIGHTS 			\\
			--out_folder_val $OUT_FOLDER_VAL 	\\
			--num_classes $NUM_CLASSES		\\
```

Results for the AP (Average Precision) will be shown in terminal.
