Semi-Supervised Learning Research using [Detectron](https://github.com/facebookresearch/detectron2)



### 
* Includes new capabilities such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, etc.
* Used as a library to support building [research projects](projects/) on top of it.
* Models can be exported to TorchScript format or Caffe2 format for deployment.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

1. Create an environment python >3.6

2. Install pytorch. Using pip, and cuda 11.1. See https://pytorch.org/
	
3. Install newer gcc compiler (reference: https://seanlaw.github.io/2019/01/17/pip-installing-wheels-with-conda-gcc/). Commands:
  * conda install gcc_linux-64
	* conda install gxx_linux-64

  After installation of gcc, just to make sure everything is correct, run the commands: 
	* echo $CC (You should see the output: /home/builder/anaconda3/envs/cc_env/bin/x86_64-conda_cos6-linux-gnu-cc).
	* echo $CPP (/home/builder/anaconda3/envs/cc_env/bin/x86_64-conda_cos6-linux-gnu-cpp).

4. Install detectron. Commands:
  * git clone https://github.com/facebookresearch/detectron2.git
	* python -m pip install -e detectron2

4. Install OpenCV. Command:
  * pip install opencv-python



## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
