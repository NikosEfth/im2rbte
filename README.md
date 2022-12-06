# Edge Augmentation for Large-Scale Sketch Recognition without Sketches
  
This is the official implementation of the method proposed in the ICPR 2022 paper [Edge Augmentation for Large-Scale Sketch Recognition without Sketches](https://arxiv.org/abs/2202.13164). 

```
@inproceedings{im2rBTE2022,
 title = {Edge Augmentation for Large-Scale Sketch Recognition without Sketches},
 author = {Efthymiadis, N. and Tolias, G. and Chum, O.},
 booktitle = {ICPR},
 year = {2022}
}
```

## Quick Description
  
![Untitled3](https://user-images.githubusercontent.com/11415657/168404222-f65833d6-9e49-4dc5-8e83-21112ea2b6ce.jpg)

The goal of this work is to recognize sketches at test time without using any sketches at training time. Labeled natural images are transformed to a pseudo-novel domain called “randomized Binary Thin Edges” (rBTEs), with different level of details to bridge the domain gap. Combined with geometric augmentations, the transformed dataset is used to train a deep network that is able to classify sketches. 

For the training, the novel dataset Im4Sketch is introduced as a superset of already existing natural-image and sketch datasets. The classes are regrouped according to shape criteria. E.g. the class bear of Im4Sketch contains the original ImageNet classes “American Black Bear” and “Ice Bear” whose shape is indistinguishable. Sketches are collected from original sketch datasets with different level of detail.

## Pipeline

![168291007-4b690233-19a3-47a7-b9e6-7132bb26058f](https://user-images.githubusercontent.com/11415657/168404210-18e3fd1b-2788-4acb-83c0-ac24ddd49571.jpg)

Overview of the training pipeline. Natural images are transformed into rBTEs, which are used with class labels to train a network classifier with
cross-entropy loss. The obtained network is used to classify free-hand sketches into the object categories

## Dependencies

* Install [PyTorch](http://pytorch.org/) and [Torchvision](http://pytorch.org/)
* Install yaml: `pip install pyyaml`
* Install PIL: `pip install Pillow`
* Install NumPy: `pip install numpy`
* Install OpenCV: `pip install opencv-contrib-python`
* Install Scikit-Image: `python -m pip install -U scikit-image`
* Tested with Python 3.8.6, PyTorch 1.9.0, Torchvision 0.10.0, CUDA 11.1, cuDNN 8.0.5.39 on a Tesla P100-PCIE-16GB

## Downloader

To download the Im4Sketch dataset as well as the pretrained models needed for the experiments:

```
python downloader.py 
```

* with `--dataset` you can choose a specific Im4Sketch sub-dataset to be downloaded, e.g. `python downloader.py --dataset pacs`. The choices are: "domainnet", "imagenet", "pacs", "sketchy", or "tu-berlin". The default is to download the whole im4sketch.
* with `--download` and `--extract` you specify if the chosen dataset is for downloading, extracting it from the tar files, or both i.e. `python downloader.py --dataset pacs --download no --extract yes`
* with `--delete` you choose if you want the tar files deleted after the extraction (default is no) i.e. `python downloader.py --delete yes`
* with `--models` you specify if you want the nms model (mandatory for all experiments) and the Im4Sketch pretrained model (mandatory for Table 3 - id5 run) to be downloaded i.e. `python downloader.py --models yes`

Or download manually [here](http://ptak.felk.cvut.cz/im4sketch/)

## Experiments

The results of the paper can be reproduced as below. The gpu can be specified by e.g. `--gpu 0`.

Table 3 - Sketchy Ablations

```
python method.py --run run_sketchy_id1.yaml
```
```
python method.py --run run_sketchy_id2.yaml
```
```
python method.py --run run_sketchy_id3.yaml
```
```
python method.py --run run_sketchy_id4.yaml
```
```
python method.py --run run_sketchy_id5.yaml
```
```
python method.py --run run_sketchy_id6.yaml
```
```
python method.py --run run_sketchy_id7.yaml
```

Table 4 - PACS Generalization

```
python method.py --run run_PACS_generalization.yaml
```

Table 5 - Im4Sketch

```
python method.py --run run_im4sketch.yaml
```

The last experiment reproduces the downloadable `./Pretrained_Models/im4sketch_model.pt` which is mandatory for the Table 3 - id5 experiment. If the file is not downloaded and instead is executed, then the renaming and copying to the aforementioned folder should be done manually.  

## External Code

For the creation of the Im4Sketch dataset and its sub-datasets used for our experiments, we use the following code:

* BDCN official code: https://github.com/pkuCactus/BDCN
* HED reimplementation in python: https://github.com/sniklaus/pytorch-hed
* Structured Forests official code: https://github.com/pdollar/toolbox

## Im4Sketch Dataset

For more information about the Im4Sketch dataset please visit the [Im4Sketch](http://cmp.felk.cvut.cz/im4sketch/) dataset webpage

![Im4Sketch2](https://user-images.githubusercontent.com/11415657/171275859-3bc572c6-8e4f-4d8a-b5bf-bceed1327704.jpg)


