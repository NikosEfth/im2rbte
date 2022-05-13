# Edge Augmentation for Large-Scale Sketch Recognition without Sketches

This is the official implementation of the method proposed in the paper [Edge Augmentation for Large-Scale Sketch Recognition without Sketches](https://arxiv.org/abs/2202.13164). 

![image](https://user-images.githubusercontent.com/11415657/168291007-4b690233-19a3-47a7-b9e6-7132bb26058f.png)

## Quick Description
The goal of this work is to recognize sketches at test time (bottom row) without any sketches at training time. Labeled natural images (top row) are transformed to rBTEs with different level of details (the two middle rows are two instances of rBTE per natural image, thickened for visualization) to bridge the domain gap. Combined with geometric augmentations, the transformed dataset is used to train a deep network that is able to classify sketche

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/11415657/168294110-2b7c0443-c17f-45ba-b0d3-3bb4ae3c8155.png">
</p>

## Dependencies

* Install [PyTorch](http://pytorch.org/) and [Torchvision](http://pytorch.org/)
* Install yaml: `pip install pyyaml`
* Install PIL: `pip install Pillow`
* Install NumPy: `pip install numpy`
* Install OpenCV: `pip install opencv-contrib-python`
* Install Scikit-Image: `python -m pip install -U scikit-image`
* Tested with Python 3.5.3 and PyTorch 1.3.1

## Downloader

To download all the datasets and the pretrained models needed:

```
python downloader.py 
```

* with `--dataset` you specify a dataset to be downloaded, ie: `python downloader.py --dataset pacs`. By specifying "domainnet", "imagenet", "pacs", "sketchy", or "tu-berlin", the im4sketch part of the corresponding dataset is downloading. By specifying nothing, the default is to download the whole im4sketch
* with `--download` and `--extract` you specify if the chosen dataset is for downloading, extracting it from the tar files, or both ie `python downloader.py --dataset pacs --download no --extract yes`
* with `--delete` you choose if you want the tar files deleted after the extraction (default is no) ie `python downloader.py --delete yes`
* with `--models` you specify if you want the nms model (mandatory for the method) and the im4sketch pretrained model (mandatory for id5 run) to be downloaded `python downloader.py --models yes`

Or download manualy [here](http://ptak.felk.cvut.cz/im4sketch/)

## Method run

To run an experiment specified by the corresponding yaml file:

```
python method.py --run run_im4sketch_id5.yaml
```
```
python method.py --run run_sketchy_id5.yaml
```

## Im4Sketch

For more information about the Im4Sketch dataset please visit the [Im4Sketch](http://cmp.felk.cvut.cz/im4sketch/) dataset webpage

![image](https://user-images.githubusercontent.com/11415657/168289673-7ab8104c-e826-47b2-865d-e8d1b76d8581.png)


## External code

For the creation of the Im4Sketch dataset and its sub-datasets used for our experiments, we use the following code:

* BDCN official code: https://github.com/pkuCactus/BDCN
* HED reimplementation in python: https://github.com/sniklaus/pytorch-hed
* Structured Forests official code: https://github.com/pdollar/toolbox
