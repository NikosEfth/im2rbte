# Edge Augmentation for Large-Scale Sketch Recognition without Sketches

This is the official implementation of the method proposed in the paper [Edge Augmentation for Large-Scale Sketch Recognition without Sketches](https://arxiv.org/abs/2202.13164). 
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
