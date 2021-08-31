# Large-Scale Sketch Classification

## Downloader

To download all the datasets and the pretrained models needed from http://ptak.felk.cvut.cz/im4sketch/
```
python downloader.py 
```
* with `--dataset` you specify a dataset ie `python downloader.py --dataset pacs`
* with `--download` and `--extract` you specify if the chosen dataset is for downloading, extracting or both ie `python downloader.py --dataset pacs --download no --extract yes`
* with `--delete` you choose if you want the tar files deleted after the extraction (default is no) ie `python downloader.py --delete yes`
* with `--models` you specify if you want the nms model (mandatory for the method) and the im4sketch pretrained model (mandatory for id5 run) to be downloaded `python downloader.py --models yes`

## Method run

To run an experiment specified by the corresponding yaml file
```
python method.py --run run_sketchy_id5.yaml
```

