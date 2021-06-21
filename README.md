HDR reconstruction using single image LDR
========================================

這次專題主要用了以下這篇paper:  
**FHDR: HDR Image Reconstruction from a SingleLDR Image using Feedback Network**<br>([FHDR](https://arxiv.org/abs/1912.11463))
Zeeshan Khan, Mukul Khanna, Shanmuganathan Raman<br>
2019 IEEE Global Conference on Signal and Information Processing

此次專題的結果:
([Results](https://drive.google.com/drive/folders/11EiU4q3OWokiYXWimu-1csjfYRuYw7qk))  

Table of contents:
-----------

- [Setup](#setup)
    - [Environment](#Environment)
    - [Prerequisites](#pre-requisites)
- [Dataset](#dataset)
- [Training](#training)
- [Pretrained models](#pretrained-models)
- [Evaluation](#evaluation)
- [Citation](#citation)


Setup
-----
### Environment
- Ubuntu 18.04
- Pytorch 1.3.1
- Cuda 10.2
- Geforce 1080 x1
### Pre-requisites

- Python3
- [PyTorch](https://pytorch.org/)
- GPU, CUDA, cuDNN
- [OpenCV](https://opencv.org)
- [PIL](https://pypi.org/project/Pillow/)
- [Numpy](https://numpy.org/)
- [scikit-image](https://scikit-image.org/)
- [tqdm](https://pypi.org/project/tqdm/)

**`requirements.txt`** has been provided for installing Python dependencies.

```sh
pip install -r requirements.txt
```

Dataset
--------

The dataset is to comprise of LDR (input) and HDR (ground truth) image pairs. The network is trained to learn the mapping from LDR images to their corresponding HDR ground truth counterparts.

The dataset should follow the following folder structure - 

```
> dataset

    > train

        > LDR

            > ldr_image_1.jpg/png
            > ldr_image_2.jpg/png
            .
            .

        > HDR

            > hdr_image_1.hdr/exr
            > hdr_image_2.hdr/exr
            .
            .

    > test
    
        > LDR

            > ldr_image_1.jpg/png
            > ldr_image_2.jpg/png
            .
            .

        > HDR

            > hdr_image_1.hdr/exr
            > hdr_image_2.hdr/exr
            .
            .

```
- Other training set can be download from:  
    -https://github.com/alex04072000/SingleHDR  
    -https://github.com/HanbyolJang/LDR-HDR-pair_Dataset
- Sample test datasets can be downloaded here - 
    - [512x512 size images](https://drive.google.com/open?id=1tv8kdeoT12AJL2iMnQkNUfgY2RjirNp9)
    - [256x256 size images](https://drive.google.com/open?id=1KQCLpXwRshmrUi10oG1aPNvOCExeCGv5)

- For evaluating on this dataset, download and unzip the folder, replace it with the `test` directory in the `dataset` folder, and refer to [Pretrained models](#pretrained-models) and [Evaluation](#evaluation).

**Note:** The pre-trained models were trained on 256x256 size images.

Training
--------

After the dataset has been prepared, the model can be trained using the **`train.py`** file.

```sh
python3 train.py
```

The corresponding parameters/options for training have been specified in the **`options.py`** file and can be easily altered. They can be logged using -

```sh
python3 train.py --help
```
- **`--iter`** param is used to specify the number of feedback iterations for global and local feedback mechanisms (refer to paper/architecture diagram)
- Checkpoints of the model are saved in the **`checkpoints`** directory. (Saved after every 2 epochs by default)
- GPU is used for training. Specify GPU IDs using **`--gpu_ids`** param.
- The iter-1 model takes around 2.5 days to train on a dataset of 12k images on an RTX 2070 SUPER GPU.

Pretrained models
---------------------------

Pre-trained models can be downloaded from the below-mentioned links. 

These models have been trained with the default options, on 256x256 size images for 200 epochs, in accordance with the paper.

- [Feed-forward (1-Iteration) model](https://drive.google.com/file/d/1iTSU-tsencVgefH8oNorf9JExGKylaXo/view?usp=sharing)
- [2-Iterations model](https://drive.google.com/open?id=13vTGH-GVIWVL79X8NJra0yiguoO1Ox4V)


Evaluation/Inference
----------

The performance of the network can be evaluated using the **`test.py`** file - 

```sh
python3 test.py --ckpt_path /path/to/pth/checkpoint
```

- Test results (LDR input, HDR prediction and HDR ground truth) are stored in the **`test_results`** directory.
- HDR images can be viewed using [OpenHDRViewer](https://viewer.openhdr.org).
- If checkpoint path is not specified, it defaults to `checkpoints/latest.ckpt` for evaluating the model.
- PSNR and SSIM scores can be logged for quantitative evaluation by -

```sh
python3 test.py --log_scores
```

Citation
----------
If you use this code for your research, please cite the following [paper](http://arxiv.org/abs/1912.11463). 

```
@INPROCEEDINGS{8969167,
    author={Z. {Khan} and M. {Khanna} and S. {Raman}},
    booktitle={2019 IEEE Global Conference on Signal and Information Processing (GlobalSIP)},
    title={FHDR: HDR Image Reconstruction from a Single LDR Image using Feedback Network},
    year={2019},
    pages={1-5},
    doi={10.1109/GlobalSIP45357.2019.8969167}
}
```

