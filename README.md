Performance Enhancement Method in CNN-based STEMI Detection from 12-lead Electrocardiogram
=====

This repository provides the source code of the paper "Performance Enhancement Method in CNN-based STEMI Detection from 12-lead Electrocardiogram".

## STEMI: ST-elevation myocardial infarction
The complete occlusion of the coronary artery is called STEMI and it is characterized by a sudden shut-down of blood flow caused by thrombus or embolism. In the ECG of STEMI, the ST-segment is more elevated than the normal.

<div align="center">
  <img src="./figures/normal.png" width="500">  
  <img src="./figures/stemi.png"  width="500">  
  <p>The ECG of normal (upper) and STEMI (lower)</p>
</div>

## Requirements
* Python 3.5.2  
* Tensorflow 1.4.0  
* Numpy 1.13.3  
* Scipy 1.2.0  
* WFDB 2.2.1  
* Matplotlib 3.0.2  


## Usage
### Preparing the dataset
First, Organize the dataset and keep as below. Keep the channel (or lead) information in axis 0, and time information in axis 1. For example, 12 lead data with 5500-time length (500Hz x 11 seconds) should be saved as `(12, 5500)` form.  
```
Raw_ECG
├── Normal
│   ├── data_1
│   ├── data_2
│   ├── data_3
│   │     ...
│   └── data_n
└── STEMI
```
Then, run the python script as following.  
```
$ python preprocessing.py
$ python preprocessing.py --help // for confirming the option
```

### Training and Test
```
$ cd CNN
$ python run_resnet.py
```
Using `run_vggnet.py` instead of `run_resnet.py` is available for training VGGNet.  

<div align="center">
  <img src="./figures/vggnet16.png" width="800">  
  <p>1D-VGGNet-16</p>  
  <img src="./figures/resnet34.png"  width="800">  
  <p>1D-ResNet-34</p>
</div>

Available since Mar.31.2019  
Developed by <a href="https://github.com/YeongHyeon"><strong>YeongHyeon Park</strong><a>
