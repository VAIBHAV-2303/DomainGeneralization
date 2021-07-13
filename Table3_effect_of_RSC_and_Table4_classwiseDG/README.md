## Installation

### Requirements:
Install all the dependencies using the command:
```bash
  pip3 install -r requirements.txt
```

## Data Preparation
Download PACS dataset from [here](https://github.com/MachineLearning2020/Homework3-PACS/tree/master/PACS) and the VLCS dataset from [here](http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file).  

## Runing on PACS dataset
To train the models, run the below commands with appropriate settings
```bash
  python3 train.py --net {model_name} --optimizer {sgd / adam} --source {List of Domains to use for training} --target {List of Domains to use for testing} --root_dir {path to directory containing domain wise images}
```
Choices for model_name are - resnet18, resnet50, inception.
In order to use RSC, add "--rsc True" to the above command.
For example to train a resnet18 model using art_painting, photo and cartoon domain, using sgd optimizer and in RSC setting, run:
```bash
  python3 train.py --net resnet18 --optimizer sgd --rsc --source art_painting cartoon photo --target sketch --root_dir ./data/images/
```
For training and testing a model in classwise-DG (Proposed in the Paper) setting, run:
```bash
  python3 non_pacs_train.py --net {model_name} --optimizer {sgd / adam} --root_dir {path to directory containing images} --root_dir {path to directory containing domain wise images}
```
In order to use RSC, add "--rsc True" to the above command.

## Runing on VLCS dataset
To train the models, run the below commands with appropriate settings
```bash
  python3 vlcs_train.py --net {model_name} --optimizer {sgd / adam} --rsc {0 / 1} --source {List of Domains to use for training} --target {List of Domains to use for testing} --root_dir {path to directory containing domain wise images}
```
For example to train a resnet18 model using CALTECH, LABELME and PASCAL domain, using sgd optimizer and in RSC setting, run:
```bash
  python3 train.py --net resnet18 --optimizer sgd --rsc 1 --source CALTECH LABELME PASCAL --target SUN --root_dir {path to directory containing domain wise images}
```
In order to use RSC, add "--rsc True" to the above command.
