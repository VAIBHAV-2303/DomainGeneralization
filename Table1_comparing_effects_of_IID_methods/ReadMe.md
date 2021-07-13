# Description

Code to get results on PACS dataset.

# How To Run

Configure the backbone/optimizer of your choice by commenting/uncommenting in the run.py file. There is also commented code at the bottom of run.py which can help in making useful plots and saving the models for future inference.

```bash
pip3 install -r requirements.txt
python3 run.py

```

* Note: For proper data loading, the PACS dataset should be present in the directory in the following manner: "data/images/\<domain>/\<class>/\<actual_images>"

# Libraries Used

* numpy==1.19.5
* torchvision==0.9.1
* matplotlib==3.4.2(optional)
* torch==1.8.1
* Pillow==8.2.0

# Authors

* Vaibhav Garg
* Akshay Goindani
* Sarath S.
