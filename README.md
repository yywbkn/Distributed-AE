# Distributed Autoencoder Classifier Network

# 0.Preface

This repo holds the implementation code of the paper.

*This repository provides code for "Distributed Autoencoder Classifier Network for Small-scale and Scattered COVID-19 Dataset Classification" .



# 1.Introduction

If you have any questions about our paper, feel free to contact us ([Contact](#5-Contact)). 

## 2. Proposed Methods

#### 2.1.Usage

1. Train

   - In the first step, please check if the dataset has been downloaded. 
   - The download link for the dataset is as follows: Click the link to view [Dis-AE-Data.zip]: https://cowtransfer.com/s/4177873fe21546, or visit [CowTransfer.com](http://cowtransfer.com/) and enter the transfer code 2yh62y to access.；
   - The training set is in the COVID_data directory, and the test set is in the COVID_data_test directory. Download the dataset and place it in the respective directories.
     Here are the corresponding Baidu Netdisk links for the dataset. 
   - Next, you can directly run the  `train.py` and just run it! 

2. Test
   - When training is completed, the weights will be saved in `./checkpoints/covid_KL_train/. 
   - Copy the entire "checkpoints" folder to the "test" directory.
   - Go to the "test" directory. you can directly run the  `test.py` and just run it! 
   - Just run it and results will be saved in `./results/'

- 

# 3. Citation

Please cite our paper if you find the work useful: 

```
@article{Yang2023,
title={Distributed Autoencoder Classifier Network for Small-scale and Scattered COVID-19 Dataset Classification},
author={Yang, Yuan and Zhang, Lin and Ren, Lei and Wang, Xiaohan},
journal={International Journal of Imaging Systems and Technology},
year={2023}}
```





### 4. Contact

If you have any questions about our paper,  Feel free to email me(yangyuan@buaa.edu.cn) . 