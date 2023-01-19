## SSTAR-LC

<p align="center">
    <img src="sources/method.jpg" alt="drawing" width="800"/>
</p>


This is the official PyTorch implementation of 2023 paper   (SXXX). 

### Abstract
XXXXXXXXXXXX.

### Preparation
- pytorch
- tqdm

### Usage
Example runs on CIFAR100 dataset with symmetric noise:
```
python main_cifar.py --theta_r 0.8 --noise_mode sym --noise_ratio 0.9 --dataset cifar100 --dataset_path path_to_cifar100

python main_cifar.py --theta_r 0.8 --noise_mode sym --noise_ratio 0.8 --dataset cifar100 --dataset_path path_to_cifar100

python main_cifar.py --theta_r 0.8 --noise_mode sym --noise_ratio 0.5 --dataset cifar100 --dataset_path path_to_cifar100

python main_cifar.py --theta_r 0.9 --noise_mode sym --noise_ratio 0.2 --dataset cifar100 --dataset_path path_to_cifar100
```

Example runs on WebVision/Clothing1M dataset~(Approximately ~16GB memory for clothing1M and ~36GB for WebVision, considering reduce the batch size or parallel training with more GPUs if you don't have enough GPU memory.):
```
python main_webvision.py --dataset_path ./WebVision --gpuid 0,1 --parallel

python main_clothing1m.py --dataset_path ./Clothing1M --gpuid 0
```

For users who are not familiar with wandb, please try `main_cifar_simple.py` with same config.


### Results
| Dataset       |  CIFAR10  |  CIFAR10  |  CIFAR10  |  CIFAR10  |  CIFAR10   | CIFAR100  | CIFAR100  | CIFAR100  | CIFAR100  |
| ------------- | :-------: | :-------: | :-------: | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: |
| Noise type    | Symmetric | Symmetric | Symmetric | Symmetric | Assymetric | Symmetric | Symmetric | Symmetric | Symmetric |
| Noise ratio   |    20%    |    50%    |    80%    |    90%    |    40%     |    20%    |    50%    |    80%    |    90%    |
| Cross-Entropy |   86.8    |   79.4    |   62.9    |   42.7    |    85.0    |   62.0    |   46.7    |   19.9    |   10.1    |
| Co-teaching+  |   89.5    |   85.7    |   67.4    |   47.9    |     -      |   65.6    |   51.8    |   27.9    |   13.7    |
| F-correction  |   86.8    |   79.8    |   63.3    |   42.9    |    87.2    |   61.5    |   46.6    |   19.9    |   10.2    |
| PENCIL        |   92.4    |   89.1    |   77.5    |   58.9    |    88.5    |   69.4    |   57.5    |   31.1    |   15.3    |
| LossModelling |   94.0    |   92.0    |   86.8    |   69.1    |    87.4    |   73.9    |   66.1    |   48.2    |   24.3    |
| DivideMix*    |   96.1    |   94.6    |   93.2    |   76.0    |    93.4    |   77.3    |   74.6    |   60.2    |   31.5    |
| ELR+*         |   95.8    |   94.8    |   93.3    |   78.7    |    93.0    |   77.6    |   73.6    |   60.8    |   33.4    |
| RRL           |   95.8    |   94.3    |   92.4    |   75.0    |    91.9    |   79.1    |   74.8    |   57.7    |   29.3    |
| NGC           |   95.9    |   94.5    |   91.6    |   80.5    |    90.6    |   79.3    |   75.9    |   62.7    |   29.8    |
| AugDesc*      |   96.3    |   95.4    |   93.8    |   91.9    |    94.6    |   79.5    |   77.2    |   66.4    |   41.2    |
| C2D*          |   96.4    |   95.3    |   94.4    |   93.6    |    93.5    |   78.7    |   76.4    |   67.8    |   58.7    |
| (ours)        |   96.3    |   95.7    |   95.2    |   94.6    |    95.1    |   79.0    |   75.9    |   69.5    |   61.8    |
| (ours)        | **96.7**  | **96.1**  | **95.6**  | **95.2**  |  **95.5**  | **79.7**  | **77.2**  | **71.9**  | **66.6**  |

### Running the code on CIFAR-10/100:

We provide the code used to simulate CIFAR-10/100 datasets with symmetric and asymmetric label noise and provide example scripts to run our approach with both noises.

To run the code use the provided scripts in CIFAR folders. Datasets are downloaded automatically when setting "--download True". The dataset has to be placed in dataset folder (should be done automatically). The training has two stages: .

We also provide another version of SSTAR-LC training code on the first training stage, which reduces the computational complexity of selection and can be applied on large-scale datasets. It can be used by running train run.py.

### Running the code on WebVision-50:

We provide the code and scripts used to evaluate our approach on WebVision-50 dataset. The dataset should be downloaded and placed into the data directory before running code.

### Citation:

If you find the code useful in your research, please consider citing our paper:

```
 @InProceedings{Hao,
  title = {Sxxxx with Noisy Labels},
  authors = {Hao and xxx},
  year={2023},
  booktitle ={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 } 
```

Note: Our implementation uses parts of some public codes [1-3].

[1] Co-teaching [https://github.com/bhanML/Co-teaching](https://github.com/bhanML/Co-teaching)

[2] MoCo https://github.com/facebookresearch/moco

[3] RRL https://github.com/salesforce/RRL/








