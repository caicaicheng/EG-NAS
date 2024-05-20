## EG-NAS

This is the official pytorch implementation for the paper: [*EG-NAS: Neural Architecture Search with Fast Evolutionary Exploration*](https://ojs.aaai.org/index.php/AAAI/article/view/28993), 
which is accepted by AAAI2024. This repo contains the implementation of architecture search and evaluation on CIFAR-10 and ImageNet using our proposed EG-NAS.

![intro](https://github.com/caicaicheng/EG-NAS/blob/main/figs/EG-NAS.png)

## Quick Start

### Prerequisites

- python>=3.5
- pytorch>=1.1.0
- torchvision>=0.3.0 
- pip install cmaes

## Usage

### Architecture Search on CIFAR-10

To search CNN cells on CIFAR-10, please run
```
export CUDA_VISIBLE_DEVICES=0
python train_search.py    \
--batch_size 256    \
--data /path/to/cifar10
```

 
### Architecture Search on ImageNet
To search CNN cells on ImageNet, please run
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_search_imagenet.py    \
--batch_size 1024    \
--data /path/to/imagennet  \
```



### Architecture Evaluation on CIFAR-10
To evaluate the derived architecture on CIFAR-10, please run
```
export CUDA_VISIBLE_DEVICES=0
python train.py   \
--data /path/to/cifar10 \
--save train_cifar10   \
--auxiliary \
--cutout    \

```


### Architecture Evaluation on ImageNet
To evaluate the derived architecture on ImageNet, please run
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_imagenet.py \
 --tmp_data_dir /path/to/imagenet \
 --save train_imagenet \
 --workers 16   \
 --auxiliary \
 --note imagenet_shapley    \
```


## Citation

Please cite our paper if you find it useful in your research:
```
@article{Cai_Chen_Liu_Ling_Lai_2024, 
    title={EG-NAS: Neural Architecture Search with Fast Evolutionary Exploration}, 
    volume={38}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/28993}, 
    DOI={10.1609/aaai.v38i10.28993}, number={10}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Cai, Zicheng and Chen, Lei and Liu, Peng and Ling, Tongtao and Lai, Yutao}, 
    year={2024}, 
    month={Mar.}, 
    pages={11159-11167} 
}
```

## Acknowledgements

We thank the authors of following works for opening source their excellent codes.

- [DARTS](https://github.com/quark0/darts)
- [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS)
- [DARTS-PT](https://github.com/ruocwang/darts-pt)
- [Shapley-NAS](https://github.com/Euphoria16/Shapley-NAS)