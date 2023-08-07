# [Siggraph Asia 2023]Low-light Image Enhancement with Wavelet-based Diffusion Models [[Paper]](https://arxiv.org/pdf/2306.00306.pdf).
<h4 align="center">Hai Jiang<sup>1,2</sup>, Ao Luo<sup>2</sup>, Songchen Han<sup>1</sup>, Haoqiang Fan<sup>2</sup>, Shuaicheng Liu<sup>3,2</sup></center>
<h4 align="center">1.Sichuan University, 2.Megvii Technology, 
<h4 align="center">3.University of Electronic Science and Technology of China</center></center>

## Presentation video:  
[[Youtube]]() and [[Bilibili]]()

## Pipeline
![](https://github.com/JianghaiSCU/DiffLL/blob/main/Figures/pipeline.png)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### Paired datasets 
LOLv1 dataset: Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep Retinex Decomposition for Low-Light Enhancement", BMVC, 2018. [[Baiduyun (extracted code: sdd0)]](https://pan.baidu.com/s/1spt0kYU3OqsQSND-be4UaA) [[Google Drive]](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing)

LOLv2 dataset: Wenhan Yang, Haofeng Huang, Wenjing Wang, Shiqi Wang, and Jiaying Liu. "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement", TIP, 2021. [[Baiduyun (extracted code: l9xm)]](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g) [[Google Drive]](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing)

LSRW dataset: Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network", Journal of Visual Communication and Image Representation, 2023. [[Baiduyun (extracted code: wmrr)]](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)

### Unpaired datasets 
Please refer to [[Project Page of RetinexNet.]](https://daooshee.github.io/BMVC2018website/)

## Pre-trained Models 
You can downlaod our pre-trained model from [[Google Drive]](https://drive.google.com/file/d/1f4zDvPsWKrID33OJdeHwc5VOBILkm0KW/view?usp=sharing) and [[Baidu Yun (extracted code:wsw7)]](https://pan.baidu.com/s/1rq8VzdnHeky0iT56coOGog)

## How to train?
You need to modify ```datasets/dataset.py``` slightly for your environment, and then
```
python train.py  
```

## How to test?
```
python evaluate.py
```

## Visual comparison
![](https://github.com/JianghaiSCU/DiffLL/blob/main/Figures/comparison.png)

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{jiang2023low,
  title={Low-Light Image Enhancement with Wavelet-based Diffusion Models},
  author={Jiang, Hai and Luo, Ao and Han, Songchen and Fan, Haoqiang and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2306.00306},
  year={2023}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion), [SDWNet](https://github.com/FlyEgle/SDWNet), and [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). We thank all the authors for their contributions.
