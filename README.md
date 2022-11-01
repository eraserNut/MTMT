# A Multi-task Mean Teacher for Semi-supervised Shadow Detection

by Zhihao Chen, Lei Zhu, Liang Wan, Song Wang, Wei Feng, and Pheng-Ann Heng [[paper link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Multi-Task_Mean_Teacher_for_Semi-Supervised_Shadow_Detection_CVPR_2020_paper.pdf)]

#### News: In 2020.9.17, We release the unsorted code for other researchers. The sorted code will be released after.

***

## Citation
@inproceedings{chen20MTMT,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Chen, Zhihao and Zhu, Lei and Wan, Liang and Wang, Song and Feng, Wei and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {A Multi-task Mean Teacher for Semi-supervised Shadow Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {CVPR},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2020}    
}

## Shadow detection results at test datasets
The results of shadow detection(w & w/o crf) on three datasets (SBU, UCF, ISTD) can be found 
at [Google Drive](https://drive.google.com/file/d/1BK4x9IUNQKBaP7ye5S2-e9_zEN7FbZUg/view?usp=sharing) or [BaiduNetdisk](https://pan.baidu.com/s/1Rdp8rQbj5f7Id4JJj99nxw)(password:131b for BaiduNetdisk).

## Trained Model
You can download the trained model which is reported in our paper at [BaiduNetdisk](https://pan.baidu.com/s/1yjnsjE7mDPnEaHxdtNFhhQ)(password: h52i) or [Google Drive](https://drive.google.com/file/d/1s-4BSmz9j8u2_WoUnzNYL0QjRYFEeEkU/view?usp=share_link).

## Requirement
* Python 3.6
* PyTorch 1.3.1(After 0.4.0 would be ok)
* torchvision
* numpy
* tqdm
* PIL
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Training
1. Set ...
2. Set ...
3. Run by ```python train.py```

The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.

## Testing
1. Set ...
2. Put ...
2. Run by ```python test_MT.py```

## Useful links
UCF dataset: [Google Drive](https://drive.google.com/open?id=12DOmMVmE-oNuJVXmkBJrkfBvuDd0O70N) or [BaiduNetdisk](https://pan.baidu.com/s/1zt9ya1lzNcoGoc2CET3mdg)(password:o4ub for BaiduNetdisk)

SBU dataset: [BaiduNetdisk](https://pan.baidu.com/s/1FYQYLSkuTivjaRJVjjJhJw)(password:38qw for BaiduNetdisk)

Part of unlabel data that collected from internet: [Google Drive](https://drive.google.com/drive/folders/1HZpR3SAVv3A8jtW1-l9v0Caz9I0UwRZW?usp=sharing) or [BaiduNetdisk](https://pan.baidu.com/s/1_kdpwBlZ-K6gcZz45Tcg7g)(password: n1nb for BaiduNetdisk)
