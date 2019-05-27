# Global-Second-order-Pooling-Convolutional-Networks
Global Second-order Pooling Convolutional Networks (cvpr2019 GSoP)

This is an implementation of GSoP Net([paper](https://arxiv.org/pdf/1811.12006.pdf)) , created by [Zilin Gao](https://github.com/zilingao).

![GSoP_arch](fig/GSoP_arch.png)
## Introduction

Deep Convolutional Networks (ConvNets) are fundamental to, besides large-scale visual recognition, a lot of vision tasks. As the primary goal of the ConvNets is to
characterize complex boundaries of thousands of classes in a high-dimensional space, it is critical to learn higherorder representations for enhancing non-linear modeling
capability. Recently, Global Second-order Pooling (GSoP), plugged at the end of networks, has attracted increasing attentions,
achieving much better performance than classical, first-order networks in a variety of vision tasks. However, how to effectively introduce higher-order representation in
earlier layers for improving non-linear capability of ConvNets is still an open problem. In this paper, we propose a novel network model introducing GSoP across from lower
to higher layers for exploiting holistic image information throughout a network. Given an input 3D tensor outputted by some previous convolutional layer, we perform GSoP to
obtain a covariance matrix which, after nonlinear transformation, is used for tensor scaling along channel dimension. Similarly, we can perform GSoP along spatial dimension
for tensor scaling as well. In this way, we can make full use of the second-order statistics of the holistic image throughout
a network. The proposed networks are thoroughly evaluated on large-scale ImageNet-1K, and experiments have shown that they outperform non-trivially the counterparts
while achieving state-of-the-art results.


## Citation

     @InProceedings{Gao_2019_CVPR,
                    author = {Zilin, Gao and Jiangtao, Xie and Qilong, Wang and Peihua, Li},
                    title = {Global Second-order Pooling Convolutional Networks},
                    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                    year = {2018}
      }


## GSoP Block

![GSoP_block](fig/GSoP_block.png)
We design a GSoP block to introduce global second-order pooling into intermediate layers of deep ConvNets,
which goes beyond the existing works where GSoP can only be used at network end. By modeling higher-order statistics of
holistic images at earlier stages, our network can enhance capability of non-linear representation learning of deep networks.

![GSoP_block_table](fig/GSoP_block_table.png)

## GSoP Nets

Based on GSoP block, we propose two GSoP networks parallel. In the two GSoP networks, GSoP blocks are embedded into the end of residual stages.

### GSoP-Net1

![GSoP_Net1](fig/GSoP_Net1.png)

In GSoP-Net1, each residual stage is attached by one GSoP block, to be specific, GSoP-Net1 with ResNet-50 backbone employs 4 GSoP blocks.
The final pooling layer keeps the original global average pooling in backbone.

### GSoP-Net2

![GSoP_Net2](fig/GSoP_Net2.png)

In GSoP-Net2, except for the last residual stage, each of other residual stage is followed by one GSoP block as in GSoP-Net. 
In the last stage, an [iSQRT-COV](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Towards_Faster_Training_CVPR_2018_paper.pdf)
meta layer is used to replace the global average pooling layer.

## Environment & Machine Configuration

- toolkit: pytorch-0.4.0

- cuda: 9.0

- GPU: GTX 1080Ti
 
- system: Ubuntu 16.04

## Start Up

You can start up the experiments by run train.sh.

```
set -e
arch=resnet50
GSoP_mode=1 #GSoP-Net2:2
batchsize=224
attpos=001\ 0001\ 000001\ 001 
#attpos=001\ 0001\ 000001\ 000 #for GSoP-Net2
attdim=128
spa_h=14
modeldir=ImageNet1k-$arch-GSoP$GSoP_mode-ch$attdim-sp$spa_h-bn-001-0001-000001-001-bs$batchsize
dataset=/path/to/dataset/

if [ ! -e $modeldir/*.pth.tar ]; then

if [ ! -d  "$modeldir" ]; then

mkdir $modeldir

fi
cp train.sh $modeldir

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py -a $arch\
               -p 100\
               -j 8\
               -b $batchsize\
               --GSoP_mode $GSoP_mode\
               --attpos $attpos\
               --attdim $attdim\
               --spa_h $spa_h\
               --modeldir $modeldir\
               $dataset
else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py -a $arch\
               -p 100\
               -j 8\
               -b $batchsize\
               --GSoP_mode $GSoP_mode\
               --attpos $attpos\
               --spa_h $spa_h\
               --attdim $attdim\
               --modeldir $modeldir\
               --resume $checkpointfile\
               $dataset

fi

```

## Experiments

### ImageNet-1K

  |ResNet-50   | top-1 error (%) | top-5 error (%) |
  |:----------:|:-----------:|:-----------:|
  |GSoP-Net1   |     22.02   |    5.88     |
  |GSoP-Net2   |     21.19   |    5.64     | 
  
  
### CIFAR-100

  |ResNet-164  | top-1 error (%) |
  |:----------:|:---------------:|
  |GSoP-Net1   |        20.86    |
  |GSoP-Net2   |        18.58    |
  

## Acknowledgments

* We thank the works as well as the accompanying  code  of [MPN-COV](https://github.com/jiangtaoxie/MPN-COV) and its fast version [iSQRT-COV](https://github.com/jiangtaoxie/fast-MPN-COV). 
* We would like to thank Facebook Inc. for developing pytorch toolbox.

## Contact Information

If you have any suggestion or question, you can leave a message here or contact us directly: gzl@mail.dlut.edu.cn . Thanks for your attention!
