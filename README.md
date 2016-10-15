# Face Classification: A Specialized Benchmark Study

Code for face classification benchmark of the following paper

[Face Classification: A Specialzed Benchmark Study](https://davidsonic.github.io/index/ccbr2016.pdf)

Jiali Duan, Shengcai Liao, Shuai Zhou, Stan Z. Li 
To appear in CCBR 2016 oral

## Requirements

0. A recent installation of [caffe](http://caffe.berkeleyvision.org) with its python wrapper.
1. All other python requirements are mentioned in `requirements.txt`
2. Matlab R2013a or higher
3. py-faster-rcnn [repository](https://github.com/rbgirshick/py-faster-rcnn)
4. WIDER FACE [benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

## Getting started

### Face Classification Benchmark

The proposed face classification benchmark contains more than 3.5 million extracted patches, with 198616 face patches
in pos-new.tgz and 3359526 nonface patches in directory neg-new2/ .

Nonface patches are compressed into 18 seperate parts, download them and run the following command:

```shell
cd neg-new2/
cat face_nonface_patches_part* > face_nonface_patches.tgz
tar -zxvf face_nonface_patches.tgz
```

### RPN network

Pretrained models for extracting proposals are tarred into Pretrained_model.tgz with zf_faster_rcnn_iter_130000.caffemodel, train.prototxt, test.prototxt and solver.prototxt included. We have made some modifications on the original py-faster-rcnn code to 
train a specialized faster-rcnn network for proposal extraction. 

Replace the original py-faster-rcnn directories with corresponding directories provided in `Proposal_Extraction_Code`. 
Run the `face_proposal.py` to extract proposals.

Make sure to change the following variables in the file according to your developing environment:  

0. `save_dir` is the path to save proposals  
1. `data_root` is the parent folder of JPEGImages of WIDER FACE  
2. `flabel` is the path to the label file. We have provided our training and testing list files in train_test_list.tgz  
3.  Add faster-rcnn-caffe-python path to your `PYTHONPATH` using the following command:  

```shell
export PYTHONPATH='/path/to/you/caffe-fast-rcnn/python':$PYTHONPATH
```

You should get similar results saved under `pos` and `neg` directories of `save_dir`:
<div align="center">
   <img src="https://github.com/davidsonic/face_classification_ccbr2016/blob/master/result.png" height=450 width=300 align=center />
</div>

## Work in progress

I'm in process of cleaning up my original evaluation code. 

__TODO__:
 - [x] LBP training and evaluation code.
 - [x] MBLBP training and evaluation code.
 - [x] LOMO training and evaluation code. 
 - [x] NPD training and evaluation code. 
 - [x] cifar-10 based CNN baseline.
 - [x] Cascade CNN baseline. 





