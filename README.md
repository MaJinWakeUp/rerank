# What is This
This is a Matlab implementation of our paper  ***Improving Object Retrieval Quality by Integration of Similarity Propagation and Query Expansion***.  Noting that in our paper, we combined our re-rank method with four different image search approaches, namely, BOW, TEMB, RMAC and ASMK. While this repository is only an example  of **RMAC**. If you want to use our re-rank method in other approaches, please add `cast_rerank.m` function after initial rank step as in `test_ramc.m`.  
In this repository, we implement:
* A modified version of RMAC. (For original RMAC, see [gtolias/rmac][7])
* Our re-rank method SP and FU. Results of  QE and Pagerank are also presented.
***
# Prerequisites
## Dependence
* [Matconvnet][1]. This is a MATLAB toolbox implementing CNNs for computer vision applications.
* Optional but recommended: [Library yael][2]. Yael is a library implementing computationally intensive functions used in large scale image retrieval. Functions needed in this experiment are already contained in folder `utils`.
## Dataset
* [Oxford5k][3] consists of 5062 images collected from Flickr by searching for particular Oxford landmarks.
* [Paris6k][4] consists of 6412 images collected from Flickr by searching for particular Paris landmarks. In our experiments, we delete the 20 corrupted images and use the other **6392** images.
* [Flickr100k][5] consists of 100071 images collected from Flickr by searching for popular Flickr tags. 
***
# Run Experiment
1. Download and install [Matconvnet][1].
    1. Download the pre-trained [vgg-16 model][6] to matconvnet root dir.
2. Download and unzip this repository in matconvnet root dir.
3. Download dataset images.
4. Run experiment.
    1. Run `my_feaure_extract.m` to extract features.
    2. Run `my_mac_query_process.m` to pre-process qury images.
    3. Run `test_rmac.m` to get results.
    
### Note 
* During step 4.i and 4.ii, you may need to change variables `data_folder` to correct path where you store dataset images. And you also need to change variables `dataset` to get features of different dataset.  
* For Oxford105k and Paris106k, please run `my_feature_100k.m` and `test_100k.m`.
***
**If you have any question, please contact:**  
*Jin Ma, m799133891@stu.xjtu.edu.cn* or *Shanmin Pang, pangsm@xjtu.edu.cn*

[1]: http://www.vlfeat.org/matconvnet/ "matconvnet home"
[2]: https://gforge.inria.fr/projects/yael/ "yael home"
[3]: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ "Oxford dataset"
[4]: http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/ "Paris dataset"
[5]: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/flickr100k.html "Flickr dataset"
[6]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat "vgg-16 model"
[7]: https://github.com/gtolias/rmac "RMAC"
