# What is This
This is a Matlab implementation of our paper  ***Improving Object Retrieval Quality by Integration of Similarity Propagation and Query Expansion***.  Noting that in our paper, we combined our re-rank method with four different image search approaches, namely, BOW, TEMB, RMAC and ASMK. While this repository is an example  of **RMAC**. If you want to use our re-rank method in other approaches, please add `cast_rerank.m` function after initial rank step as in `test_ramc.m`.
***
# Setup
## Dependence
* [Matconvnet][1]. This is a MATLAB toolbox implementing CNNs for computer vision applications.
* Optional but recommended: [Library yael][2]. Yael is a library implementing computationally intensive functions used in large scale image retrieval. Functions needed in this experiment are already contained in folder `utils`.
## Dataset
* [Oxford5k][3] consists of 5062 images collected from Flickr by searching for particular Oxford landmarks.
* [Paris6k][4] consists of 6412 images collected from Flickr by searching for particular Paris landmarks. In our experiments, we delete the 20 corrupted images and use the other **5392** images.


[1]: http://www.vlfeat.org/matconvnet/ "matconvnet home"
[2]: https://gforge.inria.fr/projects/yael/ "yael home"
[3]: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ "Oxford dataset"
[4]: http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/ "Paris dataset"
