clear
addpath ./utils

dataset_train				= 'paris6k';    % dataset to learn the PCA-whitening on
dataset_test 				= 'oxford5k';     % dataset to evaluate on 

% config files for Oxford and Paris datasets
gnd_test = load(['./data/gnd_', dataset_test, '.mat']);    

% parameters of the method
use_rmac 				= 1;  	% use R-MAC, otherwise use MAC
rerank 					= 1000; 	% number of images to re-rank, no re-ranking if 0
L 							= 3;  	% number of levels in the region pyramid of R-MAC

step_box 				= 3;		% parameter t in the paper
qratio_t 				= 1.1;   % parameter s in the paper
rf_step 					= 3;		% fixed step for refinement
rf_iter 					= 5;		% number of iterations of refinement
% nqe 						= 5;		% number of images to be used for QE

use_gpu	 				= 0;		% use GPU to get CNN responses
dout = 512;     % dimension after pca

% choose pre-trained CNN model
modelfn = 'imagenet-vgg-verydeep-16.mat';  lid = 31;		% use VGG

%% test data processing (mac or r-mac)
% load conv3d featurs
fprintf('data processing...\n');
conv3d_test = load(strcat('./data/',dataset_test,'_vgg_cnn.mat'));
conv3d_train = load(strcat('./data/',dataset_train,'_vgg_cnn.mat'));

% mac or rmac process
if ~use_rmac
    vecs = cellfun(@(x) vecpostproc(mac_act(x)), conv3d_test.images_vgg_cnn, 'un', 0);
    vecs_train = cellfun(@(x) vecpostproc(mac_act(x)), conv3d_train.images_vgg_cnn, 'un', 0);
else
    vecs = cellfun(@(x) vecpostproc(rmac_regionvec_act(x,L)), conv3d_test.images_vgg_cnn, 'un', 0);
	vecs_train = cellfun(@(x) vecpostproc(rmac_regionvec_act(x,L)), conv3d_train.images_vgg_cnn, 'un', 0);
end
fprintf('data processing end\n');

%% learn PCA
fprintf('Learning PCA-whitening\n');
[~, eigvec, eigval, Xm] = yael_pca (single(cell2mat(vecs_train)), dout);

% apply PCA-whitening
fprintf('Applying PCA-whitening\n');
vecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm, eigvec, eigval, dout)), vecs, 'un', 0);
if use_rmac
	% R-MAC: PCA-whitening is perform on region vectors, then they are aggregated
	vecs = cellfun(@(x) vecpostproc(sum(x, 2)), vecs, 'un', 0);
end

% apply PCA-whitening on query vectors
load (['./data/query_' dataset_test]);  %query vectors with pre-process (mac or r-mac) but without pca
qvecs = cellfun(@(x) vecpostproc(apply_whiten (x, Xm, eigvec, eigval, dout)), qvecs, 'un', 0);
if use_rmac
   qvecs = cellfun(@(x) vecpostproc(sum(x, 2)), qvecs, 'un', 0);
end
fprintf('PCA processing end.\n');

% database vectors and query vectors
vecs = cell2mat(vecs);
qvecs = cell2mat(qvecs);
clear vecs_train
clear conv3d_test
clear conv3d_train

%% adding 100k
path_100k = './data/100k/';
addpath(path_100k);
D = dir(strcat(path_100k,'*'));
D = D(3:end);
num_D = size(D,1);
for i=1:num_D
    tic
    conv3d_cur = load(strcat(path_100k,D(i).name));
    if ~use_rmac
        vecs_cur = cellfun(@(x) vecpostproc(mac_act(x)), conv3d_cur.images_vgg_cnn, 'un', 0);
    else
        vecs_cur = cellfun(@(x) vecpostproc(rmac_regionvec_act(x,L)), conv3d_cur.images_vgg_cnn, 'un', 0);
    end
    vecs_cur = cellfun(@(x) vecpostproc(apply_whiten (x, Xm, eigvec, eigval, dout)), vecs_cur, 'un', 0);
    if use_rmac
        % R-MAC: PCA-whitening is perform on region vectors, then they are aggregated
        vecs_cur = cellfun(@(x) vecpostproc(sum(x, 2)), vecs_cur, 'un', 0);
    end
    vecs_cur = cell2mat(vecs_cur);
    vecs = [vecs vecs_cur];
    clear conv3d_cur
    clear vecs_cur
    toc
    fprintf('The %ith component has been added.\n',i);
end

%% retrieval
fprintf('Retrieval\n');
[ranks,sim] = yael_nn(vecs, -qvecs, size(vecs, 2), 16);
map = compute_map (ranks, gnd_test.gnd);
fprintf('mAP, without re-ranking = %.4f\n', map);

%% re-ranking processing
if rerank
    tic
    [ranks_FU, ranks_SP, ranks_QE, ranks_PR] = cast_rerank(ranks, vecs, qvecs , rerank);
    toc
    map1 = compute_map (ranks_FU, gnd_test.gnd);
    map2 = compute_map (ranks_SP, gnd_test.gnd);
    map3 = compute_map (ranks_QE, gnd_test.gnd);
    map4 = compute_map (ranks_PR, gnd_test.gnd);
    fprintf('topk=%i  map, after FU=%.4f  SP=%.4f  QE=%.4f  PR=%.4f\n',...
                rerank, map1, map2, map3, map4);
end






