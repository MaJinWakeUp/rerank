% process query images

% setup matconvnet
cd ../matlab
vl_setupnn;
cd ../mac
addpath ./utils

data_folder = '/home/mj/imagesearch/'; % oxford5k/ and paris6k/ should be in here
dataset_test 				= 'oxford5k';     % dataset to evaluate on 

% config files for Oxford and Paris datasets
gnd_test = load(['./data/gnd_', dataset_test, '.mat']);    

% image files are expected under each dataset's folder
im_folder_test = [data_folder, dataset_test, '/'];

% choose pre-trained CNN model
modelfn = 'imagenet-vgg-verydeep-16.mat';  lid = 31;		% use VGG
net = load(['../' modelfn]);
net.layers = {net.layers{1:lid}}; % remove fully connected layers

% parameters of the method
use_rmac 				= 1;  	% use R-MAC, otherwise use MAC
L                       = 3;

fprintf('Process query images\n');
qimlist = {gnd_test.imlist{gnd_test.qidx}};
qim = arrayfun(@(x) crop_qim([im_folder_test, qimlist{x}, '.jpg'], gnd_test.gnd(x).bbx), 1:numel(gnd_test.qidx), 'un', 0);
if ~use_rmac
	qvecs = cellfun(@(x) vecpostproc(mac(x, net)), qim, 'un', 0);
else
	qvecs = cellfun(@(x) vecpostproc(rmac_regionvec(x, net, L)), qim, 'un', 0);
end

psave = ['./data/query_' dataset_test];
save(psave,'qvecs')