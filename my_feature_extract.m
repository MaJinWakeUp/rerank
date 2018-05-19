% feature_extract

% setup matconvnet
cd ../matlab
vl_setupnn;
cd ../mac

data_folder = '/home/mj/imagesearch/'; % oxford5k/ and paris6k/ should be in here

dataset			= 'oxford5k';    % dataset to learn the PCA-whitening on
% dataset to evaluate on 

% config files for dataset
load(['./data/gnd_', dataset, '.mat']);    

% image files are expected under each dataset's folder
im_folder = [data_folder, dataset, '/'];

% choose pre-trained CNN model
% modelfn = 'imagenet-caffe-alex.mat';   lid = 15;				% use AlexNet
modelfn = 'imagenet-vgg-verydeep-16.mat';  lid = 31;		% use VGG
net = load(['../' modelfn]);
net.layers = {net.layers{1:lid}}; % remove fully connected layers

num_images = size(imlist,1);
images_vgg_cnn = cell(1,num_images);
for imnum = 1:num_images
    tic
    im = imread(strcat(im_folder,imlist{imnum},'.jpg')) ;
%     if size(im,3) == 1
% 		im = repmat(im, [1 1 3]);
% 	end
    for i=1:3
        im_(:,:,i) = single(im(:,:,i)) - single(mean(net.meta.normalization.averageImage(i)));
    end
	rnet = vl_simplenn(net, im_);  
    images_vgg_cnn{imnum} = max(rnet(end).x, 0);
    clear im_
    toc
end

Path_save = strcat('./data/',dataset,'_vgg_cnn');
save(Path_save,'images_vgg_cnn','-v7.3')
