function [x, X] = MAC(I, net)

	if size(I,3) == 1
		I = repmat(I, [1 1 3]);
    end
    for i=1:3
        I_(:,:,i) = single(I(:,:,i)) - single(mean(net.meta.normalization.averageImage(i)));
    end
	if ~isa(net.layers{1}.weights{1}, 'gpuArray')
		rnet = vl_simplenn(net, I_);  
		X = max(rnet(end).x, 0);
	else
		rnet = vl_simplenn(net, gpuArray(I_));  
		X = gather(max(rnet(end).x, 0));
	end

	x = mac_act(X);