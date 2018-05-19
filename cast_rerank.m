% cast rerank
function [ranks_FU, ranks_SP, ranks_QE, ranks_PR] = cast_rerank(ranks, x , q , rerank)
alpha = 0.8;
init = 0.99;
fre_thre = 0.003;
ranks_QE = ranks;
ranks_SP = ranks;
ranks_FU = ranks;
ranks_PR = ranks;
num_query = size(q,2);
I = eye(rerank+1);
for i = 1:num_query
    ids_toplist = ranks(1:rerank, i);     
    vecs = [q(:,i) x(:,ids_toplist)];
    A = vecs'* vecs;
    A = diag(diag(A).^-0.5)*A*diag(diag(A).^-0.5);
    A = (A+ abs(A))/2;
    B=tril(A,0);
    y=[];
    for j=2:rerank+1
        y = [y,B(j,1:j-1)];
    end
    xx = linspace(0,1,100);
    yy = hist(y,xx);
    yy = yy/length(y);
    IND_x = find(yy<fre_thre);
    threshold = roundn(xx(IND_x(1)),-2);
    %%  QE
    qnd = 5;
    scores_QE = mean(A(1:2+qnd-1,2:end));
    [~, idx] = sort(scores_QE, 'descend');
    ranks_QE(1:rerank, i) = ranks_QE(idx, i);
    
    % diffusion
    IND = A< threshold;
    A(IND) = 0;    
    sum_A = sum(A)/alpha;
    PW = I - bsxfun(@rdivide,A,sum_A);
    PW = PW\I - I;
    PW = PW/4;
    
    %% PR
    Root_PW = sqrt(PW);
    PageRank = (1-init)*ones(rerank+1,1)/rerank;
    PageRank(1) = init; 
    PageRank = PW*PageRank;
    [~,id] = sort(PageRank(2:end,1),'descend');
    ranks_PR(1:rerank, i) = ranks_PR(id, i);
    
    %% SP
    scores_SP = Root_PW(:,[1;id(1:qnd)+1])' * Root_PW(:,2:end);
    [~, idx] = sort(scores_SP(1,:), 'descend');
    ranks_SP(1:rerank, i) = ranks_SP(idx, i);
    
    %% FU
    scores_SP = mean(scores_SP);
    scores_FU = scores_QE/sum(scores_QE) + scores_SP/sum(scores_SP) + abs(scores_SP/sum(scores_SP) - scores_QE/sum(scores_QE));

    [~, idx] = sort(scores_FU, 'descend');
    ranks_FU(1:rerank, i) = ranks_FU(idx, i);
    

end
end