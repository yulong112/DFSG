
function [superpix_img,HIM] = SLIC_fun(HIM, numSuperpixels,compactness)
[nl ns nb] = size(HIM);

%把数据按行序转成一维数组形式
input_img = zeros(1, nl * ns * nb);

startpos = 1;
for i = 1 : nl % 行
    for j = 1 : ns % 列
        input_img(startpos : startpos + nb - 1) = HIM(i, j, :); % 波段
        startpos = startpos + nb;
    end
end


%% 进行分割，输出分割结果
if ~exist('numSuperpixels','var')
    numSuperpixels = 2000;  %希望分割的数量
end
if ~exist('compactness','var')
    compactness = 0.1; %compactness2 = 1-compactness, 聚类判别依据为 compactness*dxy+compactness2*dspectral
end
dist_type = 2; % 1:欧氏距离；2：SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: 全部像元完成聚类， 2：存在未聚类的像元
% labels:以行序优先的方式存放每个像元的分区
% numlabels: 实际分区数目
% seedx: 种子的列号索引， seedy：种子的行号索引
[labels, numlabels, seedx, seedy] = hybridseg(input_img, nl, ns, nb, numSuperpixels, compactness, dist_type, seg_all);%numlabels is the same as number of superpixels
% note: "labels" is not in matlab format
clear input_img;

l = reshape(labels,ns,nl);
superpix_img = reshape(l',nl*ns,1);