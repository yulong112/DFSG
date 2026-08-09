function [flow,seg] = gcAlfaExpansion(alfa,eLabel,origLabelling,beta,cliques);
%   GCBINARY Binary segmentation in one shot using graph cut techniques for
%   energy minimization. The energy depends on one binary labeling only.
%   [SEG,ERGFIN] = GCBINARY(E0,E1,BETA,CLIQUES)
%   SEG is the segmented image and ERGFIN is the final energy.
%   origLabelling is the start labelling, that will be changed by one
%   alpha-expansion
%   eLabel is a voxel containing one layer fo each label; each layer correspond to the unary
%   terms of the energy to be minimized.
%   BETA is a parameter determining the energy clique terms.
%   CLIQUES is a vector determining the cliques to be considered.

%   For reference see: V. kolmogorov and R. Zabih, "What Energy Functions can be Minimized via
%   Graph Cuts?", European Conference on Computer Vision, May 2002.

% INPUT AND INITIALIZING %%%%%%%%%%%%%%%%%%%%%%%%
[em,en,nrLabels] = size(eLabel);    
nrPixels = em*en;
[cliquesm,cliquesn] = size(cliques); % Size of input cliques
remain = [];
% maximum displacement in a clique
maxdesl = max(max(abs(cliques)));
borderVal = nrLabels +1;

% alfaLabel:= matrix containing ones for pixels that are alfa and zero for those that
% are not alfa, in the original Labelling
alfaLabel = ones(size(origLabelling));
alfaLabel(find(origLabelling ~= alfa )) = 0;
noAlfaLabel = ~alfaLabel;

% these are the same vectors as previous ones
% but with "zeros-border"
alfaLabel = [borderVal*ones(em,maxdesl+1) alfaLabel borderVal*ones(em,maxdesl+1)];
alfaLabel = [borderVal*ones(maxdesl+1,size(alfaLabel,2)); alfaLabel; borderVal*ones(maxdesl+1,size(alfaLabel,2))];
[m,n] = size(alfaLabel);
noAlfaLabel = [borderVal*ones(em,maxdesl+1) noAlfaLabel borderVal*ones(em,maxdesl+1)];
noAlfaLabel = [borderVal*ones(maxdesl+1,size(noAlfaLabel,2)); noAlfaLabel; borderVal*ones(maxdesl+1,size(noAlfaLabel,2))];

base = origLabelling;
base = [borderVal*ones(em,maxdesl+1) base borderVal*ones(em,maxdesl+1)];
base = [borderVal*ones(maxdesl+1,size(base,2)); base; borderVal*ones(maxdesl+1,size(base,2))];

% cliquesNoAlfaNoAlfa:= "pile" of matrices, one for each clique type, that contain
% ones for pixels that have a label noAlfa and clique neighbour noAlfa; the
% other values are 0 or borderVal
% delta has ones for pixels that have the same class as its neighbour, for
% that clique type
for t = 1:cliquesm 
    auxili = circshift(noAlfaLabel,[-cliques(t,1),-cliques(t,2)]);
    tmp = noAlfaLabel;
    tmp(find(noAlfaLabel ~= auxili)) = 0;
    cliquesNoAlfaNoAlfaSource(:,:,t) = tmp;  
    auxili2 = circshift(base,[-cliques(t,1),-cliques(t,2)]);
    delta(:,:,t) = (base == auxili2);
    auxili = circshift(noAlfaLabel,[cliques(t,1),cliques(t,2)]);
    tmp = noAlfaLabel;
    tmp(find(noAlfaLabel ~= auxili)) = 0;
    cliquesNoAlfaNoAlfaSink(:,:,t) = tmp; 
end

% cliquesNoAlfaAlfa:= "pile" of matrices, one for each clique type, that contain
% ones for pixels that have a label noAlfa and clique neighbour Alfa or
% that have a label Alfa and clique neighbour noAlfa
for t = 1:cliquesm
    auxili = circshift(alfaLabel,[-cliques(t,1),-cliques(t,2)]);
    tmp = noAlfaLabel;
    tmp(find(noAlfaLabel ~= auxili)) = 0;
    cliquesNoAlfaAlfa1(:,:,t) = tmp;
end
for t = 1:cliquesm
    auxili = circshift(noAlfaLabel,[-cliques(t,1),-cliques(t,2)]);
    tmp = alfaLabel;
    tmp(find(alfaLabel ~= auxili)) = 0;
    cliquesNoAlfaAlfa2(:,:,t) = tmp;
end
for t = 1:cliquesm
    cliquesNoAlfaAlfa(:,:,t) = (cliquesNoAlfaAlfa1(:,:,t) | cliquesNoAlfaAlfa2(:,:,t));
end

% PROCESSING   %%%%%%%%%%%%%%%%%%%%%%%%

%calculate unary energy terms
B = reshape(eLabel,nrPixels,nrLabels);
L = origLabelling(:);
index = (1:nrPixels)';
E = B(index+(L)*nrPixels);
unaryEnergy = reshape(E,em,en);

% SOURCESINK %%%%%%%%%%%%%%%%%%%%%%%%%%

% source(sink) is a "pile" consisting of one matrix with image dimension
% for each clique type.

% second case in alfa expansion
clear auxili;
for t = 1:cliquesm
    sink(:,:,t) = beta*cliquesNoAlfaAlfa(:,:,t); % Existing pairwise sink links
    source(:,:,t) = 0*sink(:,:,t);
end

% third case in alfa expansion
sourceForRemain = source;
for t = 1:cliquesm
    source(:,:,t+cliquesm) = beta*cliquesNoAlfaNoAlfaSource(:,:,t).*delta(:,:,t);
    sourceForRemain(:,:,t+cliquesm) = beta*cliquesNoAlfaNoAlfaSource(:,:,t);
    sink(:,:,t+cliquesm) = beta*cliquesNoAlfaNoAlfaSink(:,:,t); % Existing pairwise sink links
end

[dummy1,dummy2,s3] = size(sourceForRemain);

% remove "borderVal"
source(1:maxdesl+1,:,:)=[]; source(em+1:em+maxdesl+1,:,:)=[];
source(:,1:maxdesl+1,:)=[]; source(:,en+1:en+maxdesl+1,:)=[];
sourceForRemain(1:maxdesl+1,:,:)=[]; sourceForRemain(em+1:em+maxdesl+1,:,:)=[];
sourceForRemain(:,1:maxdesl+1,:)=[]; sourceForRemain(:,en+1:en+maxdesl+1,:)=[];
sink(1:maxdesl+1,:,:)=[]; sink(em+1:em+maxdesl+1,:,:)=[];
sink(:,1:maxdesl+1,:)=[]; sink(:,en+1:en+maxdesl+1,:)=[];
delta(1:maxdesl+1,:,:)=[]; delta(em+1:em+maxdesl+1,:,:)=[];
delta(:,1:maxdesl+1,:)=[]; delta(:,en+1:en+maxdesl+1,:)=[];

% third case in alfa expansion
for t=1:cliquesm
    % start(endd) is a vector containing the indexes of those pixels that are
    % connected to the source(sink) and are both different from alfa,
    % pixels are numerated columnwise
    start = find(sourceForRemain(:,:,t+cliquesm) == beta);endd = find(sink(:,:,t+cliquesm) == beta);
    tmp1 = delta(:,:,t);
    tmp2 = tmp1(:);
    tmp3 = beta*ones(size(start,1),1) + beta.*tmp2(start);
    auxiliar = [start endd tmp3 zeros(size(endd,1),1)];
    % remain has rows containing index of starting pixel in clique, index
    % of ending pixel in clique, "B+C-A-D",0
    % note that weigth from ending pixel to starting pixel edge is allways
    % zero!
    remain = [remain; auxiliar];
end

remain = sortrows([remain],[1 2]);

% there will be a final layer in source and sink that correspond to the
% unary energy term
% firts case in alfa expansion
source(:,:,s3+1) = [((eLabel(:,:,alfa+1)-unaryEnergy).*((eLabel(:,:,alfa+1)-unaryEnergy)>=0))];
sink(:,:,s3+1) = [((unaryEnergy-eLabel(:,:,alfa+1)).*((unaryEnergy-eLabel(:,:,alfa+1))>0))];
% the final weights of the edges connecting the pixels to the source(sink)
% are a sum of the weights for all clique types and unary term.
sourcefinal = sum(source,3);
sinkfinal = sum(sink,3);
sourcesink = [[1:em*en]' sourcefinal(:) sinkfinal(:)];

[flow,cutside] = mincut(sourcesink,remain);
% The relabeling of each pixel relies on the cutside of that pixel:
%       if the cutside = 0 (source) then we increment the label by 1
%       if the cutside = 1 (sink) then the label remains unchanged

aux = cutside(:,2); 
seg = reshape(aux,em,en);
indexAlfa = find(seg);
indexNoAlfa = find(~seg);
seg(indexAlfa) = alfa;
seg(indexNoAlfa) = origLabelling(indexNoAlfa);

