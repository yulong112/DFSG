function totEnergy = Energy( labelling, eLabel, beta, cliques )
%UNTITLED1 Summary of this function goes here
%  Detailed explanation goes here
[em,en,nrLabels] = size(eLabel);    
nrPixels = em*en;
[cliquesm,cliquesn] = size(cliques); % Size of input cliques
% maximum displacement in a clique
maxdesl = max(max(abs(cliques)));

B = reshape(eLabel,nrPixels,nrLabels);
L = labelling(:);
index = (1:nrPixels)';
E = B(index+(L)*nrPixels);
unaryEnergy = reshape(E,em,en);

base = labelling;
base = [zeros(em,maxdesl+1) base zeros(em,maxdesl+1)];
base = [zeros(maxdesl+1,size(base,2)); base; zeros(maxdesl+1,size(base,2))];

binaryEnergy = zeros(em,en);

for t = 1:cliquesm 
    auxili = circshift(base,[-cliques(t,1),-cliques(t,2)]);
    tmpE = zeros(em+4*maxdesl,en+4*maxdesl);
    tmpE(find(base == auxili)) = -beta;
    tmpE(1:maxdesl+1,:,:)=[]; tmpE(em+1:em+maxdesl+1,:,:)=[];
    tmpE(:,1:maxdesl+1,:)=[]; tmpE(:,en+1:en+maxdesl+1,:)=[];
    binaryEnergy = binaryEnergy + tmpE;
end

totEnergy = sum(sum(unaryEnergy + binaryEnergy));