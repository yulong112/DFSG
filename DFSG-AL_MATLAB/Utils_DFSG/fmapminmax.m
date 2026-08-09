function MHyperCube=fmapminmax(HyperCube)

% input_regular=reshape(HyperCube,size(HyperCube,1)*size(HyperCube,2),size(HyperCube,3));
% Min_Mat=repmat(min(input_regular),[size(HyperCube,1)*size(HyperCube,2),1]);
% Max_Mat=repmat(max(input_regular),[size(HyperCube,1)*size(HyperCube,2),1]);
% output_regular=(input_regular-Min_Mat)./(Max_Mat-Min_Mat)*255;
% MHyperCube=reshape(output_regular,size(HyperCube,1),size(HyperCube,2),size(HyperCube,3));

input_regular=reshape(HyperCube,size(HyperCube,1)*size(HyperCube,2),size(HyperCube,3));
output_regular=mapminmax(input_regular',0,255)';
MHyperCube=reshape(output_regular,size(HyperCube,1),size(HyperCube,2),size(HyperCube,3));

