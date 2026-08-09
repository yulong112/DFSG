function   [OutputImage]=GaussianSmooth(InputImage,MaskSize,Sigma)
               
            GaussianMask=fspecial('gaussian',[MaskSize,MaskSize],Sigma); 
            OutputImage=imfilter(InputImage,GaussianMask,'conv','symmetric','same');
            
end

% test code:
% A = [2 1 2 2; 3 4 2 2; 2 2 2 2; 2 2 2 2]
% h = [5 6 1; 6 2 4; 1 4 3]
% imfilter(A, h,'conv','symmetric','same')

% A =
% 
%      2     1     2     2
%      3     4     2     2
%      2     2     2     2
%      2     2     2     2
     
% h =
% 
%      5     6     1
%      6     2     4
%      1     4     3
     
% ans =
% 
%     74    71    59    64
%     81    68    69    64
%     73    75    70    64
%     64    64    64    64