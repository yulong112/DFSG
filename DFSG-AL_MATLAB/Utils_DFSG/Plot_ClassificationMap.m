clc
clear all
close all
matlabpath(pathdef)

%%

mycolormap = [  0    0    0.5;
                1    0.75    0;%1    1  0.75;%
                1    0    0;
                0    1    0;
                1    0    1;
                0  0.5    1;
                0    1    1; %0 1 0
              1    1    0.5;
              0.5 0.25    0];
                %0.5    0  0.5;   ];

load TruthTrainMap_Zaoyuan_05Percent
[Height Width] = size(TruthTrainMap);           
figure
imagesc(TruthTrainMap)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

load TruthTestMap_Zaoyuan_05Percent
[Height Width] = size(TruthTestMap);           
figure
imagesc(TruthTestMap)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

load Map_3DG_8thComponent_Zaoyuan_05Percent
[Height Width] = size(ClassificationMap);
figure
imagesc(ClassificationMap)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

load Map_3DG_AllComponent_Zaoyuan_05Percent
[Height Width] = size(ClassificationMap);
figure
imagesc(ClassificationMap)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

load Map_3DG_8thComponent_PCA_SVM_Zaoyuan_05Percent
[Height Width] = size(Map);
figure
imagesc(Map)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

load Map_3DG_AllComponent_PCA_SVM_Zaoyuan_05Percent
[Height Width] = size(Map);
figure
imagesc(Map)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

%load Map_CSVM_Zaoyuan_05Percent
load Map_CSVM_Zaoyuan_05Percent_V2
[Height Width] = size(Map);
figure
imagesc(Map)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

load Map_3DG_RealPart_Zaoyuan_05Percent
[Height Width] = size(ClassificationMap);
figure
imagesc(ClassificationMap)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

load Map_3DG_RealPart_PCA_SVM_Zaoyuan_05Percent
[Height Width] = size(Map);
figure
imagesc(Map)
colormap(mycolormap)
set( gca,'ylim',[1 Height],'ytick',[ ]);
set( gca,'xlim',[1 Width],'xtick',[ ]);
axis image

clc
