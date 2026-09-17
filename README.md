DFSG, A novel deep fusion paradigm for multi-scale superpixel graphs, IEEE TIP, 2026
==
[Long Yu](https://faculty.scut.edu.cn/zdhkxygc/yl31_en/main.htm), [Jun Li](https://grzy.cug.edu.cn/lijun1/en/index.htm), [Antonio Plaza](https://www2.umbc.edu/rssipl/people/aplaza/), and [Li Zhuo](https://gp.sysu.edu.cn/en/teacher/189).
***

Code for the paper: [Multi-Scale Meets Active Learning: A Deep Graph Fusion Paradigm for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/11627194), IEEE Transactions on Image Processing, 2026.

<div align=center><img src="/DFSG-AL.png" width="90%" height="90%"></div>
Fig. 1. Architecture of the proposed DFSG-AL method.

### **Abstract**

Deep learning (DL) has attracted considerable attention in the field of hyperspectral image classification (HSIC). However, most DL methods still suffer from two problems: overfitting and oversmoothing, particularly when dealing with scarce labeled samples. A major challenge is that they do not make full use of the relationships among a large number of unlabeled samples and multi-scale information in structural relationships, resulting in the loss of multi-scale information. Moreover, prior information such as labels is not used to explicitly learn and modify the graph structure (including nodes, the sparsity of connections, and edge weights). To address these issues, we propose a novel deep fusion paradigm for multi-scale superpixel graphs (DFSG). Our new DFSG integrates multi-scale graphs (at both the graph-level and the feature-level) to reduce information loss while the re-segmentation based graph correction module adaptively learns new graph structures during the active learning (AL) process. In our proposed iterative updating mechanism, AL and our multi-scale methods help each other, forming a symbiotic unified DFSG-AL framework. Experiments on five real hyperspectral image (HSI) datasets demonstrate that our DFSG-AL can achieve remarkable performance in few-sample HSIC.


<div align=center><img src="/DFSGCN.png" width="90%" height="90%"></div>
Fig. 2. The overall framework of DFSGCN.

---

### **Related extension methods**

Our DFSG paradigm has been successfully applied to the field of multi-modal remote sensing data (HSI + LiDAR) fusion, constructing a new multi-modal model:

[Global–Local Aligned Cross-Modal Network for Hyperspectral Image and LiDAR Data Classification](https://ieeexplore.ieee.org/document/11435417), IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2026.

[Yanhui Chen](https://ieeexplore.ieee.org/author/37089883583), [Long Yu](https://faculty.scut.edu.cn/zdhkxygc/yl31_en/main.htm), [Yilin Duan](https://orcid.org/0009-0001-1940-9581), [Zhaozhao Zeng](https://xplorestaging.ieee.org/author/37088750424), [Jia Chen](https://ieeexplore.ieee.org/author/37087092939), and [Jun Li](https://grzy.cug.edu.cn/lijun1/en/index.htm).

<div align=center><img src="/GLAC-Net.png" width="90%" height="90%"></div>
Fig. 3. Overall framework of GLAC-Net.

<div align=center><img src="/GLAC-Net-performance.png" width="50%" height="50%"></div>
Fig. 4. The performance of GLAC-Net.

### **Future Prospects**
Continuous Learning across Scenes in Remote Sensing


### **Three algorithm implementations of our paradigm:**

(Adapting to different platforms and scenarios)

**(1) DFSG** (matlab, CPU, edge computing): 

	Use `/DFSG-AL_MATLAB/demo_DFSG-AL_smalldata.m` or `/DFSG-AL_MATLAB/demo_DFSG-AL_bigdata.m`

   _Description: transductive inference for semi-supervised learning_

**(2) DFSGCN** (python+matlab, GPU, lightweight network): 

	Use `Step1_Demo_DFSGCN.py` 
	
	(Optional) for houston: use `/Step2_ClassificationResult_Refined_Houston18/demo_DFSGCN_refined.m`

   _Description: deep learning for graph and feature representations_

**(3) DFSG-AL** (matlab, CPU, edge computing):

	Use `/DFSG-AL_MATLAB/demo_DFSG-AL_smalldata.m` or `/DFSG-AL_MATLAB/demo_DFSG-AL_bigdata.m`

   _Description:_
   
   _① transductive inference for semi-supervised learning_
   
   _② "minimal human-computer interaction" based active learning method_

---

### DFSGCN Environment:

	Python ≥3.7
	torch 1.12.1+cu113
	torchvision 0.13.1+cu113
	matplotlib 3.5.3
	numpy 1.21.6
	scikit-image 0.19.3
	scikit-learn 1.0.2
	scipy 1.7.3
	spectral 0.23.1
	thop 0.1.1.post2209072238
	setuptools 65.0.0
	matlabengineforpython ≥r2020a

### Python+Matlab (Hybrid compilation):

	conda activate YOUR_Anaconda_ENV
	cd ...\MATLAB_root\R202x\extern\engines\python
	python setup.py install

---
### Citation

The paper is available now at https://ieeexplore.ieee.org/document/11627194

If this work is helpful to you, please cite our paper as follows:

L. Yu, J. Li, A. Plaza and L. Zhuo, "Multi-scale Meets Active Learning: A Deep Graph Fusion Paradigm for Hyperspectral Image Classification," in IEEE Transactions on Image Processing, vol. 35, pp. 8619-8634, 2026, doi: 10.1109/TIP.2026.3715847.

BibTeX:

	@ARTICLE{11627194,
	  author={Yu, Long and Li, Jun and Plaza, Antonio and Zhuo, Li},
	  journal={IEEE Transactions on Image Processing}, 
	  title={Multi-Scale Meets Active Learning: A Deep Graph Fusion Paradigm for Hyperspectral Image Classification}, 
	  year={2026},
	  volume={35},
	  number={},
	  pages={8619-8634},
	  keywords={Labeling;Aluminum;Modeling;Pixel;Hyperspectral imaging;Image classification;Educational institutions;Matrices;IP networks;Timing;Graph convolution network (GCN);multi-scale;deep fusion paradigm;few samples;hyperspectral image (HSI) classification},
	  doi={10.1109/TIP.2026.3715847}}



