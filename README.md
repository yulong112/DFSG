# DFSG

Multi-scale Meets Active Learning: A novel deep fusion paradigm for multi-scale superpixel graphs (DFSG) has been accepted by IEEE Transactions on Image Processing.

---

### **Abstract**

Deep learning (DL) has attracted considerable attention in the field of hyperspectral image classification (HSIC). However, most DL methods still suffer from two problems: overfitting and oversmoothing, particularly when dealing with scarce labeled samples. A major challenge is that they do not make full use of the relationships among a large number of unlabeled samples and multi-scale information in structural relationships, resulting in the loss of multi-scale information. Moreover, prior information such as labels is not used to explicitly learn and modify the graph structure (including nodes, the sparsity of connections, and edge weights). To address these issues, we propose a novel deep fusion paradigm for multi-scale superpixel graphs (DFSG). Our new DFSG integrates multi-scale graphs (at both the graph-level and the feature-level) to reduce information loss while the re-segmentation based graph correction module adaptively learns new graph structures during the active learning (AL) process. In our proposed iterative updating mechanism, AL and our multi-scale methods help each other, forming a symbiotic unified DFSG-AL framework. Experiments on five real hyperspectral image (HSI) datasets demonstrate that our DFSG-AL can achieve remarkable performance in few-sample HSIC.

---

### **Three algorithm implementations of this paradigm:**

(Adapting to different platforms and scenarios)

**(1) DFSG** (matlab, CPU, edge computing): 

	Using `/DFSG-AL_MATLAB/demo_DFSG-AL_smalldata.m` or `/DFSG-AL_MATLAB/demo_DFSG-AL_bigdata.m`

   _Description: transductive inference for semi-supervised learning_

**(2) DFSGCN** (python+matlab, GPU, lightweight network): 

	Using `Step1_Demo_DFSGCN.py` and (for houston) `/Step2_ClassificationResult_Refined_Houston18/demo_DFSGCN_refined.m`

   _Description: deep learning for graph and feature representations_

**(3) DFSG-AL** (matlab, CPU, edge computing):

	Using `/DFSG-AL_MATLAB/demo_DFSG-AL_smalldata.m` or `/DFSG-AL_MATLAB/demo_DFSG-AL_bigdata.m`

   _Description: transductive inference for semi-supervised learning & minimal human-computer interaction based active learning_

---

### DFSGCN Environment:

	Python 3.7
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

Early Access is available now at https://ieeexplore.ieee.org/document/11627194

If this work is helpful to you, please citing our work as follows:

L. Yu, J. Li, A. Plaza and L. Zhuo, "Multi-scale Meets Active Learning: A Deep Graph Fusion Paradigm for Hyperspectral Image Classification," in IEEE Transactions on Image Processing, vol. 35, pp. 8619-8634, 2026, doi: 10.1109/TIP.2026.3715847.


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



