# Multi-modal-classification
A deep multi-modal multi-level fusion learning framework categorizes large-scale multi-modal (text and image) product data into type codes.
This is a PyTorch implementation of the Multimodal Product Classification task of the 2020 SIGIR Challenge on eCommerce (ECOM20), using Rakuten France's catalog. 

This repository contains the code and resources developed for my Master’s thesis, which focused on a multimodal approach to e-commerce product classification using a hierarchical fusion of image and text data.
Outcome: The results of this research were presented at the IEEE 2nd Conference on Information Technology and Data Science (2022), in the paper titled:
[Multimodal E-Commerce Product Classification Using Hierarchical Fusion](https://ieeexplore.ieee.org/document/9914136)
In this work, we demonstrate how combining visual and textual features improves the classification accuracy of e-commerce products, showcasing the potential of hierarchical fusion models for handling diverse input modalities.


## Dataset Overview 

The dataset comprises 55K unique commercial products. Each product has: 
- French title
- French description 
- Associated image

![alt text](https://github.com/SaraFattouh/Multi-modal-classification/blob/master/Example_products.png)

## Uni-modal Models
- CamemBERT, FlauBERT: for text classification
- SE-ResNeXt-50: for image classification

## Multi-modal Models

This repository includes two implementations: 
- The Baseline Approach
- The Proposed Hierachical Approach

In the baseline approch, a concatinated field of the product title and description was fed to CamemBERT, FlauBERT. The resulted embeddings X_t1, X_t2 were fused with the embeddings comming from the image classfication model X_im

![alt text](https://github.com/SaraFattouh/Multi-modal-classification/blob/master/Baseline.png)


In the proposed hierachical approach, product's title and description were fed individually to CamemBERT, FlauBERT. The resulted textual embeddings were fused together to get X_t1, X_t2, this is done in the first fusion layer. In the second fusion layer, textual and visual features were fused together. Several fusion approaches used and compared. That included: Concatenation, Addition, Average 

![alt text](https://github.com/SaraFattouh/Multi-modal-classification/blob/master/Multi-modal.png)

# Technical Debts

Several parts of this codebase are borrowed from [Deep-Learning-Framework-for-Multi-modal-Product-Classification](https://github.com/depshad/Deep-Learning-Framework-for-Multi-modal-Product-Classification) 


To read more about the baseline approach see: [Paper Link](https://sigir-ecom.github.io/ecom20DCPapers/SIGIR_eCom20_DC_paper_8.pdf) 

More info the data challenge: [Data challenge link](https://sigir-ecom.github.io/ecom2020/index.html) 

# Environment 
This code was tested in the following environment and with the following software versions:
- Python 3.7.11
- PyTorch 1.9.0+cu102
- cv2
- Transformers 4.8.2







