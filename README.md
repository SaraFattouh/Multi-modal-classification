# Multi-modal-classification
Deep multi-modal multi-level fusion learning framework used to categorize large-scale multi-modal (text and image) product data into type codesfeatures.
This is a PyTorch implementation of Multimodal Product Classification task of the 2020 SIGIR Challenge On eCommerce (ECOM20), using the catalog of Rakuten France. 

## Dataset Overview 

The dataset comprises 55K unique commercial products. Each product has: 
- French title
- French description 
- Associated image

![alt text](https://github.com/SaraFattouh/Multi-modal-classification/blob/master/Example_products.png)

## Uni-modal Models
CamemBERT, FlauBERT: for text classification
SE-ResNeXt-50: for image classification

## Multi-modal Models

This repository includes two implementations: 
- The Baseline Approach
- The Proposed Hierachical Approach

In the Baseline Approch, a concatinated field of the product title and description were fed to CamemBERT, FlauBERT. The resulted embeddings X_t1, X_t2 were fused with the embeddings comming from the image classfication model X_im

![alt text](https://github.com/SaraFattouh/Multi-modal-classification/blob/master/Baseline.png)


In the proposed Hierachical approach, product's title and description were fed individually to CamemBERT, FlauBERT. The resulted textual embeddings were fused together to get X_t1, X_t2, this is done in the first fusion layer. In the second fusion layer, textual and visual features were fused together. 

![alt text](https://github.com/SaraFattouh/Multi-modal-classification/blob/master/Multi-modal.png)



