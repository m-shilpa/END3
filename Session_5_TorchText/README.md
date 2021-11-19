- [0.1. Team](#team)
- [1. Dataset 1 : Amazon Reviews Polarity](#dataset1_amazon_reviews_polarity)
  - [1.1 Dataset](#dataset1_amazon_reviews_polarity_details)   
  - [1.2 Text Classification](#text_classification)

## 0.1. Team

- Shilpa
- Shailesh J
- Prathyusha Kanakam
- Raja Rajendran

## 1. Dataset 1 : Amazon Reviews Polarity 

### 1.1 Dataset:
* The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive. 
* Samples of score 3 is ignored. 
* Each class has 1,800,000 training samples and 200,000 testing samples.

### 1.2 Text Classification:
**Goal:** The dataset is used to classify reviews as positive or negative.

**Stats** ([Link to colab](https://github.com/m-shilpa/END3/blob/main/Session_5_TorchText/S5_TorchText_AmazonReviewPolarity.ipynb)):

* No. of epochs: 10   
* Train Accuracy: 91.46%  
* Validation Accuracy: 91.1%   
* Test Accuracy: 91.12%
* Classwise Test Accuracy:
  1. Negative: 90.98%
  2. Positive: 91.25%
* The trend in the accuracy for the train and validation dataset:    
  ![Accuracy Trend](https://github.com/m-shilpa/END3/blob/main/Session_5_TorchText/images/s5_accuracy_trend.png)
