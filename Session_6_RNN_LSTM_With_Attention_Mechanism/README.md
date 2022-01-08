## Dataset 1: Quora Dataset
### **Download Link :** 
[http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv](http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv)

### Description:

1. This is a dataset from Quora, a question-and-answer website.
2. The dataset consists of the following columns:
    1. row ID
    2. question 1 ID
    3. question 2 ID
    4. question 1
    5. question 2  
    6. class label which is 0 for non-duplicate pairs and 1 for duplicate pairs.
3. The question IDs (qids) uniquely identify each question.
4. A sample of the dataset looks like below:
![image](https://user-images.githubusercontent.com/36926868/148654289-50c0aaa7-bcc0-43a7-9e5f-1c8c029da199.png)


### Colab Notebook:
  [https://github.com/m-shilpa/END3/blob/main/Session_6_RNN_LSTM_With_Attention_Mechanism/END3_Session_6_Quora_Dataset.ipynb](https://github.com/m-shilpa/END3/blob/main/Session_6_RNN_LSTM_With_Attention_Mechanism/END3_Session_6_Quora_Dataset.ipynb)

### Logs of the training:
![image](https://user-images.githubusercontent.com/36926868/148654146-916b5b50-f37e-4414-9119-ad13d4f3b55f.png)

From the above image, the final loss achived is 3.3926
