# IMDB_Reviews_BERT
IMDB Reviews Classifier using a pre-trained BERT model. Detailed Project Report is available [here](https://github.com/nagarash/IMDB_Reviews_BERT/blob/main/IMDB%20Movie%20Reviews%20Classifier%20-%20Project%20Report.docx)

### Problem Statement
In this project, given a large number of movie reviews, the goal is to build a machine learning system which can accurately classify the review as either being positive or negative sentiment. 

### Dataset
The IMDB movie reviews dataset consists of 25,000 highly polarizing movie reviews [7]. Each review is tagged as “positive” or “negative”. It is well suited for text classification using supervised learning approaches. The raw data is equally divided in train and test folders with further division into pos (positive) and neg (negative) reviews. Each review within these folders is available in a text file. For this project, in order to reduce the GPU training time for the BERT classifier, we only consider 1000 reviews from the dataset, with equal number of positive and negative labels.

### Solution Statement:
We build a sentiment classifier using a pre-trained BERT model and compare the performance in terms of accuracy and ROC-AUC metrics with a Naïve Bayes classifier for the IMDB movie reviews dataset [7]. We make use of the AWS Sagemaker service for data processing, exploration, model training and evaluation.

### Baseline Model
The performance of the BERT classifier is compared against a naïve Bayes classifier. 

