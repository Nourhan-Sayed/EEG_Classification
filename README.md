# EEG Classification

# SBE3030, CDSS  Final Project
## EEG signal classification using the thinking out loud dataset 
The Dataset focuses on imagined speech that is a result of imagining the specific word and the EEG signal resulting from that, to detect  1 of 4 spanish words (up/ down/ left/ right). 

Our aim is to classify these EEG signals to help decode them to these words.

Using machine learning techniques, We reached an average multi-class classification of 28.9% 4-fold cross-validation accuracy, and 23.7% using XGBoost, compared to the accuracy of SVM and XGBoost by Gasparini et al, 26.2% and 27.9% respectively.

Using deep learning techniques, 

### Table of Contents
- [Dataset explanation](#dataset)
- [Machine Learning](#machine-learning-process)
  - [Classifiers](#1---random-forest-classifier)
  - [Results](#4--results)

### Dataset 
We used the [Inner Speech Data Set](https://openneuro.org/datasets/ds003626/versions/2.1.2).

Ten healthy right-handed volunteers, all the participants were native Spanish speakers with no prior BCI experience, and they recorded for around two hours.

The folders are divided as follows in the derivatives folder (the data after preprocessing): 
*  10 subject folders that each include 3 session folders and in each session folder we have the EEG data, External electrodes data, Events data, Baseline data, and a Report file mentioned above.

The decoding of the EEG into text should be understood as the classification of a limited number of words (commands) or the presence of phonemes (units of sound that make up words), vowels, syllables, sentences, and states, and then the corresponding brain wave signals are acquired.

The Spanish phrases Arriba, Abajo, Derecha, and Izquierda, which correspond to up, down, right, and left, respectively.

We reached an average multi-class classification of 28.9% 4-fold cross-validation accuracy, and 23.7% using XGBoost, compared to the accuracy of SVM and XGBoost by Gasparini et al, 26.2% and 27.9% respectively.

### Machine Learning Process
- Classifier
- Dataset Exploration
- Results


####  1 - Random Forest Classifier (Decision Tree Based)




####  2 -   Support vector machine (SVM)

#### 3- XG-Boost classifier







#### 4- Results
The SVC Model Showed an accuracy of 28.9% compared to Gasparini et al accuracy of 26.2%
The Random forest showed an accuracy of 24.7%
the XGBoost showed an accuracy of 27.6% compared to Gasparini et al accuracy of 27.9% 






