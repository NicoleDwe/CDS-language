#!/usr/bin/python

"""
Logistic regression classification of preprocessed GoT data, which has sentences chunked into 10 sentences

Input:
- filename: csv file with "Text" and "Label" columns, default is preprocessed GoT_preprocessed_10.csv data

Output (saved in /out):
- lr_metrics.csv: classification report
- lr_matrix.png: classification matrix
"""

### LOADING DEPENDENCIES ###

# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy
import pandas as pd
import numpy as np

# for matrix
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn import preprocessing

# matplotlib
import matplotlib.pyplot as plt

# argument parser
import argparse


### MAIN FUNCTION ###

def main():
    
    # Argument option for data file
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=False, help="Name of data file", default="GoT_preprocessed_10.csv", type=str)

    # Get the input
    args = vars(ap.parse_args())
    filename = args["filename"]
    
    # Create output directory
    out_dir = "out"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Read the data
    data = pd.read_csv(os.path.join("..", "data", filename))
    
    # Get texts and labels out out the data
    texts = data['Text'].values
    labels = data['Label'].values
    
    print("\n[INFO] Initialising LR classifier...")
    
    # Initialising classifier
    lr_classifier = LR_Classifier(texts, labels, out_dir)
    
    # Vectorising texts
    lr_classifier.vectorize_texts()
          
    print("[INFO] Fitting and evaluating model...")
    
    # Fitting and evaluating classifier
    classifier_metrics, classifier_matrix = lr_classifier.fit_evaluate()
    
    # Printing results
    print(classifier_metrics)
    # Saving results
    lr_classifier.save_results(classifier_metrics, classifier_matrix)
          
    print("\n[INFO] All done, output saved in /out!")

    
    
### FUNCTIONS USED IN MAIN ###

class LR_Classifier:
    
    def __init__(self, texts, labels, out_dir):
        """
        Initialise logistic regression classifier, by splitting texts and labels into train and test data
        And defining output directory
        """
        # Splitting test and train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(texts,labels,test_size=0.25, random_state=42)
        # Define output directory
        self.out_dir = out_dir

    def vectorize_texts(self):
        """
        Vectorise texts using a count vectorizer
        Output: X_train_vec, X_test_vec, feature_names (appended to self)
        """

        # Initialise count vectorizer
        vectorizer = CountVectorizer()

        # Fit to X_train
        self.X_train_vec = vectorizer.fit_transform(self.X_train)
        # Apply to X_test
        self.X_test_vec = vectorizer.transform(self.X_test)

        # Save the feature names of the vector
        self.feature_names = vectorizer.get_feature_names()  
    
    def fit_evaluate(self):
        """
        Fitting and evaluating logistic regression classifier
        Output: classification metrics and matrix 
        """
        
        # Fit logistic regression model
        classifier = LogisticRegression(random_state=42, max_iter=1000).fit(self.X_train_vec, self.y_train)

        # Make predictions
        y_pred = classifier.predict(self.X_test_vec)

        # Classification report
        classifier_metrics = metrics.classification_report(self.y_test, y_pred)

        # Classification matrix
        cm = pd.crosstab(self.y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        p = plt.figure(figsize=(10,10));
        p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        classifier_matrix = p.get_figure()
        
        return classifier_metrics, classifier_matrix

    def save_results(self, classifier_metrics, classifier_matrix):
        """
        Saving metrics and matrix of classifier in out_dir 
        """

        # Save the classification report
        metrics_path = os.path.join(self.out_dir, "lr_metrics.txt")
        with open(metrics_path, "w") as output_file:
            output_file.write(f"Logistic Regression Classification Metrics:\n{classifier_metrics}")

        # Save the matrix
        matrix_path = os.path.join(self.out_dir, "lr_matrix.jpg")
        classifier_matrix.savefig(matrix_path)
        
        
# If called from the command line, execute main        
if __name__=="__main__":
    main()
    
  