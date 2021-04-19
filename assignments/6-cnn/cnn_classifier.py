#!/usr/bin/python

"""
CNN classification of preprocessed GoT data, which has sentences chunked into 10 sentences

Input (all optional parameters):
- filenmame: should be .csv file with "Text" and "Label" column, default is GoT_preprocessed_10.csv
- n_epochs: number of epochs, default is 10
- batch_size: batch size, default is 20

Output (saved in "out" directory):
- cnn_summary.txt: model summary of cnn model
- cnn_metrics.txt: accuracy and loss metrics of model
- cnn_history.png: history plot of the cnn model
"""


### LOADING DEPENDENCIES ###

# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy
import pandas as pd
import numpy as np

# to save output
from contextlib import redirect_stdout

# sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, Flatten, GlobalMaxPool1D, Conv1D, Dropout, MaxPool1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# matplotlib
import matplotlib.pyplot as plt

# argument parser
import argparse


### MAIN FUNCTION ###

def main():
    
    # Argument option for data file
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=False, help="Name of data file", default="GoT_preprocessed_10.csv", type=str)
    ap.add_argument("-e", "--epochs", required=False, help="Number of epochs", default=10, type=int)
    ap.add_argument("-b", "--batch_size", required=False, help="Batch size", default=20, type=int)

    # Get the input
    args = vars(ap.parse_args())
    filename = args["filename"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    
    # Create output directory
    out_dir = "out"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Read the data
    data = pd.read_csv(os.path.join("..", "data", filename))
    
    # Get texts and labels out out the data
    texts = data['Text'].values
    labels = data['Label'].values
    
    # Initialise CNN classifier, with texts and labels
    print("\n[INFO] Initialising CNN classifier...\n")
    cnn_classifier = CNN_Classifier(texts=texts, labels=labels, out_dir=out_dir)
    
    # Binarize labels 
    cnn_classifier.binarise_labels()
    # Tokenize texts
    cnn_classifier.tokenize_texts(num_words=5000)
    # Add padding to texts
    cnn_classifier.add_padding()
    # Define model
    model = cnn_classifier.define_model(embedding_dim=150, l2_value=0.001)
   
    # Fit and evaluate model, also saving output in out directory
    print("\n[INFO] Fitting and evaluating model...\n")
    cnn_classifier.fit_evaluate_cnn(model, epochs=epochs, batch_size=batch_size)
          
    print("\n[INFO] All done, output saved in \out!\n")
    

### FUNCTIONS USED IN MAIN ###

class CNN_Classifier:
    
    def __init__(self, texts, labels, out_dir):
        """
        Initialising CNN classifier:
        - Splitting test and train documents and labels
        - Defining output directory
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(texts, labels, test_size=0.25, random_state=42) 
        self.out_dir = out_dir

    def binarise_labels(self):
        """
        Binarising training labels, appended to self as y_train_binary and y_test_binary
        """

        # Initialise binariser
        lb = preprocessing.LabelBinarizer()

        # Apply to train and test data
        self.y_train_binary = lb.fit_transform(self.y_train)
        self.y_test_binary = lb.fit_transform(self.y_test)
        
    def tokenize_texts(self, num_words):
        """
        Tokenizing the documents/texts, and defining the vocabulary size 
        - Input: num_words, refers to the maximum number of most common words to keep 
        - Output: X_train_toks, X_test_toks, vocab_size (all appended to self)
        """
    
        # Intialise tokenizer
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(self.X_train)

        # Tokenized training and test data
        self.X_train_toks = tokenizer.texts_to_sequences(self.X_train)
        self.X_test_toks = tokenizer.texts_to_sequences(self.X_test)

        # Overall vocabulary size
        self.vocab_size = len(tokenizer.word_index) + 1 
        
    def add_padding(self):
        """
        Adding padding to the tokenized texts/documents, to make sure they are the same length 
        This function first gets the maximum length of all documents, and then appends 0s to all others to match this length
        - Output: max_len, X_train_pad, X_test_pad (all appended to self)
        """
    
        # Get the maximum length of the test and train tokens separately
        max_len_train = len(max(self.X_train_toks, key=len))
        max_len_test = len(max(self.X_test_toks, key=len))
        # Get the maximum out of the two maximum lengths 
        self.max_len = max(max_len_train, max_len_test)  

        # Apply padding to training and test tokens
        self.X_train_pad = pad_sequences(self.X_train_toks, padding='post', maxlen=self.max_len)
        self.X_test_pad = pad_sequences(self.X_test_toks, padding='post', maxlen=self.max_len)
        
    def define_model(self, embedding_dim, l2_value):
        """
        Defining cnn model with word embeddings
        - Input: embedding dim: dimension of the dense embedding (output of the layer), l2 value: value for regulariser
        - Output: model, model summary saved in "out"
        """
        # Define l2 regulariser
        l2 = L2(l2_value)

        # Sequential model
        model = Sequential()
        # Embedding layer
        model.add(Embedding(self.vocab_size, embedding_dim, input_length=self.max_len))
        # Convolutional layer with regulariser
        model.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2))
        # Global max pooling
        model.add(GlobalMaxPool1D())
        # Add drop out layer
        model.add(Dropout(0.001))
        # Dense layer with reulariser
        model.add(Dense(32, activation='relu', kernel_regularizer=l2))
        # Drop out layer
        model.add(Dropout(0.01))
        # Final dense layer to categorise into 8 seasons
        model.add(Dense(8, activation='softmax'))

        # Compile model with adam optimiser
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Save model summary
        out_path = os.path.join(self.out_dir, "cnn_summary.txt")
        with open(out_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
                
        return model
            
    def fit_evaluate_cnn(self, model, epochs, batch_size):
        """
        Fitting CNN model, calculating loss and accuracy measures and creating history plot 
        - Input: model, number of epochs, batch size
        - Output: printing loss and accuracy measures, saving history plot in "out"
        """

        # Fit model
        H = model.fit(self.X_train_pad, self.y_train_binary,
                            epochs=epochs,
                            verbose=False,
                            validation_data=(self.X_test_pad, self.y_test_binary),
                            batch_size=batch_size)

        # Evaluate model
        train_loss, train_accuracy = model.evaluate(self.X_train_pad, self.y_train_binary, verbose=False)
        test_loss, test_accuracy = model.evaluate(self.X_test_pad, self.y_test_binary, verbose=False)

        # Printing accuracy metrics
        print("Training Accuracy: {:.4f}".format(train_accuracy))
        print("Testing Accuracy:  {:.4f}".format(test_accuracy))
        
        # Saving accuracy and loss metrics
        with open(os.path.join(self.out_dir, "cnn_metrics.txt"), 'w') as f:
            with redirect_stdout(f):
                print("Training Accuracy: {:.4f}".format(train_accuracy))
                print("Training Loss: {:.4f}".format(train_loss))
                print("Testing Accuracy: {:.4f}".format(test_accuracy))
                print("Training Loss: {:.4f}".format(train_loss))

        # Plot history, saving as image
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "cnn_history.png"))
        
        
        
# If called from the command line, execute main        
if __name__=="__main__":
    main()
    
  