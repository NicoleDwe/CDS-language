#!/usr/bin/env python

"""
Script to preprocess the GoT data, saved as GoT_preprocessed_{chunk_size}.csv, with columns "Label" and "Text"

The preprocessing steps are:
- Filtering out rows in which the "Sentence" is "SEASON" or "EPISODE
- Tokenizing the "Sentence" column into separate sentences, saved in "Tokenized"
- Exploding the data frame, so that there is always one sentence in each row
- Removing those rows, in which the "Tokenized" sentence only contains one token, e.g. "!"
- Chunking the sentences into chunks of 10 sentences and saving them in a new dataframe with the columns:
    "Label" (Season) and "Text" (Chunk of 10 sentences) 
"""

# Libraries
import os
import sys
sys.path.append(os.path.join(".."))

import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')

import argparse



def main():
    
    # Argument option for chunk size
    ap = argparse.ArgumentParser()
    # input option for filepath to the input file, with default of realnews data
    ap.add_argument("-c", "--chunk_size", required=False, help="Size of chunks of sentences", default=10, type=int)

    # Get the input
    args = vars(ap.parse_args())
    chunk_size = args["chunk_size"]
     
    # Reading GoT data
    file = os.path.join("..", "data", "Game_of_Thrones_Script.csv")
    data = pd.read_csv(file)
    
    # Removing rows in which "Sentence" is "SEASON" or "EPISODE
    data = data.loc[(data["Sentence"] != "EPISODE") & (data["Sentence"] != "SEASON")]
    
    # In each row, tokenize the "Sentence" column into separate sentences
    data["Tokenized"] = data["Sentence"].apply(nltk.sent_tokenize)
    
    # Exploding data, so that there is always one sentence in each row
    data = data.explode("Tokenized").reset_index()
    
    # Removing those rows in which the sentence is shorter than 2 characters, e.g. "!"
    data = data[data["Tokenized"].str.len().gt(1)]
    
    # Creating a new data frame to save the chunked data
    df = pd.DataFrame(columns=['Label', 'Text'])
    
    # For each season... 
    for season in data["Season"].unique():
        # Put all of the sentences into a list
        season_sentences = data[data["Season"] == season]["Tokenized"].tolist()
        # Chunking them up into chunks of 10 sentences
        chunks = [season_sentences[x:x+chunk_size] for x in range(0, len(season_sentences), chunk_size)]
        # Join the sentences of each chunk and add them to the data frame
        for chunk in chunks:
            joined_chunk = " ".join(chunk)
            df = df.append({"Label": season,"Text": joined_chunk}, ignore_index=True)
            
    out_file = os.path.join("..", "data", f"GoT_preprocessed_{chunk_size}.csv")
    df.to_csv(out_file)
       
                            
   
if __name__=="__main__":
    main()