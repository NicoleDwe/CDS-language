# script to calculate collocates for a specific keyword in a given text corpus. 
# intup: directory for corpus, keyword, window size (n of words)
# output: csv file with keyword, collocate, raw frequency, MI

#!/usr/bin/env python

"""
Calculate collocates for specific keyword in text corpus
Parameters:
    path: str <path-to-folder>
Usage:
    collocation_nd.py -p <path-to-folder>
Example:
    $ python collocation_nd.py -p data/100_english_novels/corpus
"""

# import dependencies
import os
import sys 
sys.path.append(os.path.join("..")) # enabling communication with home directory
import pandas as pd 
from pathlib import Path
import csv 
import re
import string
import numpy as np
import argparse


# define tokenizer function: removing non-alphannumeric tokens and splitting words to be elements of a list
def tokenize(input_string):
    # define compiling pattern: split at all characters except for letters (both lowercase and uppercase) and apostrophes
    tokenizer = re.compile(r"[^a-zA-Z']+") 
    # apply tokenizer (compiling pattern) to input string, returning a list of strings
    token_list = tokenizer.split(input_string) 
    # remove empty elements in token_list 
    token_list.remove("")
    # return list of tokens
    return token_list


# define main (collocation) function 
def main():
    # initialise arg parse
    ap = argparse.ArgumentParser()
    # parameters
    ap.add_argument("-p", "--path", required = True, help= "Path to directory of text corpus")
    ap.add_argument("-k", "--keyword", required = True, help= "Key/target word in lowercase letters")
    ap.add_argument("-w", "--windowsize", required = True, help= "Window size in number of words")
    # parse arguments
    args = vars(ap.parse_args())
    
    # get path to corpus directory
    input_path = args["path"]
    # get keyword
    keyword = args["keyword"]
    # get windowsize
    window_size = int(args["windowsize"])
            
    # create empty list for all tokens across the corpus
    token_list_corpus = []
    # create empty list for all collocates across the corpus
    collocates_list = []
    # create empty df to save information
    data = pd.DataFrame(columns=["keyword", "collocate", "raw_frequency", "MI"])
    # define u (how often the keyword occurs) to be 0 (as starting point)
    u = 0
    
    # for each text in the corpus get the tokens and indicies of the keyword in the text
    for filename in Path(input_path).glob("*.txt"):
        with open (filename, "r", encoding = "utf-8") as file:
            
            # read the text
            text = file.read()
            # apply tokenize function to create list, while making all words non-capital
            token_list_text = tokenize(text.lower()) 
            # extend the token list for entire corpus with the token list of the given text
            token_list_corpus.extend(token_list_text)
            
            # get the indicies for the keyword in the given text to be able to find words before/after
            indices = [index for index, x in enumerate(token_list_text) if x == keyword]
            # get the number of indicies (i.e. count of keyword) in the given text 
            n_indicies_text = len(indices)
            
            # add the number of times the keyword occured in the text to the number of times the keyword occurs in the corpus
            u = u + n_indicies_text
            
            # for each index (occurance of keyword) in the indicies list, get the words before/after based on window size 
            for index in indices:
                
                # define beginning and end of text window 
                window_start = max(0, index - window_size)
                window_end = index + window_size
                
                # get the keyword string based on the window, +1, as the last number when indexing lists is not included
                collocate_string = token_list_text[window_start : window_end + 1]
                
                # add each keyword_collocate_string to a list containing all strings of that type across the corpus
                collocates_list.extend(collocate_string)
                # remove the keyword from the list
                collocates_list.remove(keyword)
                
    # get unique collocates from the collocates across the corpus
    unique_collocates = set(collocates_list)
    
    # for each unique collocate, get the frequency and calculate MI
    for collocate in unique_collocates:
        # number of times the given collocate (v) occurs across the corpus
        v = token_list_corpus.count(collocate)
        # raw frequency, i.e. how often does the collocate occur with the keyword (v & u) 
        O11 = collocates_list.count(collocate)
        # how often the the key word occurs without the collocate (u & !v)
        O12 = u - O11
        # how often the collocate occurs without the keyword (v & !u)
        O21 = v - O11
        # R1: sum of O11 and O21
        R1 = O11 + O12
        # C1: sum of O11 and O21
        C1 = O11 + O21
        # total number of tokens in corpus
        N = len(token_list_corpus)
        # expected frequency
        E11 = (R1*C1)/N
        # mutual information score 
        MI = np.log(O11/E11)
        # put information into data frame
        data = data.append({"keyword": keyword, 
                     "collocate": collocate, 
                     "raw_frequency": O11, # how often does the collocate occur with the keyword (v&u)
                     "MI": MI}, ignore_index = True)
    
    # before saving dataframe, sort with highest MI at the top
    data = data.sort_values("MI", ascending = False) 
    
    # create output directory if it does not exist already
    if not os.path.exists("output"):
        os.mkdir("output")
    
    # define output path 
    output_path = os.path.join("output", f"{keyword}_collocation.csv")
    # save data as csv in output directory, as keyword_collocation_df
    data.to_csv(output_path, index = False)
    
    # let the user know that the output is saved in the file name
    print(f"Done! Output is saved in {output_path}")
    
    
# behaviour when script is called from command line
if __name__=="__main__":
    main()