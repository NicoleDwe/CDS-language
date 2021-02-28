# script to count words and unique words for novels in corpus 
# intup: path to directory of corpus
# output: csv file with filename, word count, unique wordcount

#!/usr/bin/env python

"""
Calculate collocates for specific keyword in text corpus
Parameters:
    path: str <path-to-folder>
Usage:
    word_counts_nd.py -p <path-to-folder>
Example:
    $ python word_counts_nd.py -p data/100_english_novels/corpus
"""


# dependencies
import os
from pathlib import Path
import pandas as pd
import argparse

# define main function 
def main():
    
    # initialise arg parse
    ap = argparse.ArgumentParser()
    # parameters
    ap.add_argument("-p", "--path", required = True, help= "Path to directory of text corpus")
    # parse arguments
    args = vars(ap.parse_args())
    
    # create empty data frame to save data
    df_word_counts = pd.DataFrame(columns=["filename", "total_words", "unique_words"])

    # loop to get number of words, unqiue words and save info in df_word_counts
    for filepath in Path(args["path"]).glob("*.txt"):
        
        # open each txt file in the input path
        with open(filepath, "r", encoding = "utf-8") as file:
        
        # read file and get info
            loaded_text = file.read() # read file 
            filename = filepath.name # extract filename
            words = loaded_text.split() # split into words
            unique_words = set(words) # keep the unique words
       
            # append row with info to df
            df_word_counts = df_word_counts.append({"filename": filename, 
                                                "total_words": len(words), 
                                                "unique_words": len(unique_words)}, ignore_index = True)
        
       
    # create output directory if it does not exist already
    if not os.path.exists("output"):
        os.mkdir("output")
        
    # save as csv file in output directory
    output_path = os.path.join("output", "out_word_counts.csv")
    df_word_counts.to_csv(output_path, index = False)
    
    # print message that it's completed
    print(f"Done! Output file is saved in output/out_word_counts.csv")

        
# behaviour when script is called from command line
if __name__=="__main__":
    main()