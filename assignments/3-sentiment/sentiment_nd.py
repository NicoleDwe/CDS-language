#!/usr/bin/env python

# script to calculate sentiment scores of news headlines using textblob, and generate plots of rolling averages
# intup: csv file with headlines data, optionally the dataframe can be sliced from 0:index
# output: csv file with headlines data and sentiment scores, 1-week-smoothed plot, 1-month smoothed plot


"""
Calculate sentiment score, produce 1-week-smoothed plot, 1-month smoothed plot
Parameters:
    data_path: str <file-path>
    subset_option: int <index>
Usage:
    sentiment.py -d <file-path> -s <index>
Example:
    $ python sentiment_nd.py -d ../data/abcnews-date-text.csv -s 50000
"""


# import dependencies 
import os
import pandas as pd
import spacy 
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt
import argparse


# defining function for create plots with rolling means over dates
def smoothed_sentiment_plot(date_window, text_date_window, date_sentiment_data):
    # based on date_window, calcualte rolling means od sentiment scores  
    smoothed_data = date_sentiment_data.sort_index().rolling(date_window).mean()
    # create plot with title, xlablels turned 45 degrees, ylabel and smoothed_data
    plt.figure()
    plt.title(f"Sentiment over time with a {text_date_window} rolling average")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Sentiment score")
    plt.plot(smoothed_data) 
    # save figure as png in output
    plt.savefig(os.path.join("output", f"{text_date_window}_sentiment.png"), bbox_inches='tight')
    
    
# main function 
def main():
    
    # initialise and define input options
    ap = argparse.ArgumentParser() # initialise
    ap.add_argument("-d", "--data_path", required = True, help = "Path to input file") # add input option
    ap.add_argument("-s", "--subset_option", required = False, help = "Option of only using a subset of the data")
    args = vars(ap.parse_args()) # parse arguments
    
    # read input data
    data = pd.read_csv(args["data_path"])
    # if subset is specified, slice the data
    if args["subset_option"] is not None:
        slice_index = int(args["subset_option"])
        data = data[:slice_index]
    
    # create output directory if it does not exist already
    if not os.path.exists("output"):
        os.mkdir("output")
        
    # initialise spacy, textblob and nlp pipe
    nlp = spacy.load("en_core_web_sm")
    spacy_text_blob = SpacyTextBlob()
    nlp.add_pipe(spacy_text_blob)
 
    
    #### CALCULATE SENTIMENT SCORES ####
    
    # message
    print("\nCalculating sentiment scores...")
    
    # create list of sentiment scores for each headline
    sentiment_scores = []
    # for each headline in data frame:
    for doc in nlp.pipe(data["headline_text"], batch_size = 500):
        # calculate the sentiment
        sentiment = doc._.sentiment.polarity
        # append to sentiment_scores list
        sentiment_scores.append(sentiment)
        
    # append the sentiment_scores list to the dataframe and save as output csv file in output
    data.insert(len(data.columns), "sentiment", sentiment_scores)
    out_csv_path = os.path.join("output", "sentiment_scores.csv")
    data.to_csv(out_csv_path, index = False)
    

    #### ROLLING MEAN PLOTS ####
    
    # message
    print("Done calculating sentiment scores, now generating plots...")
    
    # create dataframe with date as index and sentiment scores to calculate rolling means based on datetime
    date_sentiment_df = pd.DataFrame(
        {"sentiment": sentiment_scores}, # sentiment score column
        index = pd.to_datetime(data["publish_date"], format='%Y%m%d', errors='ignore')) # date index
    
    # apply smoothed_sentiment_plot function, to create and save plots in output
    smoothed_sentiment_plot("7d", "1-week", date_sentiment_df) # 1-week average
    smoothed_sentiment_plot("30d", "1-month", date_sentiment_df) # 1 month average
    
    # send message
    print("Done! Csv file and plots are in output directory.\n ")
    
    
    
# behaviour when script is called from command line
if __name__=="__main__":
    main()
