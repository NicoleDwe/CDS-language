# Assignment 3: Sentiment Analysis

## Task

Dictionary-based sentiment analysis with Python

Download the following CSV file from Kaggle:

https://www.kaggle.com/therohk/million-headlines

This is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19 ; End Date: 2020-12-31).

- Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
- Create and save a plot of sentiment over time with a 1-week rolling average
- Create and save a plot of sentiment over time with a 1-month rolling average
- Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
- Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) - What (if any) are the general trends? 2) What (if any) inferences might you draw from them?

__General instructions:__

- For this assignment, you should upload a standalone .py script which can be executed from the command line.
- Save your script as sentiment.py
- Make sure to include a requirements.txt file and details about where to find the data
- You can either upload the scripts here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

__Purpose:__

This assignment is designed to test that you have a understanding of:
1. how to perform dictionary-based sentiment analysis in Python;
2. how to effectively use pandas and spaCy in a simple NLP workflow;
3. how to present results visually, working with datetime formats to show trends over timekeyword

## Running the script

To run the script `sentiment_nd.py`, it is best to create the virtual environment using the bash script `create_venv.sh` and the requirements specified in `requirements.txt`. This will also load the necessary dependencies from spacy and textblob. The data to run the script is provided in the `data` directory (which is just above the `3-sentiment` directory). The output (csv file, plots) will be saved in an `output` directory. Run the following commands in your terminal to run the script:

1. Clone the repository and save as nicole-cds-language: 

```bash
git clone https://github.com/nicole-dwenger/cds-language.git cds-language-nd
```

2. Move into the correct directory containing files for this assignment:

```bash
cd cds-language-nd/assignments/3-sentiment/
```

3. Create and activate venv called venv_assignment3:

```bash
bash create_venv.sh
source venv_assignment3/bin/activate
```

4. Run the script, while specifying the following parameters:

- -d: to define the path to the input data
- -s (optional): if you wish to save time and only run the script on a subset of the data, you can slice by specfiying an index, which will be used as `data[0:index]`. 

To run the script on the entire data, you could run: 

```bash
python3 sentiment_nd.py -d ../data/abcnews-date-text.csv
```

To run the script only on a subset of the data, you could run:


```bash
python3 sentiment_nd.py -d ../data/abcnews-date-text.csv -s 50000
```


5. When the script is done, you will get a message and the output will be saved in `output/sentimet.csv`


## Written Answers

- Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) - What (if any) are the general trends? 2) What (if any) inferences might you draw from them?

__1-week-average sentiment scores__

![](assignments/3-sentiment/output/1-week_sentiment.png?raw=true)

__1-month-average sentiment scores__

![](assignments/3-sentiment/output/1-month_sentiment.png?raw=true)


In both plots it seems like the average sentiment scores are over time above 0, meaning positive and rarely negative. As these are newspaper headlines I actually expected a lot of them to be negative. However, considering the textblob method of only looking at adjectives, it might be that a lot of the headlines either contained adjectives that are not in the dictionary or no adjectives at all. However, it might also be that my intuition is wrong, and a lot of the headlines are actually positive. Generally, as expected the 1-week-average plot shows more variation than the 1-month average. Lastly, there is a spike at the beginning in both of the plots, which might be due to the running-mean calculations. 
