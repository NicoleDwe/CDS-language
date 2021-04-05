#!/usr/bin/env python

# script to run topic modelling on ted talk transcripts, generate interactive visualisations and wordclouds

"""
The script completes the following steps:
    - Cleaning and preprocessing data
    - Defining bigrams and trigrams
    - Generating dictionary and corpus
    - Running LDA model
    - Calculating Perplexity and Coherence scrore, printed to the terminal and saved in txt file
    - Saving topics and their most frequent words to txt file
    - Generating word clouds of the most common words for each topic, saved as jpg file
Usage:
    $ python3 tedtalk_lda.py 
"""

### DEPENDENCIES ###

# standards
import os
import sys
sys.path.append(os.path.join("..", ".."))

# data 
import pandas as pd
import numpy as np 

# nlp
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisations
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
# figure size for pyldavis
rcParams['figure.figsize'] = 20,10
# wordcloud
from wordcloud import WordCloud, STOPWORDS

# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
# utils function (adjusted from ross)
import ted_talk_utils

# warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


### MAIN FUNCTION ### 

    
def main():
    
    # trying to ignore deprecation warnings
    warnings.filterwarnings("ignore")
    
    # creating output directory
    if not os.path.exists("output"):
        os.mkdir("output")
    
    print("\n\n[INFO] initialising ted-talk topic modelling...")
    
    # reading and cleaning the data
    print("[INFO] preparing data")
    file = os.path.join("..", "data", "ted_talks_en.csv")
    data = prepare_data(file)
    
    # preprocessing data
    print("[INFO] processing data")
    data_processed = process_data(data)
    
    # generating dictionary and corpus
    print("[INFO] generating dictionary and corpus")
    dictionary, corpus = dict_corp(data_processed)
   
    # running the lda model and calculating perplexity and coherence score
    print("[INFO] running lda model")
    lda_model, perplexity, coherence = run_model(data_processed, dictionary, corpus, n_topics = 15)
    # printing values
    print(f"Perplexity: {perplexity}, Coherence: {coherence}")
    
    # saving scores, and topics with wors
    save_results(lda_model, perplexity, coherence)
    
    # create lda visualisation and save as html
    create_vis(lda_model, corpus, dictionary)
    
    # generating word clouds for the 15 topics
    word_clouds(lda_model)
    
    print("[INFO] all done :)")
    
    
### FUNCTIONS USED IN MAIN ###
    
def prepare_data(file, min_year = 2017):
    """
    Reading data, selecting relevant columns, removing non-speech parts, and filtering based on a minimum year
    """
    # reading data
    df = pd.read_csv(file)
    # keeping only relevant columns
    df = df.loc[:, ("talk_id", "title", "published_date", "topics", "description", "transcript")]
    # removing non-speech parts, which are in "()" from the transcripts
    df["transcript"] = df["transcript"].str.replace(r"\([^()]*\)", "").astype("str")
    # turn dates into dateformat
    df["published_date"] = pd.to_datetime(df['published_date'])
    # filtering to run it only on a limited number of talks
    df = df[df["published_date"].dt.year > min_year]
    return df

def process_data(data):
    """
    Defining bigrams and trigrams and preprocessing documents accordingly to be able to generate a dictionary and corpus
    """
    # creating bigrams and trigrams based on document (i.e. transcripts in the data)
    bigram = gensim.models.Phrases(data["transcript"], min_count=3, threshold=50) # higher threshold fewer phrases
    trigram = gensim.models.Phrases(bigram[data["transcript"]], threshold=50)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # preprocessing data and only getting nounds out of documents
    data_processed = ted_talk_utils.process_words(data["transcript"],nlp,bigram_mod,trigram_mod,allowed_postags=["NOUN"])
    return data_processed

def dict_corp(data_processed):
    """
    Creating dictionary and corpus from processed data
    """
    # creatig dictorionary
    dictionary = corpora.Dictionary(data_processed)
    # removing from the dictionary the words that occur in more tha 90% of the documents (talks)
    dictionary.filter_extremes(no_above=0.6)  
    # creatig the corpus
    corpus = [dictionary.doc2bow(text) for text in data_processed]
    return dictionary, corpus

def run_model(data_processed, dictionary, corpus, n_topics):
    """
    Running lda model, with given number of topics, and calculating perplexity and coherence score
    """
    # defining and running lda model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=n_topics, 
                                       random_state=100,
                                       chunksize=10,
                                       passes=10,
                                       iterations=100,
                                       per_word_topics=True, 
                                       minimum_probability=0.0)
    # perlexity
    perplexity = lda_model.log_perplexity(corpus)
    # coherence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_processed, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    return lda_model, perplexity, coherence

def save_results(lda_model, perplexity, coherence):
    """
    Saving perplexity, coherence and topics to txt file
    """
    topics = lda_model.print_topics()
    with open('output/results.txt','w+') as f:
        f.writelines(f"Perplexity: {perplexity}, Coherence: {coherence}")
        f.writelines('\n\n')
        f.writelines("Topics:\n")
        f.writelines(str(topics))

def create_vis(lda_model, corpus, dictionary):
    """
    Creating visualisation and saving as html file
    """
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, f"output/LDA_vis.html")

def word_clouds(lda_model):
    """
    Generating word clouds for each of the 15 topics
    """
    
    # defining colours for the writing
    cols = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6',
            '#bcf60c','#fabebe','#008080','#e6beff','#9a6324','#aaffc3','#800000']

    # defining word cloud parameters
    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=2500,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    # getting topics from lda model
    topics = lda_model.show_topics(num_topics=15, formatted=False)

    # defining subplots (i.e. one for each topic)
    fig, axes = plt.subplots(3,5, figsize=(15,15), sharex=True, sharey=True)

    # for each topic, generate a wordcloud
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title("Topic: " + str(i))
        plt.gca().axis('off')

    # plotting all wordclouds together and saving them
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig("output/wordclouds.jpg")
    
    
# behaviour when script is called from command line
if __name__=="__main__":
    main()
