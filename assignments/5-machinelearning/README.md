# Assignment 5: Unsupervised Machine Learning

- [Task](#Task)
- [Project and Data](#Project-and-Data)
- [Results](#Results)
- [Running the Script](#Running-the-Script)


## Task

__Applying (un)supervised machine learning to text data__

The assignment this week involves a little bit of a change of pace and a slightly different format. As we have the Easter break next week, you also have a longer period assigned to complete the work.

For this task, you will pick your own dataset to study.

This dataset might be something to do with COVID-19 discourse on Reddit; IMDB reviews; newspaper headlines; whatever it is that catches your eye. However, I strongly recommend using a text dataset from somewhere like Kaggle - https://www.kaggle.com/datasets

When you've chosen the data, do one of the following tasks. One of them is a supervised learning task; the other is unsupervised.

EITHER
- Train a text classifier on your data to predict some label found in the metadata. For example, maybe you want to use this data to see if you can predict sentiment label based on text content.

OR
- Train an LDA model on your data to extract structured information that can provide insight into your data. For example, maybe you are interested in seeing how different authors cluster together or how concepts change over time in this dataset.

You should formulate a short research statement explaining why you have chosen this dataset and what you hope to investigate. This only needs to be a paragraph or two long and should be included as a README file along with the code. E.g.: I chose this dataset because I am interested in... I wanted to see if it was possible to predict X for this corpus.

In this case, your peer reviewer will not just be looking to the quality of your code. Instead, they'll also consider the whole project including choice of data, methods, and output. Think about how you want your output to look. Should there be visualizations? CSVs?

You should also include a couple of paragraphs in the README on the results, so that a reader can make sense of it all. E.g.: I wanted to study if it was possible to predict X. The most successful model I trained had a weighted accuracy of 0.6, implying that it is not possible to predict X from the text content alone. And so on.

__Tips__
- Think carefully about the kind of preprocessing steps your text data may require - and document these decisions!
- Your choice of data will (or should) dictate the task you choose - that is to say, some data are clearly more suited to supervised than unsupervised learning and vice versa. Make sure you use an appropriate method for the - data and for the question you want to answer
- Your peer reviewer needs to see how you came to your results - they don't strictly speaking need lots of fancy -command line arguments set up using argparse(). You should still try to have well-structured code, of course, but you can focus less on having a fully-featured command line tool

__Bonus challenges__
- Do both tasks - either with the same or different datasets

__General instructions__
- You should upload standalone .py script(s) which can be executed from the command line
- You must include a requirements.txt file and a bash script to set up a virtual environment for the project You can use those on worker02 as a template
- You can either upload the scripts here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

__Purpose__

This assignment is designed to test that you have an understanding of:
- how to formulate research projects with computational elements;
- how to perform (un)supervised machine learning on text data;
- how to present results in an accessible manner.

## Project and Data

For this assignment, I decided to look at transcriptions of Ted Talks. The dataset I used is available on [Kaggle](https://www.kaggle.com/miguelcorraljr/ted-ultimate-dataset). My main aim was to investigate whether topics can be derived from the transcriptions of these talks, and if so which topics the talks cover. From here, many further steps could be taken e.g. finding similar talks or looking at a time development of the topics (I have not done this (yet) though). In relation to topic modelling, each transcript, i.e. each talk is considered to be one document. 

In the script I derive 15 topics from Ted Talks which were published in the years 2018-2020. These are visualised in an interactive visualisation (saved as html), and I also generated word clouds for each of the topics (see below). 

## Results

The model had a perplexity value of -7.68 and a coherence measure of 0.42. These can also be found in the `output/results.txt` file. If you'd like to look at the interactive visualisation of topics and frequent words, you can clone the repository and open the `output/LDA_vis.html` file. Below, you can see word clouds of the 15 topics the model has come up with. Here, it becomes clear, that some of these word clouds could be given names, e.g.:

- Topic 0: Family & Gender
- Topic 1: World Problems (health, food, population(
- Topic 2: Biology & Medicine
- Topic 3: History & Law
- Topic 4: Children and Education
- Topic 5: Human-Computer Interactions
- Topic 6: Diseases
- Topic 7: Universe
- Topic 8: Not sure what this would be
- Topic 9: Business
- Topic 10: Uban Development
- Topic 11: Industry & Environment
- Topic 12: Health Care
- Topic 13: Gender
- Topic 14: Psychology, Neuroscience, Cognition 


![word-clouds](https://github.com/nicole-dwenger/cds-language/blob/main/assignments/5-machinelearning/output/wordclouds.jpg)

Overall, I feel like even though I could give most of the topics some name, some of the topics are not really clear. This could maybe be improven by removing some other words which occur across all categories or splitting it up into fewer/more topics. 


## Running the script

To run the script `tedtalk_lda.py`, it is best to create the virtual environment using the bash script `create_venv.sh` and the requirements specified in `requirements.txt`. This will also load the necessary dependencies from spacy. The data to run the script is provided in the `data` directory (which is just above the `5-machinelearning` directory). The output will be saved in an `output` directory. Run the following commands in your terminal to run the script:

1. Clone the repository and save as nicole-cds-language: 

```bash
git clone https://github.com/nicole-dwenger/cds-language.git cds-language-nd
```

2. Move into the correct directory containing files for this assignment:

```bash
cd cds-language-nd/assignments/5-machinelearning/
```

3. Create and activate venv called venv_assignment5:

```bash
bash create_venv.sh
source venv_assignment5/bin/activate
```

4. Run the script:

I did not allow for any arguments in the script, so you can just run it using: 

```bash
python3 tedtalk_lda.py
```

There will be some deprecation warnings (which you can just ignore), and it will take some time to run the script. 


5. When the script is done, you will get a message and the output will be saved in the `output/` directory. 
