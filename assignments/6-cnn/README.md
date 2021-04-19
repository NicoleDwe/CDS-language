# Assignment 5: Unsupervised Machine Learning

- [Task](#Task)
- [Scripts and Data](#Scripts-and-Data)
- [Running the Scripts](#Running-the-Scripts)
- [Results](#Results)


## Task

__Text classification using Deep Learning__

Winter is... hopefully over.

In class this week, we've seen how deep learning models like CNNs can be used for text classification purposes. For your assignment this week, I want you to see how successfully you can use these kind of models to classify a specific kind of cultural data - scripts from the TV series Game of Thrones.

You can find the data here: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons

In particular, I want you to see how accurately you can model the relationship between each season and the lines spoken. That is to say - can you predict which season a line comes from? Or to phrase that another way, is text content a good predictor of series?

Start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. Then you should try to come up with a solution which uses a DL model, such as the CNNs we went over in class.


__Tips__
- Think carefully about the kind of preprocessing steps your text data may require and document these decisions.
- Think just as carefully about the kind of parameters you use in you model. They all make a difference!
- Some sentences are very short; some are longer. Think about how you should handle this.

__General instructions__
- You should upload standalone .py script(s) which can be executed from the command line - one for the LogisticRegression model and one for the DL model.
- You must include a requirements.txt file and a bash script to set up a virtual environment for the project You can use those on worker02 as a template
- You can either upload the scripts here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

__Purpose__

This assignment is designed to test that you have an understanding of:
- how to build CNN models for text classification;
- how to use pre-trained word embeddings for downstream tasks;
- how to work with real-world, complex cultural text data.


## Scripts and Data

__Data:__

This dataset from kaggle with sentences from Games of Thrones is the basis of this assignment. This raw data is stored in `../data/Game_of_Thrones_Script.csv`.

__Preprocessing:__

The raw data was preprocessed using the `GoT_preprocess.py` script. The preprocessd data is stored as `../data/GoT_preprocessed_10.csv`. This preprocessing included:
- Removing rows in which the "Sentence" was "SEASON" or "EPISODE"
- Tokenizing all texts in the "Sentence" column into single sentences, and chunking these single sentences into chunks of 10 sentences (hence, the 10 in the filename)

__Classification Scripts:__

Besides the preprocessing script, this repository contains a `lr_classification.py` script, which runs a logistic regression classifier on the preprocessed GoT data. Further, it contains a `cnn_classification.py` script, which runs a convolutional neural network with an itial embedding layer to classify the preprocessed GoT data by season. 

__Remaining Scripts:__

Besides the preprocessing and classification scripts, this repository contains a `create_venv.sh` file and a `requirements.txt` file, which enables reproducing the environment to run the scripts. 


## Running the Scripts

To run any of the scripts, it is best to create the virtual environment using the bash script `create_venv.sh` and the requirements specified in `requirements.txt`:

1. Clone the repository and save as nicole-cds-language: 

```bash
git clone https://github.com/nicole-dwenger/cds-language.git cds-language-nd
```

2. Move into the correct directory containing files for this assignment:

```bash
cd cds-language-nd/assignments/6-cnn/
```

3. Create and activate venv called venv_assignment5:

```bash
bash create_venv.sh
source venv_assignment6/bin/activate
```

__Preprocessing:__

The preprocessed data is already stored in the `../data` directory and can be used to run the classification scripts. However, if you wish to run the script on your own, feel free. The raw data, which is also stored in the `../data/` directory is loaded by default. The only parameter, that you can optionally define is the chunk size, which by default is 10. 

```bash
# with default parameters
python3 GoT_preprocess.py

# with different chunk size 
python3 GoT_preprocess.py -c 20
```

__Logistic Regression Classifier:__

The logistic regression classifier script does not require any input to run the script. However, if you wish to run the script on a different file you can change it using 
- `--filename`: filename, default is the preprocessed GoT data (any other csv file stored in the `../data/` directory, which has the columns "Text" and "Label" would also work)

The outputs of the script are saved in `out/` and are:
- `lr_metrics.txt`: classification metrics
- `lr_matrix.txt`: classification matrix

To run the script, you can run:
```bash
python3 lr_classifier.py
```

__CNN Classifier:__

The CNN classifier script does not require any input to run the script. However, there are several parameters, which you can optionally provide, if you wish: 
- `--filename`: filename, default is the preprocessed GoT data
- `--epochs`: number of epochs to run the model, default is 10 
- `--batch_size`: batch size to run the model, default is 20

The outputs of the script are saved in `out/` and are:
- `cnn_summary.txt`: model summary of cnn model
- `cnn_metrics.txt`: accuracy and loss metrics of model
- `cnn_history.png`: history plot of the cnn model

To run the script, you can run:
```bash
python3 cnn_classifier.py
```

There will be some tensorflow warnings, but the script still runs, so you can just ignore them. 


## Results

Example results of the script are provided in the `out/` directory. These indicate, that when running the scripts with the defined default parameters on the preprocessed data (chunks of 10 sentences), the LR model gets to an weighted accuracy of 0.38, while the CNN script achieves an accuracy on the training data of 1, and of 0.31 on the testing data. This indicates that the model is overfitting. I tried to avoid overfitting by using drop-out layers and regularisation, but this did not seem to help much. So, it'll probably need to some testing and adjusting of parameters to improve performance (or maybe the task is just too complex). 


