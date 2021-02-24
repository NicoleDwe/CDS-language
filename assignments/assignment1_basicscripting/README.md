# Assignment 1: Basic Scripting with Python

## Task

Using the corpus called 100-english-novels found on the cds-language GitHub repo, write a Python programme which does the following:
- Calculate the total word count for each novel
- Calculate the total number of unique words for each novel
- Save result as a single file consisting of three columns: filename, total_words, unique_words

__General instructions__
- For this exercise, you can upload either a standalone script OR a Jupyter Notebook
- Save your script as word_counts.py OR word_counts.ipynb
- You can either upload the script/notebook here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow the structure of your script.
- Similarly, remember to use descriptive variable names! A name like word_count is more readable than wcnt.

__Purpose__
This assignment is designed to test that you have a understanding of:
- how to structure, document, and share a Python script;
- how to effectively make use of native Python data structures, functions, and flow control;
- how to load, save, and process text files.

## Running the script

To run the script `word_counts_nd.p`, it is best to create the virtual environment using the bash script `create_venv.sh` and the requirements specified in `requirements.txt`. Further, a corpus of 100 english novels can be found in the `data` directory. The output (csv file) will be saved in a `output` directory. Run the following commands in your terminal to run the script:

1. Clone the repository and save as nicole-cds-language: 

```bash
git clone https://github.com/nicole-dwenger/cds-language.git cds-language-nd
```

2. Move into the correct directory containing files for this assignment:

```bash
cd cds-language-nd/assignments/assignment1-basicscripting/
```

3. Create and activate venv called venv_assignment1:

```bash
bash create_venv.sh
source venv_assignment1/bin/activate
```

4. Run the script, while specifying the path to the corpus of texts:`

```bash
python3 word_counts_nd.py -p data/100_english_novels/corpus/
```

5. When the script is done, you will get a message and the output will be saved in output/output_word_counts.csv

