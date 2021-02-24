# Assignment 2: String Processing with Python

## Task

Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.

- The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and an output file called out/{filename}.csv
- These parameters can be defined in the script itself
- Find out how often each word collocates with the target across the corpus
- Use this to calculate mutual information between the target word and all collocates across the corpus
- Save result as a single file consisting of three columns: collocate, raw_frequency, MI

BONUS CHALLENGE: Use argparse to take inputs from the command line as parameters

__General instructions:__

For this assignment, you should upload a standalone .py script which can be executed from the command line.

- Save your script as collocation.py
- Make sure to include a requirements.txt file and your data
- You can either upload the scripts here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

__Purpose:__

This assignment is designed to test that you have a understanding of:

1. how to structure, document, and share a Python scripts;
2. how to effectively make use of native Python packages for string processing;
3. how to extract basic linguistic information from large quantities of text, specifically in relation to a specific target keyword

## Running the script

To run the script `collocation_nd.p`, it is best to create the virtual environment using the bash script `create_venv.sh` and the requirements specified in `requirements.txt`. Further, a corpus of 100 english novels can be found in the `data` directory (which is just above the `assignment2-stringprocessing` directory). The output (csv file) will be saved in a `output` directory. Run the following commands in your terminal to run the script:

1. Clone the repository and save as nicole-cds-language: 

```bash
git clone https://github.com/nicole-dwenger/cds-language.git cds-language-nd
```

2. Move into the correct directory containing files for this assignment:

```bash
cd cds-language-nd/assignments/assignment2-stringprocessing/
```

3. Create and activate venv called venv_assignment1:

```bash
bash create_venv.sh
source venv_assignment2/bin/activate
```

4. Run the script, while specifying the following parameters:

- -p: can be used to specify the path,
- -k: can be used to specify the keyword in " "
- -w: can be used to specify the window size

For example you could run:

```bash
python3 collocation_nd.py -p ../data/100_english_novels/corpus/ -k "season" -w 1
```

5. When the script is done, you will get a message and the output will be saved in output/output_word_counts.csv

