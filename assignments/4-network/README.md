# Assignment 4: Network Analysis


- [Task](#task)
- [Running the Script](#Running-the-Script)


## Task

__Creating reusable network analysis pipeline__

This exercise is building directly on the work we did in class. I want you to take the code we developed together and in you groups and turn it into a reusable command-line tool. You can see the code from class here:

https://github.com/CDS-AU-DK/cds-language/blob/main/notebooks/session6.ipynb

This command-line tool will take a given dataset and perform simple network analysis. In particular, it will build networks based on entities appearing together in the same documents, like we did in class.

- Your script should be able to be run from the command line
- It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB"
- For any given edgelist, your script should then create a weighed edgelist. This should be used to create a network visualization, which will be saved in a folder called viz.
- It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called output.

__Tips:__
- You should use argparse() in the Python standard library
- Your code should contain a main() function
- Don't worry too much about efficiency - networkx is really slow, there's no way around i!
- If you have issues with pygraphviz, just use the built-in matplotlib functions in networkx.
- You may want to create an argument for the user to define a cut-off point to filter data. E.g. only include node pairs with more than a certain edge weight.
- Make sure to use all of the Python scripting skills you've learned so far, including in the workshops with Kristoffer Nielbo

__Bonus challenges:__
- Attempt to implement coreference resolution on entities (time-consuming)
- Bundle your code up into a Python class, focusing on code modularity
- Let the user define which graphing algorithm they use (pretty tricky)
- Are there other ways of creating networks, rather than just document co-occurrence? (really tricky)

__General instructions:__
- For this assignment, you should upload a standalone .py script which can be executed from the command line
- Save your script as network.py
- You must include a requirements.txt file and a bash script to set up a virtual environment for the project. You can use those on worker02 as a template
- You can either upload the scripts here or push to GitHub and include a link - or both!
- Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

__Purpose:__

This assignment is designed to test that you have an understanding of:
1. how to create command-line tools with Python;
2. how to perform network analysis using networkx;
3. how to create resuable and reproducible pipelines for data analysis.


## Running the Script

__Cloning the Repository__

To run the script `network_nd.py`, it is best to clone this repository to your own machine/server and move into the `assignments/4-network` directory by running the following commands:

```bash
# clone repository into cds-language-nd
git clone https://github.com/nicole-dwenger/cds-language.git cds-language-nd

# move into directory of assignment4
cd cds-language-nd/assignments/4-network
```

__Dependencies__

To run the script `network_nd.py`, it is best to create the virtual environment using the bash script `create_venv.sh` and the requirements specified in `requirements.txt`. To install and activate the environment called `venv_assignment4`run the following commands: 

```bash
# create environment
bash create_venv.sh
# activate environment
source venv_assignment4/bin/activate
```

__Data__

For this assignement I used a csv file containing values of weights for edges extracted from real newspaper headlines.  These were extracted from the real_and_fake_news.csv data set. How they were extracted can be seen in the `notebooks/session_6.ipynb` file. The data is stored in `assignments/data/weighted_edgelist_realnews.csv`, and contains the columns 'nodeA', 'nodeB' and 'weight'. The columns 'nodeA' and 'nodeB', refer to named entities of persons extracted from the headlines, and the column 'weight' refers to the total number of occurances of the node pair, i.e. edge across the headlines. 
While this data is set as default input data, any other csv with the columns 'nodeA', 'nodeB' and 'weight' can be used to run the script.  

__Running the script__

The script takes two *optional* input parameters:
- `-i`: input csv file with columns nodeA, nodeB, weight
- `-m`: minimum edge weight of interest for network analysis, i.e. which edges should be included in graph and centrality measures

As default parameters, I defined the above described data as the input file and a minimum edge weight of 500. If you wish to run the script with these default values, you can simply run:

```bash 
python3 network_nd.py
```

If you which you change these parameters, you can run:

```bash 
python3 network_nd.py -i your_file.csv -m 200
```

__Output__

The output of the script is a network graph, and a dataframe with centrality measures of degree, betweennes and eigenvector. The network graph is saved in the the `viz/` directory and the dataframe of centrality measures in the `output/` directory. 

