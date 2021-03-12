#!/usr/bin/env python

"""
From input csv of weighted edelist with "nodeA", "nodeB", "weight" columns, create a network graph and calculate measures of degree, betweennness and eigenvector 

Parameters (optional)
    input_file: str <file-path>
    min_edgeweight: int <min_edgeweight>
Usage:
    network_nd.py -i <file-path> -m <min_edgeweight>
Example:
    $ python3 network_nd.py -i ../data/weighted_edgelist_realnews.csv -m 500
"""


#### DEPENDENCIES ####

import os
import argparse

# data
import pandas as pd

# network
import networkx as nx
import matplotlib.pyplot as plt


#### MAIN FUNCTION ####

def main():
    
    # define input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", required=False, help="path to input file", default="../data/weighted_edgelist_realnews.csv")
    ap.add_argument("-m", "--min_edgeweight", required=False, help="minimum edge weight of interest", default=500, type=int)
    args = vars(ap.parse_args())
    
    # save input parameters
    input_file = args["input_file"]
    min_edgeweight = args["min_edgeweight"]
    
    # run network analysis
    Network_analysis(input_file, min_edgeweight)
    
    

#### BUNDLED UP NETWORK ANALYSIS CLASS ####

class Network_analysis:
    
    
    def __init__(self, input_file, min_edgeweight): 
        """
        Main process of Network analysis: creating output directories, loading data, generating network
        graph and calcualting centrality measures 
        - Input: input path to the input csv file of weighted edgelist, minimum edgeweigth of interest
        - Ouput: network graph in viz/ and centrality_measures.csv in output/
        """
     
        # start message
        print("\nInitialising network analysis...")
        
        # create output directories
        self.create_output_directory("output")
        self.create_output_directory("viz")
        
        # load data
        weighted_edgelist = pd.read_csv(input_file)
        
        # create and save network graph 
        network_graph = self.network_graph(weighted_edgelist, min_edgeweight)
        
        # calculate and save centraliy measures
        centrality_measures = self.centrality_measures(network_graph)
        
        print("All done, network graph saved in viz/ and centrality measures in output/!\n")
        
    
    def create_output_directory(self, directory_name):
        """
        Creatig output directory in the current directory, if it does not exist
        Input: name of the directory
        """
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
    
    
    def network_graph(self, weighted_edges_df, min_edgeweight):
        """
        Creating a network graph and saving it as a png file
        Input: list of all paris and their weight 
        Output: network figure saved in viz directory, and returned to be used for centrality measures
        """
        # filter based on minimum edge weight
        filtered_edges_df = weighted_edges_df[weighted_edges_df["weight"] > min_edgeweight]
        # create graph using nx package
        network_graph = nx.from_pandas_edgelist(filtered_edges_df, "nodeA", "nodeB", ["weight"])
        pos = nx.nx_agraph.graphviz_layout(network_graph, prog = "neato")
        nx.draw(network_graph, pos, with_labels=True, node_size=20, font_size=10)
        # saving network graph 
        plt.savefig("viz/network.png", dpi=300, bbox_inches="tight")
        return network_graph
        
    
    def centrality_measures(self, network_graph):
        """
        Calculatinng the betweenness and eigenvector values for each edge and saving them in a dataframe
        Input: network graph
        Output: dataframe saved as csv file in output directory
        """
        # calcualte the three relevant metrics
        degree_metric = nx.degree_centrality(network_graph)
        betweenness_metric = nx.betweenness_centrality(network_graph)
        eigenvector_metric = nx.eigenvector_centrality(network_graph)
        # saving the three metrics in a dataframe
        centrality_df = pd.DataFrame({
            'degree':pd.Series(degree_metric),
            'betweenness':pd.Series(betweenness_metric),
            'eigenvector':pd.Series(eigenvector_metric)  
        }).sort_values(['degree', 'betweenness', 'eigenvector'], ascending=False)
        # saving the csv file
        centrality_df.to_csv("output/centrality_measures.csv")
        
        
#### DEFAULT BEHAVIOUR ####
          
# in script is called from the command line, exectute main
if __name__ == "__main__":
    main()