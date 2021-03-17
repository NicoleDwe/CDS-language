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
import math


#### MAIN FUNCTION ####

def main():
    
    # define input parameters
    ap = argparse.ArgumentParser()
    # input option for filepath to the input file, with default of realnews data
    ap.add_argument("-i", "--input_file", 
                    required=False, 
                    help="path to input file", 
                    default="../data/weighted_edgelist_realnews.csv")
    # input option for minimum edgeweight, with default of 500
    ap.add_argument("-m", "--min_edgeweight", 
                    required=False, 
                    help="minimum edge weight of interest", 
                    default=500, 
                    type=int)
    # get the input
    args = vars(ap.parse_args())
    
    # save input parameters
    input_file = args["input_file"]
    min_edgeweight = args["min_edgeweight"]
    
    # run network analysis
    Network_Analysis(input_file, min_edgeweight)
    
    

#### BUNDLED UP NETWORK ANALYSIS CLASS ####

class Network_Analysis:
    
    
    def __init__(self, input_file, min_edgeweight): 
        """
        Main process of Network analysis: creating output directories, loading data, generating network
        graph and calcualting centrality measures 
        - Input: input path to the input csv file of weighted edgelist, minimum edgeweigth of interest
        - Ouput: network graph in viz/ and centrality_measures.csv in output/
        """
     
        # start message
        print("\nInitialising network analysis...")
       
        # get the name of the data
        file_name = os.path.basename(input_file) 
        self.data_name = os.path.splitext(file_name)[0] 
        
        # create output directories
        self.create_output_directory("output")
        self.create_output_directory("viz")
        
        # load data
        self.weighted_edgelist = pd.read_csv(input_file)
        # define minimum edgeweig
        self.min_edgeweight = min_edgeweight
        
        # create and save network graph 
        network_graph = self.network_graph()
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
    
    
    def network_graph(self):
        """
        Creating a network graph and saving it as a png file
        Input: list of all paris and their weight 
        Output: network figure saved in viz directory, and returned to be used for centrality measures
        """
        # filter based on minimum edge weight
        filtered_edges_df = self.weighted_edgelist[self.weighted_edgelist["weight"] > self.min_edgeweight]
        
        # draw graph with the filtered edgelist
        graph = nx.from_pandas_edgelist(filtered_edges_df, "nodeA", "nodeB", ["weight"])
        # define size of figure
        plt.figure(figsize=(15,15))
        
        # defining layout as spring layout, with increased distance between nodes
        spring_layout = nx.spring_layout(graph, k=math.sqrt(graph.order()))
        # drawing nodes, edges and labels
        nx.draw_networkx_nodes(graph, spring_layout, node_size=20, node_color="steelblue", alpha = 0.7)
        nx.draw_networkx_edges(graph, spring_layout, alpha = 0.3)
        nx.draw_networkx_labels(graph, spring_layout, font_size=9, verticalalignment="bottom", font_weight="semibold")
        
        # save the plot
        plt.savefig(f"viz/network_{self.data_name}_{self.min_edgeweight}.png", dpi=300, bbox_inches="tight")
        return graph
        
    
    def centrality_measures(self, network_graph):
        """
        Calculatinng the betweenness and eigenvector values for each edge and saving them in a dataframe
        Input: network graph
        Output: dataframe saved as csv file in output directory
        """
        # calcualte the three relevant metrics
        nodes = nx.nodes(network_graph)
        degree_metric = nx.degree_centrality(network_graph)
        betweenness_metric = nx.betweenness_centrality(network_graph)
        eigenvector_metric = nx.eigenvector_centrality(network_graph)
        # saving the three metrics in a dataframe
        centrality_df = pd.DataFrame({
            'degree':pd.Series(degree_metric),
            'betweenness':pd.Series(betweenness_metric),
            'eigenvector':pd.Series(eigenvector_metric),
        }).sort_values(['degree', 'betweenness', 'eigenvector'], ascending=False)
        # saving the csv file
        centrality_df.to_csv(f"output/centrality_measures_{self.data_name}_{self.min_edgeweight}.csv")
        
        
#### DEFAULT BEHAVIOUR ####
          
# in script is called from the command line, exectute main
if __name__ == "__main__":
    main()