# Connection-Predictor
To demonstrate proficiency with network analysis using NetworkX


**Skills demonstrated:** classification, scikit-learn, NetworkX, model evaluation, general coding

Overview: 

> The nodes in the graph defined in assets/email_prediction_NEW.txt represent employees at a company and two nodes are connected if either employee has ever sent the other an email. The dataset assets/Future_Connections.csv contains pairs of currently unconnected nodes and indicates whether they will form a connection in the future (1) or not (0). This dataset is incomplete, with many labeled None. The missing future connection data is contained in assets/Future_Connections_testing.csv.


Purpose: 
> Train a classifier to predict whether the pairs of nodes with missing information in assets/Future_Connections.csv will form a connection in the future.

Datasets/graph information: 

> - assets/email_prediction_NEW.txt: Contains the graph of employees connected by emails
> - assets/Future_Connections.csv: CSV indexed by pairs of currently unconnected nodes with labels indicating whether they will form connections in the future. Some labels are missing.
> - assets/Future_Connections_testing.csv: CSV containing the labels missing in assets/Future_Connections.csv.
    
Features:

> Each node contains information identifying the department that employee works in and some salary data. The salary data is dropped and unused in this code. All other features are derived from the graph structure. In particular, we consider
> 1. the [Jaccard coefficient](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.jaccard_coefficient.html)
> 2. the shortest path between the two nodes if they are in the same connected component
> 3. the [resource allocation index](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.resource_allocation_index.html)
> 4. the [preferential attachment score](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.preferential_attachment.html)
> 5. the [common neighbor centrality score](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.common_neighbor_centrality.html#networkx.algorithms.link_prediction.common_neighbor_centrality)

Target variable:

> Future Connection - Binary (yes/no) label or probability of future connection between two nodes. 
