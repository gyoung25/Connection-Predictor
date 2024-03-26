import networkx as nx
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score


def path_length(G, x,y,diam=7):
    '''
    Define a path length function that handles NetworkxNoPath exceptions
    
    Arguments
        G: a graph
        x,y: two nodes to find the shortest path length between
        diam: either the diameter of the largest connected component or another large number
    Returns
        The shortest path length if x and y are in the same connected component or 2*diam if they are not
    '''

    try:
        return nx.shortest_path_length(G,x,y)
    except nx.NetworkXNoPath:
        return diam*2
    
def construct_feature_matrix(G, future_connections_df, jaccard=True, same_dept=False, shortest_path=False, diam=7,
                            resource=False, pref_attach=False, ccpa=False):
    
    '''
    Arguments
        G: A graph
        future_connections_df: DataFrame indexed by node pairs with column 'Future Connection' containing
                               information about whether a connection will be made between the pair of nodes
        jaccard: Boolean. If true, include the Jaccard coeffiecient for each node pair in X
        Same_dept: Boolean. If true, include in X a 1 if the nodes correspond to individuals in the same dept,
                   0 if different depts.
        shortest_path: Boolean. If true, include in X the shortest path between the two nodes if they are in
                       the same connected component, 2*diameter of the largest connected component if in
                       different connected components.
        diam: Diameter of the largest connected component. Only necessary if shortest_path is True.
        resource: Boolean. If true, include in X the resource allocation index of each node pair.
        pref_attach: Boolean. If true, include in X the preferential attachment of each node pair.
        ccpa: Booolean. If true, include in X the common neighbor centrality of each node pair.
        
    Returns
        X: DataFrame indexed by node pairs with columns containing relational information about those nodes,
           to be used for training
        X_test: Dataframe containing the same columns as X, to be used for testing the classifier
        y: DataFrame containing future connection information for the node pairs contained in X
    '''
    #construct feature matrix
    X = future_connections_df.copy()

    edges = list(X.index)
    depts = pd.Series(nx.get_node_attributes(G, 'Department'), index = X.index)
    
    if jaccard:
        X['Jaccard'] = pd.Series([x[2] for x in nx.jaccard_coefficient(G, ebunch = edges)], index = X.index)
        
    if same_dept:
        X['Same_dept'] = pd.Series([(depts[i] == depts[j])*1 for (i,j) in edges], index = X.index)

    if shortest_path:
        X['Shortest_Path'] = pd.Series([path_length(G,x,y,diam) for (x,y) in X.index], index = X.index)
    
    if resource:
        X['Resource'] = pd.Series([x[2] for x in nx.resource_allocation_index(G, ebunch = edges)], index = X.index)
    
    if pref_attach:
        X['Pref_attach'] = pd.Series([x[2] for x in nx.preferential_attachment(G, ebunch = edges)], index = X.index)
    
    if ccpa:
        X['CCPA'] = pd.Series([x[2] for x in nx.common_neighbor_centrality(G, ebunch = edges)], index = X.index)
    
    #Split X into two DataFrames: one containing node pairs where future connection status is known, one containing
    # node pairs where future connection status is not known
    X_test = X[pd.isnull(X['Future Connection'])].copy()
    X = X[~pd.isnull(X['Future Connection'])].copy()
    #Store future connection status among node pairs with known future connection status in y
    y = X.pop('Future Connection')
    X_test.pop('Future Connection')
    
    return X, X_test, y


def roc_info(y_test, y_prob):
    '''
    Plot the ROC curve for each classifier above and return the probability threshold value that minimizes the Euclidean
    distance to fpr = 0, tpr = 1, thereby optimizing the fpr-tpr tradeoff.
    
    Arguments
        y_test: test target Series
        y_prob: output of any classifier giving the probability that a video will be engaging
    Returns
        fpr: x-coordinates of the ROC curve
        tpr: y-coordinates of the ROC curve
        (x0, y0, thresh): the (x,y) coordinate and corresponding probability threshold value that minimizes the Euclidean                                   distance to fpr = 0, tpr = 1, thereby optimizing the fpr-tpr tradeoff
    '''
    from sklearn.metrics import roc_curve
    from numpy import where, abs, argmin
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # determine probability threshold that minimizes the distance to (fpr,tpr)=(0,1), 
    # which is the optimal threshold in the sense of tpr-fpr tradeoff
    dist = fpr**2 + (tpr-1)**2
    ind = argmin(dist)
    x0, y0, thresh = fpr[ind], tpr[ind], thresholds[ind]
    
    return fpr, tpr, (x0, y0, thresh)

def roc_plotter(fpr, tpr, opt_thresh, clf_type):
    '''
    Plot the ROC curve for each classifier above.
    
    Arguments
        fpr: x-coordinates of the ROC curve returned by roc_info()
        tpr: y-coordinates of the ROC curve returned by roc_info()
        opt_thresh: 3-tuple of the form (x0, y0, thresh), where the (x0,y0) is the coordinate and thresh is the corresponding                         probability threshold value that minimizes the Euclidean distance to fpr = 0, tpr = 1, thereby optimizing the                     fpr-tpr tradeoff returned by roc_info()
        clf_type: String used to label the corresponding ROC curve on the generated plot
    '''
    x0, y0, thresh = opt_thresh
    plt.plot(fpr, tpr, linewidth = 2, label = clf_type)
    plt.plot(x0, y0, 'o', markersize = 10)#, label = f'Threshold {thresh:.3f}')
    plt.legend()
    plt.title('ROC curves', size = 14)
    plt.xlabel('False Positive Rate', size = 14)
    plt.ylabel('True Positive Rate', size = 14)
    return None

def network_feature_summary(G):
    '''
    Prints a summary of features of graph G, noting
        how many nodes and edges,
        whether the graph is directed, weighted, or connected,
        how many connected components it contains, and
        the size of the three largest and three smallest connected components.
    Arguments
        G: a graph
    '''
    if nx.is_directed(G):
        is_directed = 'directed'
    else:
        is_directed = 'undirected'
        
    
    if nx.is_weighted(G):
        is_weighted = 'weighted'
    else:
        is_weighted = 'unweighted'
        
        
    if not nx.is_directed(G):
        if nx.is_connected(G):
            is_connected = 'connected'
        else:
            is_connected = 'unconnected'
    
    connected_comps = nx.connected_components(G)
    connected_comps_list = list(connected_comps)
    connected_comps_list.sort(key=len, reverse=True)
    
    connected_comps_len = [len(x) for x in connected_comps_list]
    num_conn_comps = len(connected_comps_len)
    
    if num_conn_comps > 5:
        extreme_conn_comps = [None]*6
        extreme_conn_comps[:3] = connected_comps_len[:3]
        extreme_conn_comps[-3:] = connected_comps_len[-3:]
        extreme_conn_text = str(f'The three largest connected components are of size {extreme_conn_comps[0]}, '
                                f'{extreme_conn_comps[1]}, and {extreme_conn_comps[2]}.\n'
                                f'The three smallest connected components are of size {extreme_conn_comps[3]}, '
                                f'{extreme_conn_comps[4]}, and {extreme_conn_comps[5]}.')
    else:
        extreme_conn_text = str('The connected components are of size ' + str(connected_comps_len))
    
    
    print(f'The graph contains {len(nx.nodes(G))} nodes and {len(nx.edges(G))} edges '
          f'and is {is_directed}, {is_weighted}, and {is_connected}.')
    print(f'It contains {len(connected_comps_len)} connected components.')
    print(extreme_conn_text)
    
    return None