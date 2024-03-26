import networkx as nx
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_recall_fscore_support
from connect_utils import *

G = pickle.load(open('assets/email_prediction_NEW.txt', 'rb'))

network_feature_summary(G)

#Construct a list of all connected subgraphs
S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

#nx.draw_networkx(S[0])

#Remove unnecessary node data
type(G.nodes)
for i, _ in enumerate(G.nodes):
    G.nodes[i].pop('ManagementSalary')
list(G.nodes(data=True))[:5]

#Load CSV containing information on future connections
X_orig = pd.read_csv('assets/Future_Connections.csv', index_col=0, converters={0: eval})

X, X_test, y = construct_feature_matrix(G, X_orig, shortest_path=True, diam=nx.diameter(S[0]), resource=True)

# Split X and y into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

#Scale features in X to fall between 0 and 1
scaler = StandardScaler()
scaler.fit(X_train)

#train and test random forest classifier
clf = RandomForestClassifier(n_estimators = 50, max_depth = 8, random_state = 0)
clf.fit(scaler.transform(X_train), np.ravel(y_train))
y_probs = clf.predict_proba(scaler.transform(X_valid))
#print(roc_auc_score(y_valid, y_probs[:,1]))

#Load dataset containing missing future connection data
y_test = pd.read_csv('assets/Future_Connections_testing.csv', index_col=0, header = None, converters={0: eval})

y_probs_test = clf.predict_proba(scaler.transform(X_test))[:,1]

#Construct and plot the ROC curve. Indicate the point on the curve closest to the upper-left corner of the domain
fpr, tpr, thresh_info = roc_info(y_test, y_probs_test)

roc_plotter(fpr, tpr, thresh_info,'Random Forest')

#print(y_pred > 0.1)
rf_thresh = thresh_info[-1]
rf_auc = roc_auc_score(y_test, y_probs_test)
rf_acc = accuracy_score(y_test, (y_probs_test > rf_thresh))
#rf_recall = recall_score(y_test, (y_probs_test > rf_thresh))
rf_prec, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, (y_probs_test > rf_thresh))

print('We can compare the AUC and accuracy scores of each model:')

print(f'Random forest AUC score: {rf_auc:.3f}')
print(f'Random forest accuracy score at optimal probability threshold: {rf_acc:.3f}')
print(f'Random forest true positive rate (recall) at optimal probability threshold: {rf_recall[1]:.3f}')
print(f'Random forest precision at optimal probability threshold: {rf_prec[1]:.3f}')
print(f'Random forest F1 score at optimal probability threshold: {rf_f1[1]:.3f}')
print(f'Random forest false positive rate at optimal probability threshold: {thresh_info[0]:.3f}')


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

cm = confusion_matrix(y_test, y_probs_test>rf_thresh, labels=clf.classes_)

ax = sns.heatmap(cm, fmt='d', annot=True, square=True,
            cmap='gray_r', vmin=0, vmax=0,  # set all to white
            linewidths=1, linecolor='k',  # draw black grid lines
            cbar=False)                     # disable colorbar
ax.set(xlabel="Predicted label", ylabel="True label", title='Confusion matrix')