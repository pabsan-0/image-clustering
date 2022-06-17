import torch
from torch import optim, nn
from torchvision import models, transforms
import cv2
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import numpy as np
import glob
import random
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import sys



def print_usage():
    print(f"python3 {sys.argv[0]} path/feats.pickle [path/out.csv]")


if __name__ == "__main__":

    try:
        load_path = sys.argv[1]
        assert load_path.split('.')[1] in ['pickle', 'pkl']
        with open(load_path,'rb') as file:
            featurizer_out = pickle.load(file)

    except Exception:
        print("Bad input file.")
        print_usage()
        quit()

    # Load results from featurizer output
    images   = featurizer_out['images']
    features = featurizer_out['features']
    airports = featurizer_out['airports']


    ## Clustering
    CLUSTERS = 10    
    model = KMeans(n_clusters=CLUSTERS, random_state=42)
    model.fit(np.array(features))
    labels = model.labels_


    ## Analysis

    # Dataframe and confusion matrix
    df = pd.DataFrame({'label': labels, 'airport': airports, 'image': images})
    cm = pd.crosstab(df.airport, df.label)
    print(cm)

    # Plotting the correlation matrix
    sns.set_theme(style="white")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cm, cmap=cmap, linewidths=.5, square=True, annot=True, 
        center=True, annot_kws={"size": 7}, fmt="d")
    
    # Export dataframe to file
    try:
        save_path = sys.argv[2]
        assert save_path.split('.')[1] == 'csv'
        df.to_csv(save_path)
    except Exception:
        print("No valid save path found for csv results. This is not an error.")

    plt.show()