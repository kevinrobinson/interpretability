from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.neighbors import NearestNeighbors
import sqlite3 as sql
import re
import numpy as np
import umap
import json
from tqdm import tqdm
import nltk
from sklearn.cluster import KMeans



RANDOM_SEED = 42


# seed is actually n_clusters
def project_to_cluster_on_umap(points, seed=RANDOM_SEED):
  umap_cluster_xys_for_layers = []
  for layer in points:
    kmeans = KMeans(random_state=seed, n_clusters=5)
    clusters = kmeans.fit_predict(layer)
    xys = list([int(cluster), int(index)] for cluster, index in enumerate(clusters))
    # print(xys)
    umap_cluster_xys_for_layers.append(xys)
  return umap_cluster_xys_for_layers


# def project_umap(points, seed=RANDOM_SEED):
#   """Project the words (by layer) into 3 dimensions using umap."""

#   print('Projecting to umap with seed %d'%seed)
#   points_transformed = []
#   for layer in points:
#     reducer = umap.UMAP(random_state=seed)
#     transformed = reducer.fit_transform(layer).tolist()
#     points_transformed.append(transformed)
#   return points_transformed

def main():
  # on single umap seed
  for file in glob.glob('static/100k/umaps/0/*.json'):
    word = file.split('/')[-1].split('.')[0]
    print("Processing %s ..."%word)

    # read
    with open(file) as f:
      checkpoint = json.load(f)
    points = checkpoint['data']
    labels = checkpoint['labels']
    
    # write
    for seed in range(0, 16):
      print("  seed %d ..."%seed)
      points_for_seed = project_to_cluster_on_umap(points, seed=seed)
      if not os.path.exists('static/defaults_clusters_on_umap/%d'%seed):
        os.mkdir('static/defaults_clusters_on_umap/%d'%seed)
      with open('static/defaults_clusters_on_umap/%d/%s.json'%(seed,word), 'w') as outfile:
        json.dump({'labels':labels, 'data':points_for_seed}, outfile)

if __name__ == '__main__':
  main()