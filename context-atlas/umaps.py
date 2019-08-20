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

RANDOM_SEED = 42

def project_umap(points, seed=RANDOM_SEED):
  """Project the words (by layer) into 3 dimensions using umap."""

  print('Projecting to umap with seed %d'%seed)
  points_transformed = []
  for layer in points:
    reducer = umap.UMAP(random_state=seed)
    transformed = reducer.fit_transform(layer).tolist()
    points_transformed.append(transformed)
  return points_transformed

def main():
  for file in glob.glob('static/embeddings/*.json'):
    word = file.split('/')[-1].split('.')[0]
    print("Processing %s ..."%word)

    # read
    with open(file) as f:
      checkpoint = json.load(f)
    points = checkpoint['points']
    labels = checkpoint['labels']
    
    # write
    for seed in range(0, 3):
      print("  seed %d ..."%seed)
      points_for_seed = project_umap(points, seed=seed)
      if not os.path.exists('static/umaps/%d'%seed):
        os.mkdir('static/umaps/%d'%seed)
      with open('static/umaps/%d/%s.json'%(seed,word), 'w') as outfile:
        json.dump({'labels':labels, 'data':points_for_seed}, outfile)

if __name__ == '__main__':
  main()