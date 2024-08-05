import numpy as np

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import get_labels
from clustering import AGLP_clustering, proj_hdbscan, dissim_hdbscan, evaluate_clustering, ConnectedComponents_clustering


import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    labels = get_labels("Paphos").astype(str)

    sim = np.load('similarities/example_xfeat_paphos.npy')

    #print(' --- AGLP ---')
    #partition = AGLP_clustering(sim)
    #print(evaluate_clustering(partition, labels))
    #print('\n --- Connected Components ---')
    #partition = ConnectedComponents_clustering(sim)
    #print(evaluate_clustering(partition, labels))
#
    #print('\n --- Dissimilarity HDBSCAN ---')
    #partition = dissim_hdbscan(sim)
    #print(evaluate_clustering(partition, labels))

    for _ in range(5):
        print('\n --- Projection HDBSCAN ---')
        partition = proj_hdbscan(sim)
        print(evaluate_clustering(partition, labels))