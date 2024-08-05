from utils import get_config, get_images
from extract_features import compute_roma_sim, compute_xfeat_sim
from clustering import AGLP_clustering, proj_hdbscan, dissim_hdbscan
import numpy as np

if __name__ == '__main__':

    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    images = get_images(cfg['Dataset'])
    filtering = cfg['Filtering']
    FNAME = "similarities/matches.npy"

    if cfg['Matching']['Algo'] == "XFeat":
        compute_xfeat_sim.save_matches(images, top_k=int(cfg['Matching']['Params']['XFeat-TopK']), filtering=filtering, fname=FNAME)
    elif cfg['Matching']['Algo'] == "RoMa":
        compute_roma_sim.save_matches(images, threshold=float(cfg['Matching']['Params']['RoMa-Threshold']), fname=FNAME)
    else:
        raise ValueError('Wrong matching algorithm selected. Must be in : XFeat | RoMa')
    
    #### Clustering
    sim = np.load(FNAME)
    partition = []
    if cfg['Clustering'] == 'AGLP':
        partition = AGLP_clustering(sim)
    elif cfg['Clustering'] == 'HDBSCAN-Proj':
        partition = proj_hdbscan(sim)
    elif cfg['Clustering'] == 'HDBSCAN-Dissim':
        partition = dissim_hdbscan(sim)
    else:
        raise ValueError('Wrong Clustering selected. Must be in : AGLP | HDBSCAN-Dissim | HDBSCAN-Proj ')

    print(partition, partition.dtype, partition)
    #np.savetxt("die_studie.txt", partition, fmt="%i")