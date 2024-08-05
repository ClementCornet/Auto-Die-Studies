import numpy as np
from skimage import io
import torch
from extract_features.xfeat_cache import XFeat
import tqdm
import pandas as pd
import time
import cv2

def save_matches(files, top_k=5000, filtering=True, fname='similarities/matches.npy'):
    """
    Extract pairwise XFeat matches from a collection.

    Parameters:
        - files (list(str)) : List of image paths
        - top_k (int) : Maximum number of regions to match through XFeat
        - fname (str) : Path to save the results, as a `.npy` file
    """

    start_time = time.time()

    IMAGES = [io.imread(im) for im in files]
    im_list = []
    # if len(IMAGES[0].shape) == 3:
    #     im_list = [torch.tensor(im.transpose(2,0,1)) for im in IMAGES]
    # else:
    #     im_list = [torch.tensor(im[None,:,:]) for im in IMAGES]
    # previous code is fine if __all__ images have the same size as the first one
    for im in IMAGES:
        if len(im.shape) == 3:
            im_list.append(torch.tensor(im.transpose(2,0,1)))
        else:
            im_list.append(torch.tensor(im[None,:,:]))
    xfeat = XFeat(top_k=top_k)
    xfeat.cache_feats(im_list)

    
    def matches_two_files(i1, i2):
        #x1 = torch.tensor(IMAGES[i1].transpose(2,0,1))
        #x2 = torch.tensor(IMAGES[i2].transpose(2,0,1))
        matches_list = xfeat.match_xfeat_star_from_cache(i1, i2)
        if not filtering: return len(matches_list[0])
        
        _, mask = cv2.findHomography(
            matches_list[0],
            matches_list[1],
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=8,
        )
        return mask.sum()

    similarities = np.zeros((len(files), len(files)))
    for i in tqdm.tqdm(range(len(files)), desc='Matching'):
        for j in range(len(files)):
            similarities[i][j] = matches_two_files(i, j)
    np.save(fname, similarities)

    print(f'Elapsed : {(time.time() - start_time):4f}s')
