import time
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
import tqdm
import cv2
import numpy as np
from hdbscan import HDBSCAN
from umap import UMAP
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

MEDIAN_BLUR_KSIZE = 3
PATCH_SIZE = 31

def open_process_image(path_img):
    """
    Preprocessing one CADS image.
    """

    img = cv2.imread(path_img)

    #img = cv2.resize(img, (int(img.shape[0]/5), int(img.shape[1]/5)), interpolation = cv2.INTER_AREA)[40:520, 175:687]
    img_bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_bw = cv2.medianBlur(img_bw,MEDIAN_BLUR_KSIZE)
    return img_bw


def cads_distance(files):
    """
    Compute CADS-like distances

    Parameters:
        - files(list(str)) : Names of image files

    Returns:
        - dissimilarities (numpy.array) : CADS-like Pairwise dissimilarity matrix
    """

    N = len(files)

    IMAGES = [open_process_image(f) for f in files]

    orb = cv2.ORB_create(nfeatures = 500, patchSize = PATCH_SIZE)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ORB_LIST = [orb.detectAndCompute(im, None) for im in tqdm.tqdm(IMAGES, desc='ORB')]

    dissimilarities = np.zeros((N, N))

    for i in tqdm.tqdm(range(N), desc='Matching'):
        for j in range(N):
            desc_A = ORB_LIST[i][1]
            desc_B = ORB_LIST[j][1]
            matches = matcher.match(desc_A,desc_B)
            matches = sorted(matches, key = lambda x:x.distance)

            up = np.min([20, len(matches)])
            dist = np.mean([matches[i].distance for i in range(up)])
            dissimilarities[i][j] = dist
            dissimilarities[j][i] = dist

    return dissimilarities

def cads_hdbscan(files):
    """
    CADS-HDBSCAN reproduction of CADS. Contrary to original CADS, this one is fully automatic and does not require human-selected hierarchy disambiguation.

    Parameters:
        - files(list(str)) : Names of image files
    Returns:
        - out_hdbscan(list) : List of die labels
    
    """

    mat = cads_distance(files)
    predictor =  HDBSCAN(min_cluster_size=2, min_samples=1, 
                     metric="precomputed",
                     match_reference_implementation=True)
    raw_hdbscan = predictor.fit_predict(mat)

    # `hdbscan.HDBSCAN` does not return clusters of size==2
    outout=mat[:][raw_hdbscan==-1][:,raw_hdbscan==-1]
    outliers_indices = np.where(np.array(raw_hdbscan) == -1)[0]
    closest = pd.DataFrame(outout).apply(lambda x: np.argpartition(x, 2)[1])
    tot = 0
    for idx,c in enumerate(closest):
        if closest[c] == idx and idx < c: 
            tot+=1
            clust_idx = idx + 1 + len(raw_hdbscan)
            raw_hdbscan[outliers_indices[ c ]] = clust_idx
            raw_hdbscan[outliers_indices[idx]] = clust_idx

    out_hdbscan = []
    for i, l in enumerate(raw_hdbscan):
        if l == -1:
            out_hdbscan.append(i + len(raw_hdbscan))
        else:
            out_hdbscan.append(l)

    return out_hdbscan


def cads_ag_star(files, labels):
    """
    Reproduction of CADS. Original CADS creates a hierarchy between coins, but does not automatically select a cut from this hierarchy to build clusters, and requires a human intervention. Therefore, we return the partition with the highest possible AMI obtainable with this approach.  

    Parameters:
        - files(list(str)) : Names of image files
        - labels
    Returns:
        - selected(list) : List of die labels
    """

    mat = cads_distance(files)
    
    N = len(mat)
    partitions = [AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete').fit_predict(mat) for k in tqdm.tqdm(range(1,N))]
    amis = np.array([adjusted_mutual_info_score(p, labels) for p in partitions])
    selected = partitions[np.argmax(amis)]
    return selected