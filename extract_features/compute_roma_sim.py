import numpy as np
from skimage import io
import torch
from romatch.models.model_zoo import roma_outdoor
import tqdm
import time
from PIL import Image

def save_matches(files, threshold=0.9, fname='similarities/matches.npy'):
    """
    Extract pairwise RoMa matches from a collection.

    Parameters:
        - files (list(str)) : List of image paths
        - threshold (float) : Certainty threshold to consider a match 
        - fname (str) : Path to save the results, as a `.npy` file
    """

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roma_model = roma_outdoor(device=device)

    def matches_two_files(im1, im2):
        _, certainty = roma_model.match(im1, im2, device=device)
        return (certainty > threshold).sum().item()

    similarities = np.zeros((len(files), len(files)))
    for i in tqdm.tqdm(range(len(files))):
        for j in range(len(files)):
            similarities[i][j] = matches_two_files(files[i], files[j])
    np.save(fname, similarities)

    print(f'Elapsed : {(time.time() - start_time):4f}s')