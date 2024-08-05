import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from extract_features import compute_xfeat_sim, compute_roma_sim
from utils import get_config, get_images

if __name__ == '__main__':
    #### Get Config
    cfg = get_config()

    #### Extract Similarities
    images = get_images(cfg['Dataset'])
    filtering = cfg['Filtering']
    FNAME = "similarities/matches.npy"

    if cfg['Matching']['Algo'] == "XFeat":
        FNAME = "similarities/xfeat_example.npy"
        compute_xfeat_sim.save_matches(images, top_k=int(cfg['Matching']['Params']['XFeat-TopK']), fname=FNAME)
    elif cfg['Matching']['Algo'] == "RoMa":
        compute_roma_sim.save_matches(images, threshold=float(cfg['Matching']['Params']['RoMa-Threshold']), fname=FNAME)
    else:
        raise ValueError('Wrong matching algorithm selected. Must be in : XFeat | RoMa')
    
    print("Saved to", FNAME)