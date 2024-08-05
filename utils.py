import yaml
import glob
import pandas as pd
from skimage import io
import os

def get_config(cfg_file='config.yaml'):
    """
    Utility to read configuration from YAML.

    Parameters:
        - cfg_file (str) : YAML file name
    Returns:
        - cfg (dict) : Configuration as a dict
    """
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_images(dataset):
    """
    Get images names for a dataset. Supposes that files are all in the same folder, as PNGs or JPGs.

    Parameters:
        - dataset (str) : Dataset name, subfolder in `datasets`
    Returns:
        - files (list(str)) : File names
    """
    files = glob.glob(f"datasets/{dataset}/*.*")
    files = list(
        filter(
            lambda f: f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'), 
            files
            )
        )
    return sorted(files)

def get_labels(dataset):
    """
    Get die labels for a dataset. Only needed for clustering evaluation.

    Parameters:
        - dataset (str) : Dataset name, subfolder in `datasets`
    Returns:
        - files (list(str)) : Dies corresponding to each coin
    """
    df = pd.read_excel(f'datasets/{dataset}_Dies.xlsx')
    df = df.sort_values(by="File")
    return df['Die']


def split_obverse_reverse(origin_folder, target_folder, half="left"):
    """
    Utility to save only the left/right half of photos in a directory.

    Parameters:
        - origin_folder (str) : Folder containing images to split
        - target_folder (str) : Folder to save half-images, with the same file names as in `origin_folder`
        - half (str, "left"|"right") : Whether to keep left or right half of the image
    """

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    files = glob.glob(f"{origin_folder}/*.*")
    files = list(
        filter(
            lambda f: f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'), 
            files
            )
        )
    
    images = [io.imread(f) for f in files]

    if half == "left":
        images = [im[:,:im.shape[1]//2,:] for im in images]
    elif half == "right":
        images = [im[:,im.shape[1]//2:,:] for im in images]
    else:
        raise ValueError("Selected half must be in : right | left")
    
    for fpath, im in list(zip(files, images)):
        io.imsave(fpath.replace(origin_folder, target_folder), im)


def create_labels_excel(files, dies, collection_name):
    """
    Create labels files in Excel format, readable from `clustering.evaluate_clustering`.

    Parameters:
        - files (list(str)) : list of file paths, the images of the collections
        - dies (list) : list of die labels


    Excel file structure:
    ```
             File Die
    1   first.jpg  D1
    2  second.jpg  D2
    3   third.jpg  D3
    ```
    """

    df = pd.DataFrame()
    df['File'] = files
    df['Die'] = dies
    df.index += 1
    df.to_excel(f'{collection_name}.xlsx')



if __name__ == '__main__':
    print(get_config())
    print('\n'.join(get_images('Paphos')[:5]))