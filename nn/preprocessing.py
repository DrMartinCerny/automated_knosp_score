import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml

from datetime import datetime
from pathlib import Path
from skimage.io import imsave
from yaml.loader import SafeLoader

"""
Script for automated preprocessing of the input dataset.

2D dataset, 1 channel (grayscale)

Please, specify the correct paths below before execution.
"""


# PARAMS
h_crop, w_crop = 194, 194
labels = [0, 1, 2, 3, 4, 5] # 0, 1, 2, 3A, 3B, 4


# PATHS
input_folder = Path(r"C:\Users\Filip\OneDrive - České vysoké učení technické v Praze\Dokumenty\FEL\knosp\knosp_large\train")
new_dataset = Path(r"C:\Users\Filip\OneDrive - České vysoké učení technické v Praze\Dokumenty\FEL\knosp_clean\data\raw\train")
log =  new_dataset / Path("logs.txt")
csv_info = new_dataset / Path("info.csv")
print("Please make sure correct paths were set in the script.")

# FUNCTIONS

def normalize255(img: np.ndarray) -> np.ndarray:
    minval = img.min()
    maxval = img.max()
    out = 255. * ((img - minval) / (maxval - minval))

    return out.astype(np.uint8)


# PROCESSING
for label in labels:
    (new_dataset / str(label)).mkdir(exist_ok=True)

with open(log, 'w') as l:
    l.write(f"{datetime.now()} Folders initialized.\n")

subjects = next(os.walk(input_folder))[1]

saved = []
channels = []

for subject in tqdm(subjects):

    path = input_folder / Path(subject)

    img_data = nib.load(path / "COR_T1_C.nii")
    img = img_data.get_fdata().transpose(1,0,2)

    h, w, num_layers = img.shape
    channels.append(num_layers)
    h_margin = (h-h_crop)//2
    w_margin = (w-w_crop)//2
    crop_img = img[h_margin:h_margin+h_crop, w_margin:w_margin+w_crop, :]

    # Preallocate empty classifications:
    df_empty = pd.DataFrame({'knosp_left': [0] * num_layers, 'knosp_right': [0] * num_layers})

    # Populate from yaml file
    with open(path / "knosp.yaml", 'r') as f:
        scores_all_layers = yaml.load(f, Loader=SafeLoader)
    scores_all_layers = pd.DataFrame.from_dict(scores_all_layers, orient="index")
    scores_all_layers.index = scores_all_layers.index.astype(int)
    df_empty.update(scores_all_layers)
    scores_all_layers = df_empty

    # For each layer
    for layer in range(num_layers):
        slice_L = np.fliplr(crop_img[:,:,layer])
        slice_R = crop_img[:,:,layer]

        name = f"{subject}_{layer:02d}"

        score_R = scores_all_layers["knosp_right"][layer]
        score_L = scores_all_layers["knosp_left"][layer]
        path_R = new_dataset / str(score_R) / f"{name}_R.png"
        path_L = new_dataset / str(score_L) / f"{name}_L.png"
        imsave(path_R, normalize255(slice_R), check_contrast=False)
        imsave(path_L, normalize255(slice_L), check_contrast=False)

        saved.append([score_R, path_R.stem, subject, layer, 'R'])
        saved.append([score_L, path_L.stem, subject, layer, 'L'])

    with open(log, 'a') as l:
        l.write(f"{datetime.now()} Subject {subject} processed.\n")


pd_saved = pd.DataFrame(saved, columns=["label", "img_name",
                                        "subject", "layer", "side"])
pd_saved.to_csv(csv_info, index=False)
