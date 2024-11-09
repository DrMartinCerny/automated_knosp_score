import csv
import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from src.methods import classify_vessels, find_lines, classify_carcinoma, preprocess_mask
from src.visualization import save_layer_img


def export_outputs(lst: list, out_folder: str, name: str = "dataset.csv") -> None:
    """Export outputs to csv.

    Args:
        lst (list): list of entries to be exported
        out_folder (str): destination folder
        name (str, optional): name of the output file. Defaults to "dataset.csv".
    """
    Path(out_folder).mkdir(exist_ok=True, parents=True)
    df = pd.concat(lst)
    df.to_csv(Path(out_folder)/name)


def export_scan_scores(out_folder, scores):
    """Export the scores of the scan to csv.
    """

    csv_name = os.path.join(out_folder, "knosp.csv")
    for i in range(len(scores)+1):
        if i==0:
            with open(csv_name, 'w', newline='') as f:
                wrt = csv.writer(f)
                wrt.writerow(["layer", "left", "right"])
                f.close()
        else:
            with open(csv_name, 'a', newline='') as f:
                wrt = csv.writer(f)
                wrt.writerow(scores[i-1,:])
                f.close()
    with open(csv_name, 'a', newline='') as f:
        wrt = csv.writer(f)
        wrt.writerow(["overall"]+list(np.max(scores[:,1:], axis=0)))
        f.close()


def process_scan(subject: str, in_folder: str, out_folder: str,
                 mode: str = "auto", plots: bool = False,
                 subfolders: bool = False, th_extent: float = 0.5) -> None:

    """Processing of one scan.
    """

    outputs = []

    # Prepare paths
    if subfolders or plots:
        Path(out_folder).mkdir(parents=True, exist_ok=True)
    
    img_path = os.path.join(in_folder, "COR_T1_C.nii")
    
    if mode == "auto":
        mask_path = os.path.join(in_folder, "predicted.nii")
    else:
        mask_path = os.path.join(in_folder, "mask.nii")

    # Load data with NiBabel
    img_data = nib.load(img_path)
    img = img_data.get_fdata()
    mask_data = nib.load(mask_path)
    mask = mask_data.get_fdata()

    num_layers = img.shape[2]
    scores = []

    # For each layer
    for layer in range(num_layers):
        img_slice = img[:,:,layer].T
        mask_slice = mask[:,:,layer].T

        if mode == "auto":
            mask_slice = preprocess_mask(mask_slice)

        vessels = (mask_slice==2).astype(int)
        carcinoma = (mask_slice==1).astype(int)

        classified_vessels = classify_vessels(vessels)

        if classified_vessels is None:
            outputs.append([subject, in_folder, layer, "right", 0])
            outputs.append([subject, in_folder, layer, "left", 0])
            continue
    
        LT, LB, RT, RB = classified_vessels
        left_lines = find_lines(LT, LB)
        right_lines = find_lines(RB, RT) # B and T reversely, because "out" is meant left to line

        score_l, score_r, carc_mask = classify_carcinoma(carcinoma, left_lines, right_lines, th_extent)
        scores.append([layer, score_l, score_r])

        if plots:
            out_name = os.path.join(out_folder, "layer_{:0>2d}.png".format(layer))
            save_layer_img(out_name, img_slice, vessels, carc_mask, left_lines, right_lines,
                            layer, score_l, score_r)
        
        # remember outputs
        outputs.append([subject, in_folder, layer, "right", score_r])
        outputs.append([subject, in_folder, layer, "left", score_l])

    scores = np.array(scores)
    
    # Export csv
    if subfolders:
        export_scan_scores(out_folder, scores)

    # Output for all results
    df = pd.DataFrame(outputs, columns=["subject", "img_folder", "layer", "side", "predicted_label"])

    return df
