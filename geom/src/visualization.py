import matplotlib.pyplot as plt
import numpy as np

def save_layer_img(out_name: str, img_slice: np.ndarray, vessels: np.ndarray,
                    carc_mask: np.ndarray, left_lines: list, right_lines: list,
                    layer: int, score_l: float, score_r: float):
    """Visualization utility to save the slice with classification overlay.

    Args:
        out_name (str): path to output file
        img_slice (np.ndarray): the slice
        vessels (np.ndarray): mask of arteries
        carc_mask (np.ndarray): mask of tumour
        left_lines (list): left critical lines
        right_lines (list): right critical lines
        layer (int): layer's number
        score_l (float): left score
        score_r (float): right score
    """

    carc_mask[carc_mask==10] = 15
    carc_mask[carc_mask==11] = 19
    carc_mask[carc_mask==12] = 22
    carc_mask[carc_mask==13] = 24

    labels = ['0', '1', '2', '3A', '3B', '4']

    plt.imshow(img_slice, cmap="gray")
    plt.imshow(carc_mask, alpha=0.5*(carc_mask>0).astype(float), vmin=0, vmax=36, cmap="gist_ncar")
    plt.imshow(vessels, alpha=0.75*vessels, vmin=0, vmax=1, cmap="OrRd")
    for line in left_lines:
        plt.axline(line.a[::-1], line.b[::-1], color="white", linewidth=0.5, linestyle=':')
    for line in right_lines:
        plt.axline(line.a[::-1], line.b[::-1], color="white", linewidth=0.5, linestyle=':')
    plt.title("Layer "+str(layer)+" - score left: "+labels[score_l]+", score right: "+labels[score_r])
    plt.axis("off")

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(out_name, bbox_inches="tight", dpi=150, format='png')

    plt.close()
