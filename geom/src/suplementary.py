import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, binary_erosion


"""
Suplementary functions that are not directly used
in the pipeline but can be useful.
"""


def rename_grades(score: float) -> str:
    """Get string representation of the grade.

    Args:
        score (float): float representation

    Returns:
        str: string representation
    """
    if score < 3:
        return str(score)
    elif score == 3.1:
        return '3A'
    elif score == 3.2:
        return '3B'
    elif score == 3.3:
        return '3C'
    elif score == 4:
        return '4'
    else:
        return 'unknown'


def crop194(img):
    """Crop height and width to 194 px.

    Args:
        img: input image

    Returns:
        cropped image
    """
    h, w = img.shape
    h_start = h//2 - 97
    w_start = w//2 - 97

    return img[h_start:h_start+194, w_start:w_start+194], h_start, w_start


def save_overlay_LR(out_folder: str, name: str, img_slice: np.ndarray, vessels: np.ndarray,
                    carc_mask: np.ndarray, left_lines: list, right_lines: list,
                    score_l: float, score_r: float, crop: bool = True):
    
    """Visualizations independently for left and right sides.

    Args:
        out_folder (str): output folder
        name (str): name of image
        img_slice (np.ndarray): slice of the scan (2D array)
        vessels (np.ndarray): mask of arteries
        carc_mask (np.ndarray): mask of tumour
        left_lines (list): left critical lines
        right_lines (list): right critical lines
        score_l (float): left score
        score_r (float): right score
        crop (bool, optional): use cropping to 194×194 px if True. Defaults to True.
    """

    carc_mask[carc_mask==10] = 15
    carc_mask[carc_mask==11] = 19
    carc_mask[carc_mask==12] = 22
    carc_mask[carc_mask==13] = 24

    h, w = carc_mask.shape

    mask_l = carc_mask.copy()
    mask_l[:, w//2:] = np.minimum(mask_l[:, w//2:], 15)
    mask_r = carc_mask.copy()
    mask_r[:, :w//2] = np.minimum(mask_r[:, :w//2], 15)
    
    vessels_smooth = binary_erosion(binary_dilation(vessels, footprint=np.ones((3,3))))
    rims = np.logical_and(vessels_smooth, ~binary_erosion(binary_erosion(vessels_smooth, footprint=np.ones((3,3))), footprint=np.ones((3,3))))

    if crop:
        img_slice, h_margin, w_margin = crop194(img_slice)
        mask_l, _, _ = crop194(mask_l)
        mask_r, _, _ = crop194(mask_r)
        rims, _, _ = crop194(rims)
    else:
        h_margin, w_margin = 0, 0

    margin = np.array([h_margin, w_margin])
    
    # Left -----------------------------------------------------------

    plt.imshow(img_slice, cmap="gray")
    plt.imshow(mask_l, alpha=0.5*(mask_l>0).astype(float), vmin=0, vmax=36, cmap="gist_ncar")
    plt.imshow(rims, alpha=0.75*rims, vmin=0, vmax=1, cmap="OrRd")
    for line in left_lines:
        ptA = line.a[::-1]
        ptB = line.b[::-1]
        plt.axline(ptA-margin, ptB-margin, color="white", linewidth=3, linestyle=':')
    plt.axis("off")
    plt.gca().invert_xaxis()

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(out_folder+"/"+rename_grades(score_l)+"/"+name+"_L.png",
                bbox_inches="tight", dpi=150, format='png', pad_inches = 0)

    plt.close()

    # Right ----------------------------------------------------------

    plt.imshow(img_slice, cmap="gray")
    plt.imshow(mask_r, alpha=0.5*(mask_r>0).astype(float), vmin=0, vmax=36, cmap="gist_ncar")
    plt.imshow(rims, alpha=0.75*rims, vmin=0, vmax=1, cmap="OrRd")
    for line in right_lines:
        ptA = line.a[::-1]
        ptB = line.b[::-1]
        plt.axline(ptA-margin, ptB-margin, color="white", linewidth=3, linestyle=':')
    plt.axis("off")

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(out_folder+"/"+rename_grades(score_r)+"/"+name+"_R.png",
                bbox_inches="tight", dpi=150, format='png', pad_inches = 0)

    plt.close()


def save3imgs(out_name: str, img_slice: np.ndarray, vessels: np.ndarray,
                    carc_mask: np.ndarray, left_lines: list, right_lines: list,
                    show_mask: bool = True, crop: bool = False):
    """_summary_

    Args:
        out_folder (str): output folder
        img_slice (np.ndarray): slice of the scan (2D array)
        vessels (np.ndarray): mask of arteries
        carc_mask (np.ndarray): mask of tumour
        left_lines (list): left critical lines
        right_lines (list): right critical lines
        show_mask (bool, optional): visibility of mask. Defaults to True.
        crop (bool, optional): cropping to 194×194 px if True. Defaults to False.
    """

    out_name = out_name[:-4] # delete extension

    carc_mask[carc_mask==10] = 15
    carc_mask[carc_mask==11] = 19
    carc_mask[carc_mask==12] = 22
    carc_mask[carc_mask==13] = 24

    mask_only = 9 * np.ones_like(carc_mask, dtype=int)
    mask_only[carc_mask > 0] = 160
    mask_only[vessels > 0] = 240

    if crop:
        img_slice, h_margin, w_margin = crop194(img_slice)
        carc_mask, _, _ = crop194(carc_mask)
        mask_only, _, _ = crop194(mask_only)
        vessels, _, _ = crop194(vessels)
    else:
        h_margin, w_margin = 0, 0

    margin = np.array([h_margin, w_margin])
    
    # Overlays ----------------------------------------------------------

    plt.imshow(img_slice, cmap="gray")
    plt.imshow(carc_mask, alpha=0.5*(carc_mask>0).astype(float), vmin=0, vmax=36, cmap="gist_ncar")
    plt.imshow(vessels, alpha=0.75*vessels, vmin=0, vmax=1, cmap="OrRd")
    for line in left_lines:
        ptA = line.a[::-1]
        ptB = line.b[::-1]
        plt.axline(ptA-margin, ptB-margin, color="white", linewidth=1, linestyle=':')
    for line in right_lines:
        ptA = line.a[::-1]
        ptB = line.b[::-1]
        plt.axline(ptA-margin, ptB-margin, color="white", linewidth=1, linestyle=':')
    plt.axis("off")

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(out_name+"_overlay.png", bbox_inches="tight", dpi=150, format='png', pad_inches = 0)

    plt.close()

    # Mask --------------------------------------------------------------

    plt.imshow(mask_only, cmap="nipy_spectral", vmin=0, vmax=255)
    plt.axis("off")

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(out_name+"_mask.png", bbox_inches="tight", format='png', pad_inches = 0)

    plt.close()

    # Slice only --------------------------------------------------------

    plt.imshow(img_slice, cmap="gray")
    plt.axis("off")

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(out_name+"_slice.png", bbox_inches="tight", dpi=150, format='png', pad_inches = 0)

    plt.close()
