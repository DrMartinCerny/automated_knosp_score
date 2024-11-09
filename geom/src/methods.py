import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull
from skimage.measure import label
from skimage.morphology import flood_fill, binary_closing, remove_small_objects, disk

import os
os.environ["OMP_NUM_THREADS"] = '1' #to prevent warnings from parallel processing

from sklearn.cluster import KMeans
from src.objects import Vessel, Line


def rename_grades(grade: float) -> int:
    """Convert a float to a string representation of the score."""
    if grade < 3:
        return int(grade)
    elif grade == 3.0:
        print("warning: inspecific grade 3")
        return 3
    elif grade == 3.1:
        return 3
    elif grade == 3.2:
        return 4
    elif grade == 4:
        return 5
    else:
        print("warning: unknown grade")
        return 6


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Preprocessing of the mask for automatically generated masks.

    Args:
        mask (np.ndarray): 2D slice of a mask

    Returns:
        np.ndarray: smoothened mask
    """

    # assign to segmentation type based on label
    vessels = (mask == 2)
    carcinoma = (mask == 1) + (mask == 3)

    # morphological closing for carcinoma segmentation
    footprint = disk(3, decomposition="sequence")
    carcinoma = binary_closing(carcinoma, footprint=footprint)

    # delete small disconnected parts
    carcinoma = remove_small_objects(carcinoma>0, min_size=0.05*np.sum(carcinoma))

    # compute size of small object
    lab_vessels, n_vessels = label(vessels.astype(np.uint8), return_num=True)

    if n_vessels > 0:
        areas = np.zeros(n_vessels, dtype=int)
        for i in range(n_vessels):
            areas[i] = np.sum(lab_vessels == (i+1))
        areas = np.sort(areas, axis=None)[::-1]
        size_th = np.mean(areas[:4]) / 6 # based on 4 largest vessels

        # remove small objects from vessels
        vessels = remove_small_objects(vessels>0, min_size=size_th)

    # copy objects to output
    out = np.zeros_like(mask, dtype=mask.dtype)
    out[carcinoma>0] = 1
    out[vessels>0] = 2

    return out  


def classify_vessels(vessels: np.ndarray):
    
    """Separate the individual arteries.

    Args:
        vessels (np.ndarray): mask of arteries

    Returns:
        list of 4 Vessels objects
    """
    
    # there must be at least 4 px
    pts = np.argwhere(vessels>0)
    if pts.shape[0] < 4:
        return None
            
    # use K means to find 4 clusters
    kmeans = KMeans(n_clusters=4)
    labels = kmeans.fit_predict(pts)

    coms = np.zeros((4,3)) # centre of mass
    for i in range(4):
        coms[i,2] = i
        coms[i,0:2] = np.mean(pts[labels==i,:], axis=0)

    # split left and right side
    middle = np.mean(coms[:, 0:2], axis=0)
    left = coms[coms[:,1]<=middle[1]]
    right = coms[coms[:,1]>middle[1]]

    if not (left.shape[0] == 2 and right.shape[0] == 2):
        print("\nWARNING: Cannot distinguish left and right vessels.")
        return None

    # distinguish left/right, top/bottom
    LT_idx = left[0,2] if left[0,0] < left[1,0] else left[1,2]
    LB_idx = left[0,2] if left[0,0] >= left[1,0] else left[1,2]
    RT_idx = right[0,2] if right[0,0] < right[1,0] else right[1,2]
    RB_idx = right[0,2] if right[0,0] >= right[1,0] else right[1,2]

    out = []
    # return in expected order
    for idx in [LT_idx, LB_idx, RT_idx, RB_idx]:
        pts_i = pts[labels==idx]
        x_i, y_i = pts_i.T
        bin_mask = np.zeros_like(vessels)
        bin_mask[x_i, y_i] = 1
        out.append(Vessel(pts_i, bin_mask, coms[coms[:,2]==idx,0:2].squeeze()))
    
    return out


def is_pt_out(pts: np.ndarray, line: Line, th: float = 0.5):
    out = (pts - line.b) @ line.v
    div = norm(line.v)
    if div == 0:
        div = 10e-5
    norm_v = 1 / div
    tolerance = norm_v * np.abs(out)

    return (out < 0) * (tolerance > th) # half-pixel distance tolerated


def find_lines(LT: Vessel, LB: Vessel):
    """Find connecting lines between vessels.
    
    Input (Vessel class) in this order: 
    - left top, left, bottom
    OR
    - right bottom, right top
    """

    mid = Line(LT.com, LB.com)

    all_pts = np.vstack((LT.pts, LB.pts))

    if len(np.unique(all_pts[:,0])) == 1 or len(np.unique(all_pts[:,1])) == 1:
        print("\nWARNING: All lines are the same - both vessels are only 1 px.")
        return [mid, mid, mid]

    both = np.vstack((LT.pts, LB.pts))
    hull = ConvexHull(both)
    vertices = hull.points[hull.vertices]
    vertices = np.vstack((vertices, [vertices[0]])).astype(int)
    vert_id = LT.mask[vertices[:,0], vertices[:,1]] + 2*LB.mask[vertices[:,0], vertices[:,1]]
    diff = np.diff(vert_id)

    idx1 = np.argwhere(diff==-1)
    idx2 = np.argwhere(diff==1)

    if is_pt_out(vertices[idx1], mid):
        lat = Line(vertices[idx1+1].squeeze(), vertices[idx1].squeeze())
        med = Line(vertices[idx2].squeeze(), vertices[idx2+1].squeeze())
    else:
        med = Line(vertices[idx1+1].squeeze(), vertices[idx1].squeeze())
        lat = Line(vertices[idx2].squeeze(), vertices[idx2+1].squeeze())

    return [lat, mid, med]


def classify_carcinoma(carcinoma: np.ndarray, left_lines: list, right_lines: list, th: float = 0.5):
    
    """Classify the tumour.

    Args:
        carcinoma (np.ndarray): tumour's mask
        left_lines (list): left critical lines
        right_lines (list): right critical lines
        th (float, optional): threshold for surpassing a line. Defaults to 0.5.

    Returns:
        mask and grade of the tumour
    """
    
    # empty mask
    if np.sum(carcinoma==1) == 0:
        return 0, 0, (carcinoma==0).astype(carcinoma.dtype)
    
    carc_pts = np.argwhere(carcinoma==1)
    
    l_lat, l_mid, l_med = left_lines
    r_lat, r_mid, r_med = right_lines

    LT_com, LB_com = l_mid.a, l_mid.b
    RB_com, RT_com = r_mid.a, r_mid.b

    # initialize lines' vectors
    LT_to_LB = (LT_com - LB_com) @ (LT_com - LB_com)
    RT_to_RB = (RT_com - RB_com) @ (RT_com - RB_com)

    # Left side
    gr1_l = is_pt_out(carc_pts, l_med, th)
    gr2_l = is_pt_out(carc_pts, l_mid, th)
    gr3_l = is_pt_out(carc_pts, l_lat, th)

    mask_l = 10*np.copy(carcinoma)
    mask_l[carc_pts[gr1_l,0], carc_pts[gr1_l,1]] = 11
    mask_l[carc_pts[gr2_l,0], carc_pts[gr2_l,1]] = 12
    mask_l[carc_pts[gr3_l,0], carc_pts[gr3_l,1]] = 13

    # decrease classification to 0 for pixels above superior vessel
    mask_l_1or2 = np.zeros_like(mask_l)
    mask_l_1or2[mask_l == 11] = 1
    mask_l_1or2[mask_l == 12] = 1
    labeled_l, nlab_l = label(mask_l_1or2, return_num=True)
    for i in range(nlab_l):
        pts_i = np.argwhere(labeled_l == i+1)
        com_i = np.mean(pts_i, axis=0)
        if (com_i - LB_com) @ (LT_com - LB_com) >= LT_to_LB:
            mask_l_1or2[labeled_l == i+1] = -1
    mask_l[mask_l_1or2 == -1] = 10

    score_l = mask_l.max()-10

    # distinguish grade 3 subclasses and grade 4
    if score_l == 3:
        com_l_inf = l_mid.a if l_mid.a[0] > l_mid.b[0] else l_mid.b
        com_l_sup = l_mid.a if l_mid.a[0] < l_mid.b[0] else l_mid.b
        new_l_inf, mask_l = classify_grade4(mask_l, com_l_inf, com_l_sup)
        if new_l_inf == 3.0: # decrease grade to 0 for pixels above superior vessel
            mask_l[mask_l == 13] = 10
            score_l = mask_l.max()-10
        else:
            score_l = new_l_inf

    # Right side
    gr1_r = is_pt_out(carc_pts, r_med, th)
    gr2_r = is_pt_out(carc_pts, r_mid, th)
    gr3_r = is_pt_out(carc_pts, r_lat, th)

    mask_r = 10*np.copy(carcinoma)
    mask_r[carc_pts[gr1_r,0], carc_pts[gr1_r,1]] = 11
    mask_r[carc_pts[gr2_r,0], carc_pts[gr2_r,1]] = 12
    mask_r[carc_pts[gr3_r,0], carc_pts[gr3_r,1]] = 13

    # decrease classification to 0 for pixels above superior vessel
    mask_r_1or2 = np.zeros_like(mask_r)
    mask_r_1or2[mask_r == 11] = 1
    mask_r_1or2[mask_r == 12] = 1
    labeled_r, nlab_r = label(mask_r_1or2, return_num=True)
    for i in range(nlab_r):
        pts_i = np.argwhere(labeled_r == i+1)
        com_i = np.mean(pts_i, axis=0)
        if (com_i - RB_com) @ (RT_com - RB_com) >= RT_to_RB:
            mask_r_1or2[labeled_r == i+1] = -1
    mask_r[mask_r_1or2 == -1] = 10

    score_r = mask_r.max()-10

    # distinguish grade 3 subclasses and grade 4
    if score_r == 3:
        com_r_inf = r_mid.a if r_mid.a[0] > r_mid.b[0] else r_mid.b
        com_r_sup = r_mid.a if r_mid.a[0] < r_mid.b[0] else r_mid.b
        new_r_inf, mask_r = classify_grade4(mask_r, com_r_inf, com_r_sup)
        if new_r_inf == 3.0: # decrease grade to 0 for pixels above superior vessel
            mask_r[mask_r == 13] = 10
            score_r = mask_r.max()-10
        else:
            score_r = new_r_inf

    # Total
    carc_mask = np.where(mask_l == 10, mask_r, mask_l)

    score_l = rename_grades(np.maximum(score_l, 0))
    score_r = rename_grades(np.maximum(score_r, 0))

    return score_l, score_r, carc_mask


def is_grade4(mask: np.ndarray, com_i: np.ndarray):
    
    """Return True for grade 4 if encapsulation detected.

    Args:
        mask (np.ndarray): mask of a tumour
        com_i (np.ndarray): centre of mass of an artery

    Returns:
        bool (True for grade 4)
    """

    flooded = flood_fill(mask, seed_point=(0,0), new_value=1)
    holes, num = label(flooded==0, return_num=True)

    # for every hole in the tumour
    for hole_i in range(num):
        pixels_i = np.argwhere(holes==(hole_i+1))
        # if the centre of mass of the artery lies in any hole in the tumour
        if np.any(np.equal(pixels_i, np.round(com_i)).all(axis=1)):
            return True
    
    return False


def classify_grade4(mask: np.ndarray, com_i: np.ndarray, com_s: np.ndarray):
    """Distinguish grade 4 and grades 3A and 3B.

    Args:
        mask (np.ndarray): tumour's mask
        com_i (np.ndarray): inferior artery's centre of mass
        com_s (np.ndarray): superior artery's centre of mass

    Returns:
        grade and mask of the tumour
    """

    # first accept / reject grade 4 (easier)
    if is_grade4(mask, com_i):
        return 4, mask
    if is_grade4(mask, com_s):
        return 4, mask

    # then investigate the rest
    dist_coms = (com_i - com_s) @ (com_i - com_s)
    mask3 = (mask==13).astype(int)
    labeled, n = label(mask3, return_num=True)
    sup = False
    sup_count = 0
    inf = False
    inf_count = 0

    for i in range(1, n+1):
        sup_i = False
        inf_i = False
        pxs = np.argwhere(labeled==i)
        for px in pxs:
            y, x = px
            if 12 in mask[y-1:y+2, x-1:x+2]:
                # if point is closer to superior than inferior to superior
                if (px - com_s) @ (com_i - com_s) < dist_coms:
                    sup_i = True
                    sup_count += 1
                else:
                    inf_i = True
                    inf_count += 1
        if sup_i and inf_i: # touch bellow and under inferior
            sup = sup or sup_i
            inf = inf or inf_i
        elif not sup_i and not inf_i:
            mask[labeled==i] = 10
        else:
            sup = sup or sup_i
            inf = inf or inf_i
    
    if sup and inf:
        if sup_count > inf_count:
            return 3.1, mask
        else:
            return 3.2, mask
    elif sup:
        return 3.1, mask
    elif inf:
        return 3.2, mask
    else:
        return 3.0, mask
