import os
import sys

from datetime import datetime
from src.pipeline import process_scan, export_outputs


def process_all_folders(data_folder: str, out_folder: str,
                        mode: str = "auto", plots: bool = False,
                        subfolders: bool = False, th_extent: float = 0.5):
    """Run the prediction for every available subject.

    Args:
        data_folder (str): folder with input data
        out_folder (str): destination folder for outputs
        mode (str, optional): "auto" or "manual. Defaults to "auto".
        plots (bool, optional): plots are stored if True. Defaults to False.
        subfolders (bool, optional): annotations saved in patients' subfolders if True.
                                     Defaults to False.
        th_extent (float, optional): threshold for distance from line. Defaults to 0.5.
    """
    
    start_time = datetime.now()

    subjects = next(os.walk(data_folder))[1]
    df_all = []
    total_subjects = len(subjects)

    # iterate over patients
    for i, subject in enumerate(subjects):
        print("Processing data from folder: ", subject, ".", sep='', end=' ')
        in_path = os.path.join(data_folder, subject)
        out_path = os.path.join(out_folder, subject)

        df_subj = process_scan(subject, in_path, out_path,
                               mode, plots, subfolders, th_extent)

        df_all.append(df_subj)
        print(f"Done. ({i+1}/{total_subjects})")
    export_outputs(df_all, out_folder)

    end_time = datetime.now()

    print(f"Processed {len(subjects)} patient scan(s) in {end_time-start_time}.")
    print("Finished.")


if __name__ == "__main__":
    n_inputs = len(sys.argv)

    # usage: python main.py "/path/to/src/folder" "/path/to/dest/folder" "auto/manual" 0/1 0/1 float

    data = sys.argv[1] if n_inputs > 1 else "../knosp_large/test"
    out = sys.argv[2] if n_inputs > 2 else "../knosp_large/out-test"
    mode = sys.argv[3] if n_inputs > 3 else "auto"
    plots = (sys.argv[4]=="1") if n_inputs > 4 else False
    subfolders = (sys.argv[5]=="1") if n_inputs > 5 else False
    th_extent = float(sys.argv[6]) if n_inputs > 6 else 0.5

    print("\nInitializing computation with parameters:")
    print(f"- input path: {data}")
    print(f"- output path: {out}")
    print(f"- segmentation mode: {'automatic (masks predicted by CNN)' if mode == 'auto' else 'manual (manually created masks)'}")
    print(f"- saving visualizations: {'on' if plots else 'off'}")
    print(f"- exporting results in subfolders: {'on' if subfolders else 'off'}")
    print(f"- threshold of extent behind line: {th_extent} px\n")

    process_all_folders(data, out, mode, plots, subfolders, th_extent)
