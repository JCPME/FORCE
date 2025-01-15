import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import random

"""
Download the experimental setup and the suturing dataset and rename the Suturing folder in the Experimental_setup folder to Suturing_meta.
Then extract the Suturing data and copy it into the Experimental_setup folder.
Run this script from this folder to generate the jigsawsdataset.pkl

Download the Experimental_setup folder from the following link:
https://www.kaggle.com/datasets/aditya9790/jigsaws-skill-assessment-working-set

Download the Suturing folder from:
https://www.cs.jhu.edu/~los/jigsaws/info.php

This is the file structure

Experimental_setup (Download from kaggle)
- Suturing (Copied in, from link above)
- Suturing_meta (Renamed Suturing folder from the Experimental_setup folder)
- extraction.py

"""




################################################################################
# 1) Helper: read all train/test lines, parse them, and produce a list of steps
################################################################################

def parse_train_test_line(line):
    """
    Example line:
      'Suturing_B001_000080_005635.txt           13'
    Splits into:
      filename_part = 'Suturing_B001_000080_005635.txt'
      label_part    = '13'
    Then parse filename_part => base_name='Suturing_B001', start=80, end=5635
    Returns: (base_name, start_frame, end_frame, label)
    """
    line = line.strip()
    if not line:
        return None
    
    parts = line.split()
    if len(parts) < 2:
        return None

    filename_part = parts[0].strip()
    label_str = parts[-1].strip()
    
    try:
        label = int(label_str)
    except ValueError:
        return None

    name_no_ext = os.path.splitext(filename_part)[0]  
    splitted = name_no_ext.split('_')
    if len(splitted) < 4:
        return None
    
    tool = splitted[0]               # e.g. "Suturing"
    participant_run = splitted[1]    # e.g. "B001"
    start_str = splitted[2]          # e.g. "000080"
    end_str   = splitted[3]          # e.g. "005635"

    try:
        start_frame = int(start_str)
        end_frame   = int(end_str)
    except ValueError:
        return None


    base_name = f"{tool}_{participant_run}"
    
    return (base_name, start_frame, end_frame, label)

def collect_steps_from_out_folders(base_folder):
    """
    Looks in each *_Out folder under `base_folder` for 'train.txt' and 'test.txt'.
    Parses each line to produce a list of (base_name, start, end, label, set_type)
    where set_type is "test" if the file is test.txt and "train" otherwise.
    """
    all_steps = []
    out_folders = glob.glob(os.path.join(base_folder, "*_Out", "itr_1"))
    for folder_path in out_folders:
        folder_name = os.path.basename(folder_path)
        print(f"[DEBUG] Checking folder: {folder_name}")

        for txt_name in ["train.txt", "test.txt"]:
            txt_path = os.path.join(folder_path, txt_name)
            if not os.path.isfile(txt_path):
                continue

            set_type = "test" if "test.txt" in txt_path.lower() else "train"

            print(f"   -> Reading file: {txt_path} (set: {set_type})")
            with open(txt_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parsed = parse_train_test_line(line)
                if parsed:
                    all_steps.append(parsed + (set_type,))
    
    return all_steps

################################################################################
# 2) Main Script: uses the steps to slice the kinematics data
################################################################################

print("=== Starting script ===")

columns = [
    "participant_num",      # e.g., 1, 2, 3, 4, ...
    "tool",                 # 'master' or 'slave'
    "case",                 # e.g., 'Suturing_B001'
    "translation_array",    # Nx3 translations
    "rotation_array",       # Nx4 quaternions [w, x, y, z]
    "timestamp_array",      # Nx1 timestamps
    "avg_grs_score",
    "total_procedure_time",
    "label",                # from train/test => skill label
    "start_frame",          # from train/test
    "end_frame",            # from train/test
    "set"                   # new column: "test" or "train"
]
print(f"[DEBUG] Columns for final DataFrame: {columns}\n")

participant_map = {
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'I': 6,
}

meta_path = './Suturing/meta_file_Suturing.txt'
grs_map = {}
print(f"[DEBUG] Reading meta file: {meta_path}\n")

with open(meta_path, 'r') as f:
    lines = f.readlines()
    print(f"[DEBUG] Read {len(lines)} lines from meta file.")
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        recording_name = parts[0].strip() 
        try:
            avg_grs_score = float(parts[2])  
        except ValueError:
            continue
        grs_map[recording_name] = avg_grs_score

print("[DEBUG] GRS map loaded:", grs_map, "\n")

base_out_folder = r".\Suturing_meta\unBalanced\SkillDetection\UserOut"
steps = collect_steps_from_out_folders(base_out_folder)
print(f"[DEBUG] Found {len(steps)} total steps in train/test files.\n")

kinematics_folder = './Suturing/kinematics/AllGestures/'
print(f"[DEBUG] Kinematics folder: {kinematics_folder}\n")

rows_list = []

###############################################################################
# For each step, we:
#  - parse base_name => "Suturing_B001"
#  - load "Suturing_B001.txt" from the kinematics folder
#  - slice data from start_frame .. end_frame
#  - build 2 rows: one for MASTER and one for SLAVE
#  - add a new column "set" with "test" or "train"
###############################################################################
for step in steps:
    base_name, start_f, end_f, label, set_type = step
    
    kine_filename = f"{base_name}.txt"
    kine_path = os.path.join(kinematics_folder, kine_filename)
    
    if not os.path.isfile(kine_path):
        print(f"[WARNING] Kinematics file not found: {kine_path}. Skipping.")
        continue
    
    splitted = base_name.split('_', 1)
    if len(splitted) < 2:
        print(f"[WARNING] Could not parse participant from {base_name}")
        continue
    participant_with_run = splitted[1]  
    letter = participant_with_run[0]    
    participant_num = participant_map.get(letter, 999)

    case = base_name

    try:
        data = np.loadtxt(kine_path)
    except Exception as e:
        print(f"[ERROR] Could not load {kine_path}: {e}")
        continue

    N, num_cols = data.shape
    if end_f >= N:
        print(f"[WARNING] end_frame={end_f} is beyond data length={N}, adjusting.")
        end_f = N - 1
    if start_f >= N:
        print(f"[WARNING] start_frame={start_f} is beyond data length={N}, skipping step.")
        continue

    step_data = data[start_f : end_f+1]
    nrows = step_data.shape[0]
    if nrows < 1:
        print(f"[WARNING] 0-length slice for {base_name}, frames {start_f}-{end_f}. Skipping.")
        continue

    master_xyz  = step_data[:, 0:3]
    master_rmat = step_data[:, 3:12].reshape(nrows, 3, 3)
    master_quat = R.from_matrix(master_rmat).as_quat()
    master_quat = master_quat[:, [3, 0, 1, 2]]  # reorder to [w, x, y, z]

    slave_xyz  = step_data[:, 38:41]
    slave_rmat = step_data[:, 41:50].reshape(nrows, 3, 3)
    slave_quat = R.from_matrix(slave_rmat).as_quat()
    slave_quat = slave_quat[:, [3, 0, 1, 2]]

    timestamps = np.arange(nrows) / 32.0

    avg_grs_score = grs_map.get(base_name, float('nan'))
    total_procedure_time = nrows / 32.0

    row_master = {
        "participant_num": participant_num,
        "tool": "master",
        "case": case,
        "translation_array": master_xyz,
        "rotation_array": master_quat,
        "timestamp_array": timestamps,
        "avg_grs_score": avg_grs_score,
        "total_procedure_time": total_procedure_time,
        "label": label,
        "start_frame": start_f,
        "end_frame": end_f,
        "set": set_type
    }
    rows_list.append(row_master)

    row_slave = {
        "participant_num": participant_num,
        "tool": "slave",
        "case": case,
        "translation_array": slave_xyz,
        "rotation_array": slave_quat,
        "timestamp_array": timestamps,
        "avg_grs_score": avg_grs_score,
        "total_procedure_time": total_procedure_time,
        "label": label,
        "start_frame": start_f,
        "end_frame": end_f,
        "set": set_type
    }
    rows_list.append(row_slave)

df_final = pd.DataFrame(rows_list, columns=columns)

print("\n=== Finished processing all steps ===")
print(f"[DEBUG] Final DataFrame shape: {df_final.shape}")
print(df_final.head(10))

df_final.to_pickle('jigsawsdataset.pkl')
