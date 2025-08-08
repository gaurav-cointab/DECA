import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from tqdm import tqdm

CONFIDENCE_THRESHOLD = 0.5
MATCH_THRESHOLD = 0.01
EYE_UPPER_DROP = np.array([37, 38, 43, 44], dtype=int)
MIN_POINTS_FLOOR = 20
MIN_POINTS_FRAC = 0.40


def preprocess_landmarks_coords_only(landmarks):
    return np.asarray(landmarks[:, :3], dtype=float)


def valid_landmarks3d(lmk1, lmk2,
                      conf_thr=CONFIDENCE_THRESHOLD,
                      drop_idx=EYE_UPPER_DROP,
                      min_floor=MIN_POINTS_FLOOR,
                      min_frac=MIN_POINTS_FRAC):
    coords1 = preprocess_landmarks_coords_only(lmk1)
    coords2 = preprocess_landmarks_coords_only(lmk2)
    conf1 = np.asarray(lmk1[:, 3], dtype=float)
    conf2 = np.asarray(lmk2[:, 3], dtype=float)
    n_pts = coords1.shape[0]
    all_idx = np.arange(n_pts)
    drop_idx = np.asarray(drop_idx, dtype=int)
    candidates = np.setdiff1d(all_idx, drop_idx, assume_unique=True)
    joint_ok = (conf1 >= conf_thr) & (conf2 >= conf_thr)
    keep_idx = candidates[joint_ok[candidates]]
    min_required = max(min_floor, int(np.ceil(min_frac * candidates.size)))
    if keep_idx.size < min_required:
        raise ValueError(
            f"Too few reliable landmarks: kept {keep_idx.size}, need ≥ {min_required} "
            f"(candidates={candidates.size}, floor={min_floor}, frac={min_frac:.0%})."
        )
    A = coords1[keep_idx]
    B = coords2[keep_idx]
    return A, B


def compare_landmarks3d(lmk1, lmk2,
                        conf_thr=CONFIDENCE_THRESHOLD,
                        drop_idx=EYE_UPPER_DROP,
                        min_floor=MIN_POINTS_FLOOR,
                        min_frac=MIN_POINTS_FRAC):
    X, Y = valid_landmarks3d(lmk1, lmk2, conf_thr, drop_idx, min_floor, min_frac)
    _, _, disparity = procrustes(X, Y)
    return disparity


def is_same_person(disparity, threshold=MATCH_THRESHOLD):
    return disparity < threshold


def visualize_landmarks(lmk1, lmk2, title="Landmark Overlay", fileName=None, labelA=None, labelB=None, colorA="red",
                        colorB="blue"):
    norm1, norm2 = valid_landmarks3d(lmk1, lmk2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(norm1[:, 0], norm1[:, 1], norm1[:, 2], label=labelA, alpha=0.7, color=colorA)
    ax.scatter(norm2[:, 0], norm2[:, 1], norm2[:, 2], label=labelB, alpha=0.7, color=colorB)
    ax.set_title(title)
    ax.legend()
    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()
    plt.close()


def main(landmarka_path, landmarkb_path, output):
    os.makedirs(output, exist_ok=True)
    lman = os.path.splitext(os.path.basename(landmarka_path))[0]
    lmbn = os.path.splitext(os.path.basename(landmarkb_path))[0]
    fileName = os.path.join(output, lmbn + ".jpg")
    landmarks3d_a = np.loadtxt(landmarka_path)
    landmarks3d_b = np.loadtxt(landmarkb_path)
    disparity = compare_landmarks3d(landmarks3d_a, landmarks3d_b)
    is_same = is_same_person(disparity)
    title = "Similarity Result: "
    if is_same:
        title = title + "Same"
    else:
        title = title + "Diff"
    title = title + f"({disparity:.5f})"
    visualize_landmarks(landmarks3d_a, landmarks3d_b, title=title, fileName=fileName, labelA=lman, labelB=lmbn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    parser.add_argument('-a', '--landmarka', type=str, help='path to the landmark a')
    parser.add_argument('-b', '--landmarkb', type=str, help='path to the landmark b')
    parser.add_argument('-f', '--landmarks', type=str, help='path to the landmarks folder')
    parser.add_argument('-o', '--output', type=str, help='path to the output folder')
    args = parser.parse_args()
    if args.landmarks is not None:
        lmn_files = sorted(glob.glob(os.path.join(args.landmarks, "**", "*kpt3d.txt")))
        code_files = sorted(glob.glob(os.path.join(args.landmarks, "**", "*codedict.json")))

        for i in tqdm(range(len(lmn_files))):
            for j in tqdm(range(len(lmn_files))):
                if j > i:
                    main(lmn_files[i], lmn_files[j], str(Path(lmn_files[i]).resolve().parent))
    else:
        main(args.landmarka, args.landmarkb, args.output)
