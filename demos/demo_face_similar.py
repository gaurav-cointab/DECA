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


def preprocess_landmarks(landmarks):
    coords = landmarks[:, :3]
    conf = landmarks[:, 3]

    mask = conf >= CONFIDENCE_THRESHOLD
    filtered_coords = coords[mask]

    centroid = np.mean(filtered_coords, axis=0)
    normalized = filtered_coords - centroid

    return normalized


def compare_landmarks3d(lmk1, lmk2):
    norm1 = preprocess_landmarks(lmk1)
    norm2 = preprocess_landmarks(lmk2)
    n = min(len(norm1), len(norm2))
    norm1 = norm1[:n]
    norm2 = norm2[:n]
    _, _, disparity = procrustes(norm1, norm2)
    return disparity


def is_same_person(disparity, threshold=MATCH_THRESHOLD):
    return disparity < threshold


def visualize_landmarks(lmk1, lmk2, title="Landmark Overlay", fileName=None, labelA=None, labelB=None, colorA="red",
                        colorB="blue"):
    norm1 = preprocess_landmarks(lmk1)
    norm2 = preprocess_landmarks(lmk2)
    n = min(len(norm1), len(norm2))
    norm1, norm2 = norm1[:n], norm2[:n]
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
    parser.add_argument('-a', '--landmarka', type=str,
                        help='path to the landmark a')
    parser.add_argument('-b', '--landmarkb', type=str,
                        help='path to the landmark b')
    parser.add_argument('-f', '--landmarks', type=str,
                        help='path to the landmarks folder')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='path to the output folder')
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
