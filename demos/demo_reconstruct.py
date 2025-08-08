# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


def rotvec_to_euler_deg(rotvec: torch.Tensor):
    rotvec = rotvec.float()
    theta = torch.linalg.norm(rotvec)
    if theta < 1e-8:
        return 0.0, 0.0, 0.0

    axis = rotvec / theta
    K = torch.tensor(
        [[0.0, -axis[2].item(), axis[1].item()],
         [axis[2].item(), 0.0, -axis[0].item()],
         [-axis[1].item(), axis[0].item(), 0.0]],
        dtype=torch.float32
    )
    I = torch.eye(3, dtype=torch.float32)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    r11, r21, r31 = R[0, 0].item(), R[1, 0].item(), R[2, 0].item()
    r32, r33 = R[2, 1].item(), R[2, 2].item()

    yaw = math.degrees(math.atan2(r21, r11))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r31))))
    roll = math.degrees(math.atan2(r32, r33))
    return yaw, pitch, roll


def topk_by_abs(x: torch.Tensor, k: int):
    if x.numel() == 0:
        return []
    k = min(k, x.numel())
    vals = x.flatten()
    idx = torch.topk(vals.abs(), k).indices.tolist()
    out = []
    for i in idx:
        v = vals[i].item()
        out.append({"index": int(i), "value": v, "abs_value": abs(v)})
    out.sort(key=lambda d: (-d["abs_value"], d["index"]))
    return out


def save_codedict_human_readable(
        codedict: dict,
        path: str,
        *,
        topk_shape: int = 10,
        round_ndigits: int = 4
):
    def r(x):
        return round(float(x), round_ndigits)

    # Select first item if batched
    def first_row(t):
        if t is None:
            return None
        if t.ndim >= 2:
            return t[0].detach().cpu()
        return t.detach().cpu()

    # Pull components safely
    pose = first_row(codedict.get("pose"))
    shape = first_row(codedict.get("shape"))

    # Pose block
    pose_block = None
    if pose is not None and pose.numel() >= 3:
        rotvec = pose[:3]
        trans = pose[3:] if pose.numel() >= 6 else torch.tensor([])
        yaw, pitch, roll = rotvec_to_euler_deg(rotvec)
        pose_block = {
            "rotation_axis_angle": [r(v) for v in rotvec.tolist()],
            "yaw_pitch_roll_deg": [r(yaw), r(pitch), r(roll)],
            "translation": [r(v) for v in trans.tolist()] if trans.numel() else None
        }

    # Shape summary (top-k by |value|)
    shape_block = None
    if shape is not None and shape.numel() > 0:
        topk = topk_by_abs(shape, topk_shape)
        shape_block = {
            "top_k_by_abs": [{**d, "value": r(d["value"]), "abs_value": r(d["abs_value"])} for d in topk],
            "l2_norm": r(float(torch.linalg.norm(shape).item())),
            "mean": r(float(shape.mean().item())),
            "std": r(float(shape.std(unbiased=False).item()))
        }

    # Assemble final JSON
    out = {
        "pose": pose_block,
        "shape_summary": shape_block,
    }

    # Optional: drop None fields to keep it tidy
    out = {k: v for k, v in out.items() if v is not None}

    # Round nested floats for readability (already rounded; ensure clean printing)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    global args
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector,
                                 sample_step=args.sample_step, device=args.device)

    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)

        with torch.no_grad():
            codedict = deca.encode(images)
            torch.save(codedict, os.path.join(savefolder, name, name + '_codedict_orig.txt'))
            save_codedict_human_readable(codedict, os.path.join(savefolder, name, name + '_codedict_orig.json'))
            opdict, visdict = deca.decode(codedict)
            if args.saveKpt:
                np.savetxt(os.path.join(savefolder, name, name + '_kpt2d_orig.txt'),
                           opdict['landmarks2d'][0].cpu().numpy())
                np.savetxt(os.path.join(savefolder, name, name + '_kpt3d_orig.txt'),
                           opdict['landmarks3d'][0].cpu().numpy())
            if args.neutral:
                codedict['exp'] = torch.zeros_like(codedict['exp'])
                codedict['pose'] = torch.zeros_like(codedict['pose'])
                torch.save(codedict, os.path.join(savefolder, name, name + '_codedict_neutral.txt'))
                save_codedict_human_readable(codedict, os.path.join(savefolder, name, name + '_codedict_neutral.json'))
                if args.saveKpt:
                    np.savetxt(os.path.join(savefolder, name, name + '_kpt2d_neutral.txt'),
                               opdict['landmarks2d'][0].cpu().numpy())
                    np.savetxt(os.path.join(savefolder, name, name + '_kpt3d_neutral.txt'),
                               opdict['landmarks3d'][0].cpu().numpy())
            opdict, visdict = deca.decode(codedict)
            if args.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1, 2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)
                orig_visdict['inputs'] = original_image

        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
            if args.render_orig:
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images',
                             'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'),
                            util.tensor2image(visdict[vis_name][0]))
                if args.render_orig:
                    cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name + '.jpg'),
                                util.tensor2image(orig_visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--neutral', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to convert the images neutral and then render')
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    args = parser.parse_args()
    main()
