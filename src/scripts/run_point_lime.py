#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import logging

import numpy as np
import torch
import open3d as o3d

from src.lime import lime_3d_remove
from src.models.pointnet.pointnet import PointNetGSSystem


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SHAPE_NAMES = [
    line.rstrip()
    for line in open(
        os.path.join(BASE_DIR, "data/shape_names.txt")
    )
]


def take_first(elem):
    return elem[0]


def parse_args():
    parser = argparse.ArgumentParser("PointNet + LIME")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ścieżka do pliku wejściowego (.ply/.txt/.npy)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ścieżka do checkpointa modelu (.ckpt/.pth)"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="ID GPU"
    )

    parser.add_argument(
        "--num_point",
        type=int,
        default=1024,
        help="Liczba punktów"
    )

    parser.add_argument(
        "--num_votes",
        type=int,
        default=1,
        help="Liczba głosowań"
    )

    return parser.parse_args()


def sampling(points, sample_size):
    num_p = points.shape[0]
    np.random.seed(1)

    sampled_index = np.random.choice(
        range(num_p),
        size=sample_size,
        replace=False
    )

    return points[sampled_index]


def load_point_cloud(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    if filepath.endswith(".npy"):
        points = np.load(filepath)

    elif filepath.endswith(".txt"):
        points = np.loadtxt(filepath, delimiter=",")

    elif filepath.endswith(".ply"):
        pc = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pc.points)

    else:
        raise ValueError(
            "Obsługiwane formaty: .ply, .txt, .npy"
        )

    return points


def load_model(checkpoint_path, num_classes=40):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Nie znaleziono checkpointa: {checkpoint_path}"
        )

    classifier = PointNetGSSystem(
        num_classes=num_classes,
        in_channels_total=11
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu")
    )

    if "state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["state_dict"])

    elif "model_state_dict" in checkpoint:
        classifier.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:
        classifier.load_state_dict(checkpoint)

    classifier = classifier.model
    classifier.eval()

    return classifier


def prepare_input(points, num_points=1024):
    if points.shape[1] > 3:
        points = points[:, :3]

    if points.shape[0] > num_points:
        points = sampling(points, num_points)

    return points


def test(model, filepath, num_points=1024):
    points = load_point_cloud(filepath)
    points = prepare_input(points, num_points)

    B = 1
    N = points.shape[0]

    xyz = torch.from_numpy(points).float()
    xyz = xyz.unsqueeze(0)  # [1, N, 3]

    gs_extra = torch.zeros(B, 8, N)
    mask = torch.ones(B, N, dtype=torch.bool)

    pred, _, _ = model(xyz, gs_extra, mask)

    pred_choice = pred.data.max(1)[1]

    print("Prediction logits:\n", pred[0])
    print(
        "Predict Result:",
        pred_choice.item(),
        SHAPE_NAMES[pred_choice.item()]
    )

    return xyz, pred_choice, pred


def gen_pc_data(
    ori_data,
    segments,
    explain,
    label,
    filename
):
    os.makedirs("visu", exist_ok=True)

    basic_path = "visu/"
    color = np.zeros([ori_data.shape[0], 3])

    max_contri = 0
    min_contri = 0

    for k in explain[label]:
        if k[1] > 0 and k[1] > max_contri:
            max_contri = k[1]

        elif k[1] < 0 and k[1] < min_contri:
            min_contri = k[1]

    positive_color_scale = (
        1 / max_contri if max_contri > 0 else 0
    )

    negative_color_scale = (
        1 / min_contri if min_contri < 0 else 0
    )

    ex_sorted = sorted(
        explain[label],
        key=take_first,
        reverse=False
    )

    for i in range(segments.shape[0]):
        if ex_sorted[segments[i]][1] > 0:
            color[i][0] = (
                ex_sorted[segments[i]][1]
                * positive_color_scale
            )

        elif ex_sorted[segments[i]][1] < 0:
            color[i][2] = (
                ex_sorted[segments[i]][1]
                * negative_color_scale
            )

    pc_colored = np.concatenate(
        (ori_data, color),
        axis=1
    )

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(
        pc_colored[:, 0:3]
    )
    pc.colors = o3d.utility.Vector3dVector(
        pc_colored[:, 3:6]
    )

    output_path = os.path.join(
        basic_path,
        filename
    )

    o3d.io.write_point_cloud(output_path, pc)

    print(
        f"Wygenerowano chmurę punktów: {output_path}"
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging.basicConfig(level=logging.INFO)

    model = load_model(args.checkpoint)

    with torch.no_grad():
        points, pred, logits = test(
            model=model,
            filepath=args.input,
            num_points=args.num_point
        )

    label = pred.detach().numpy()[0]

    points_for_exp = np.asarray(
        points.squeeze(0)
    )

    def predict_fn(input_points):
        if isinstance(input_points, np.ndarray):
            input_points = torch.from_numpy(
                input_points
            ).float()

        if len(input_points.shape) == 2:
            input_points = input_points.unsqueeze(0)

        B, N, _ = input_points.shape

        gs_extra = torch.zeros(B, 8, N)
        mask = torch.ones(
            B,
            N,
            dtype=torch.bool
        )

        pred, _, _ = model(
            input_points,
            gs_extra,
            mask
        )

        return pred.detach().numpy()

    explainer = lime_3d_remove.LimeImageExplainer(
        random_state=0
    )

    start = time.time()

    explanation = explainer.explain_instance(
        points_for_exp,
        predict_fn,
        top_labels=5,
        num_features=20,
        num_samples=10,
        random_seed=0
    )

    print(
        "LIME execution time:",
        time.time() - start,
        "s"
    )

    gen_pc_data(
        points_for_exp,
        explanation.segments,
        explanation.local_exp,
        label,
        "test_lime.ply"
    )

    return explanation


if __name__ == "__main__":
    args = parse_args()
    main(args)