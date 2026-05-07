#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import os
import math
import random
import struct
import argparse

import numpy as np
import matplotlib.cm as cm
import open3d as o3d

from tqdm import tqdm

from src.models.pointnet.pointnet import PointNetGSSystem


def normalize_points(points):
    pc = points - np.mean(points, axis=0, keepdims=True)
    denom = np.max(np.linalg.norm(pc, axis=1))

    if denom > 0:
        pc = pc / denom

    return pc.astype(np.float32)


def random_downsample(points, k, seed=0):
    rng = np.random.default_rng(seed)

    idx = rng.choice(
        len(points),
        size=min(k, len(points)),
        replace=False
    )

    return points[idx], idx


def load_point_cloud(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Input file not found: {filepath}"
        )

    if filepath.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(
            pcd.points,
            dtype=np.float32
        )

    elif filepath.endswith(".npy"):
        points = np.load(filepath).astype(
            np.float32
        )

    elif filepath.endswith(".txt"):
        points = np.loadtxt(
            filepath,
            delimiter=","
        ).astype(np.float32)

    else:
        raise ValueError(
            "Supported formats: .ply, .txt, .npy"
        )

    return points


def load_model(checkpoint_path, num_classes=40):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
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
        classifier.load_state_dict(
            checkpoint["state_dict"]
        )

    elif "model_state_dict" in checkpoint:
        classifier.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:
        classifier.load_state_dict(checkpoint)

    classifier = classifier.model
    classifier.eval()

    return classifier


def forward_model(model, point_cloud, device):
    if isinstance(point_cloud, np.ndarray):
        x = torch.from_numpy(
            point_cloud
        ).float()
    else:
        x = point_cloud.float()

    if x.ndim == 2:
        x = x.unsqueeze(0)

    B, N, _ = x.shape

    gs_extra = torch.zeros(
        B, 8, N,
        device=device
    )

    mask = torch.ones(
        B, N,
        dtype=torch.bool,
        device=device
    )

    x = x.to(device)

    logits, _, _ = model(
        x,
        gs_extra,
        mask
    )

    return logits


def predict(
    model,
    point_cloud,
    class_idx,
    device,
    empty_baseline=0.0
):
    if len(point_cloud) == 0:
        return float(empty_baseline)

    with torch.no_grad():
        logits = forward_model(
            model,
            point_cloud,
            device
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        return probs[
            0,
            int(class_idx)
        ].item()


def get_predicted_class(
    point_cloud,
    model,
    device
):
    with torch.no_grad():
        logits = forward_model(
            model,
            point_cloud,
            device
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        return int(
            torch.argmax(
                probs,
                dim=1
            ).item()
        )


def generate_sampled_coalitions(
    num_points,
    M,
    stratified=True
):
    coalition_dict = {}

    for target in tqdm(
        range(num_points),
        desc="Generating coalitions"
    ):
        others = [
            i for i in range(num_points)
            if i != target
        ]

        unique_subsets = set()
        attempts = 0
        max_attempts = M * 10

        while (
            len(unique_subsets) < M
            and attempts < max_attempts
        ):
            if stratified:
                k = random.randint(
                    0,
                    len(others)
                )

                subset = (
                    random.sample(
                        others,
                        k
                    )
                    if k > 0 else []
                )
            else:
                subset = [
                    i for i in others
                    if random.random() < 0.5
                ]

            unique_subsets.add(
                tuple(
                    sorted(subset)
                )
            )

            attempts += 1

        coalition_dict[target] = {
            "without": list(unique_subsets),
            "with": [
                c + (target,)
                for c in unique_subsets
            ],
        }

    return coalition_dict


def compute_shapley_values(
    model,
    points,
    class_idx,
    device,
    M=200
):
    N = len(points)

    coalitions = generate_sampled_coalitions(
        N,
        M
    )

    shapley = np.zeros(N)

    for target in tqdm(
        range(N),
        desc="Computing SHAP"
    ):
        deltas = []

        for without, with_ in zip(
            coalitions[target]["without"],
            coalitions[target]["with"]
        ):
            subset_without = (
                points[list(without)]
                if len(without) > 0
                else np.zeros((0, 3))
            )

            subset_with = points[
                list(with_)
            ]

            v_without = predict(
                model,
                subset_without,
                class_idx,
                device
            )

            v_with = predict(
                model,
                subset_with,
                class_idx,
                device
            )

            deltas.append(
                v_with - v_without
            )

        shapley[target] = np.mean(
            deltas
        )

    return shapley


def _logit(p, eps=1e-6):
    p = np.clip(
        p,
        eps,
        1.0 - eps
    )

    return np.log(
        p / (1.0 - p)
    )


def save_supersplat_pointlike_ply(
    points,
    original_points,
    shapley_values,
    ply_path,
    opacity=0.95,
    size_mode="bbox_frac",
    size_value=0.003,
):
    SH_C0 = 0.28209479177387814

    points = np.asarray(
        points,
        dtype=np.float32
    )

    shapley_values = np.asarray(
        shapley_values,
        dtype=np.float32
    )

    vmin = float(
        shapley_values.min()
    )

    vmax = float(
        shapley_values.max()
    )

    denom = (
        vmax - vmin
        if vmax != vmin
        else 1.0
    )

    t = np.clip(
        (shapley_values - vmin)
        / denom,
        0.0,
        1.0
    )

    rgb = cm.get_cmap(
        "inferno"
    )(t)[:, :3].astype(
        np.float32
    )

    f_dc = (
        rgb - 0.5
    ) / SH_C0

    opacity_raw = float(
        _logit(opacity)
    )

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    diag = float(
        np.linalg.norm(
            maxs - mins
        )
    )

    if size_mode == "bbox_frac":
        displayed_sigma = max(
            1e-9,
            size_value * diag
        )

    else:
        displayed_sigma = max(
            1e-9,
            float(size_value)
        )

    scale_raw = float(
        np.log(
            displayed_sigma
        )
    )

    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
        "",
    ]).encode("ascii")

    pack = struct.Struct(
        "<" + "f" * 17
    ).pack

    with open(
        ply_path,
        "wb"
    ) as f:
        f.write(header)

        for i in range(
            len(points)
        ):
            x, y, z = original_points[i]
            r, g, b = f_dc[i]

            f.write(
                pack(
                    float(x),
                    float(y),
                    float(z),
                    0.0,
                    0.0,
                    0.0,
                    float(r),
                    float(g),
                    float(b),
                    float(opacity_raw),
                    scale_raw,
                    scale_raw,
                    scale_raw,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )

    print(
        f"Saved explanation: {ply_path}"
    )


def construct_output_filename(
    output_path,
    input_path
):
    base_name = os.path.splitext(
        os.path.basename(
            input_path
        )
    )[0]

    return os.path.join(
        output_path,
        f"{base_name}_shap.ply"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="PointNet SHAP explanation"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input point cloud"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--num_point",
        type=int,
        default=1024,
        help="Number of sampled points"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=300,
        help="Number of SHAP samples"
    )

    return parser.parse_args()


def main(args):
    os.makedirs(
        args.output_path,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        args.checkpoint
    ).to(device)

    points_full = load_point_cloud(
        args.input
    )

    points_full_norm = normalize_points(
        points_full
    )

    points, sampled_indices = (
        random_downsample(
            points_full_norm,
            args.num_point
        )
    )

    class_to_explain = (
        get_predicted_class(
            points,
            model,
            device
        )
    )

    print(
        "Explaining class:",
        class_to_explain
    )

    shapley_values = (
        compute_shapley_values(
            model,
            points,
            class_to_explain,
            device=device,
            M=args.num_samples
        )
    )

    output_file = (
        construct_output_filename(
            args.output_path,
            args.input
        )
    )

    save_supersplat_pointlike_ply(
        points,
        points_full[sampled_indices],
        shapley_values,
        output_file
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)