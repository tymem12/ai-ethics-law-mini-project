from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
from plyfile import PlyData
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from sklearn.preprocessing import normalize 
from einops import repeat, rearrange

FEATURE_NAMES: list[str] = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity",
]


def exists(x):
    return x is not None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))





def prepare_gaussian_cloud(pts: np.ndarray) -> tuple[np.ndarray]:

    pts[:, 10] = sigmoid(pts[:, 10])

    if pts.shape[0] == 0:
        return (np.zeros((0, 8), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(0, dtype=bool))

    q = pts[:, 6:10]
    q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    pts[:, 6:10] = q / q_norm
    pts[:, 3:6] = normalize(pts[:, 3:6])

    xyz = pts[:, :3]
    gauss = pts[:, 3:]

    ctr = xyz.mean(axis=0, keepdims=True)
    xyz_c = xyz - ctr
    scale = np.max(np.linalg.norm(xyz_c, axis=1)) + 1e-8
    xyz_normalized = (xyz_c / scale).astype(np.float32)

    return gauss.astype(np.float32), xyz_normalized.astype(np.float32)


def farthest_point_sampling(x: torch.Tensor, n_sample: int, generator: torch.Generator = None, start_idx: int = None):
    # x: (b, n, 3)
    b, n = x.shape[:2]
    assert n_sample <= n, "not enough points to sample"
    

    if n_sample == n:
        return repeat(torch.arange(n_sample, dtype=torch.long, device=x.device), 'm -> b m', b=b)

    # start index
    if exists(start_idx):
        sel_idx = torch.full((b, n_sample), start_idx, dtype=torch.long, device=x.device)
    else:
        sel_idx = torch.randint(n, (b, n_sample), dtype=torch.long, device=x.device, generator=generator)

    cur_x = rearrange(x[torch.arange(b), sel_idx[:, 0]], 'b c -> b 1 c')
    min_dists = torch.full((b, n), dtype=x.dtype, device=x.device, fill_value=float('inf'))
    for i in range(1, n_sample):
        dists = torch.linalg.norm(x - cur_x, dim=-1)
        min_dists = torch.minimum(dists, min_dists)

        idx_farthest = torch.max(min_dists, dim=-1).indices
        sel_idx[:, i] = idx_farthest
        cur_x[:, 0, :] = x[torch.arange(b), idx_farthest]

    return sel_idx
class GaussianPointCloud(Dataset):
    def __init__(
        self,
        root: Path,
        num_points: int = 2048,
        sampling_method: str | None = "random", 
        random_seed: int | None = None
    ):
        self.root = Path(root)
        self.num_points = num_points
        self.sampling_method: str | None = sampling_method
        self.random_seed = random_seed
        self.rng = default_rng(self.random_seed) if exists(self.random_seed) else None 
        self.pt_generator = torch.Generator() if exists(random_seed) else None  
        if exists(random_seed):
            self.pt_generator.manual_seed(self.random_seed)

        self.files: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx = {}

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            self.class_to_idx[class_name] = len(self.class_to_idx)
            self.classes.append(class_name)
            for ply_path in class_dir.glob("*.ply"):
                self.files.append((ply_path, self.class_to_idx[class_name]))

    @staticmethod
    def _read_ply(path: Path) -> np.ndarray:
        plydata = PlyData.read(str(path))
        vertex = plydata["vertex"]
        data = np.vstack([vertex[name] for name in FEATURE_NAMES]).T
        return data.astype(np.float32)

    def _random_sample(self, pts: np.ndarray) -> np.ndarray:
        N = pts.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False) if self.rng is None else self.rng.choice(N, self.num_points, replace=False)
        else:
            idx = np.random.choice(N, self.num_points, replace=True) if self.rng is None else self.rng.choice(N, self.num_points, replace=True)
        return idx

    
    def _sample_index(self, pts: np.ndarray) -> np.ndarray:
        if self.sampling_method == "random":
            return self._random_sample(pts)
        elif self.sampling_method == "fps": 
            pts_tensor = torch.from_numpy(pts[:, :3]).float().unsqueeze(0)
            indices = farthest_point_sampling(pts_tensor, self.num_points, self.pt_generator).squeeze(0)
            return indices.numpy()
        else: raise ValueError(f"Unknown sampling method: {self.sampling_method}")


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, label = self.files[idx]
        pts = self._read_ply(path)
        indices = np.arange(pts.shape[0])

        if self.sampling_method != "original_size":
            indices = self._sample_index(pts)
            pts = pts[indices]

        gauss, xyz_normalized = prepare_gaussian_cloud(pts)
        gauss = torch.from_numpy(gauss)
        xyz_normalized = torch.from_numpy(xyz_normalized)
        
        return {
            "gauss": gauss,
            "xyz_normalized": xyz_normalized,
            "label": torch.tensor(label, dtype=torch.long),
            "indices": torch.from_numpy(indices).long(),
        }


def collate_fn(batch):
    max_points = max(item["gauss"].shape[0] for item in batch)

    padded_features = []
    padded_xyz_normalized = []
    padded_indices = []
    labels = []
    masks = []

    for item in batch:
        features = item["gauss"]
        xyz_normalized = item["xyz_normalized"]
        indices = item["indices"]
        num_points = features.shape[0]
        
        mask = torch.zeros(max_points, dtype=torch.bool)
        mask[:num_points] = True
        masks.append(mask)

        padding_size = max_points - num_points
        
        if padding_size > 0:
            feature_padding = torch.zeros((padding_size, features.shape[1]), dtype=features.dtype)
            features = torch.cat([features, feature_padding], dim=0)
            
            xyz_padding = torch.zeros((padding_size, 3), dtype=xyz_normalized.dtype)
            xyz_normalized = torch.cat([xyz_normalized, xyz_padding], dim=0)

            indices_padding = torch.full((padding_size,), -1, dtype=torch.long)
            indices = torch.cat([indices, indices_padding], dim=0)

        padded_features.append(features)
        padded_xyz_normalized.append(xyz_normalized)
        padded_indices.append(indices)
        labels.append(item["label"])


    return {
        "gauss": torch.stack(padded_features).transpose(1, 2), # (B, D, N)
        "xyz_normalized": torch.stack(padded_xyz_normalized), # (B, N, 3)
        "label": torch.stack(labels),
        "mask": torch.stack(masks), # (B, N)
        "indices": torch.stack(padded_indices), # (B, N)    
    }


class GaussianDataModule(pl.LightningDataModule):
    def __init__(self,
             data_dir: str,
             batch_size: int = 32,
             num_workers: int = 4,
             val_split: float = 0.1,
             sampling: str = "random",
             num_points: int = 4096,
             seed: int = 42) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds, self.val_ds = None, None
        self.num_classes, self.in_dim = 0, len(FEATURE_NAMES)
        self.data_dir = data_dir
        self.sampling = sampling

    def setup(self, stage: str | None = None):
        root_path = Path(self.data_dir)
        if (root_path / "train").exists() and (root_path / "test").exists():
            train_path = root_path / "train"
            test_path = root_path / "test"
        else:
            train_path = root_path
            test_path = root_path

        self.test_ds = GaussianPointCloud(
            test_path,
            num_points=self.hparams.num_points,
            sampling_method=self.hparams.sampling,
            random_seed=self.hparams.seed,
        )
        dataset = GaussianPointCloud(
            train_path,
            num_points=self.hparams.num_points,
            sampling_method=self.hparams.sampling,
            random_seed=self.hparams.seed,
        )
        self.num_classes = len(dataset.classes)
        n_val = int(len(dataset) * self.hparams.val_split)
        n_train = len(dataset) - n_val
        self.train_ds, self.val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.hparams.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=False,
            pin_memory=True,

        )

    def test_dataloader(self):
        
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            persistent_workers=False,
            pin_memory=True,

        )



    

