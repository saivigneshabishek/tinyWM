from pathlib import Path
import h5py
import torch
from random import Random
from torch.utils.data import Dataset

class TinyWMDataset(Dataset):
    """returns frames and actions"""
    def __init__(self, path="data/smb_30fps_256x240.h5", window_len=64, stride=1, train=True, n_val_episodes=25, seed=404):
        self.h5_path = Path(path)
        self.window_len = window_len
        self.split = "train" if train else "val"
        self._h5 = None
        self.sample_indices = []
        with h5py.File(self.h5_path, "r") as h5:
            episode_offsets = h5["episode_offsets"][:]
            self.actions_ram = h5["action"][:]
        
        num_episodes = len(episode_offsets) - 1
        episode_ids = [i for i in range(num_episodes)]
        Random(seed).shuffle(episode_ids)
        if self.split == "train":
            split_episode_ids = episode_ids[n_val_episodes:]
        else:
            split_episode_ids = episode_ids[:n_val_episodes]
        for idx in split_episode_ids:
            start = int(episode_offsets[idx])
            end = int(episode_offsets[idx + 1])
            for i in range(start, end - self.window_len + 1, stride):
                self.sample_indices.append((i, i+self.window_len))

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        start, end = self.sample_indices[idx]
        frames = self._h5["frames"][start:end]
        frames = torch.from_numpy(frames).permute(0,3,1,2) # (T, 3, H, W) uint8
        actions = torch.from_numpy(self.actions_ram[start:end]).long()
        return frames, actions
    
    def __len__(self):
        return len(self.sample_indices)