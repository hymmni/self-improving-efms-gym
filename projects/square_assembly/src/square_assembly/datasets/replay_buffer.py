"""Minimal read-only ReplayBuffer used by ZarrSequenceDataset
(square_assembly/src/square_assembly/datasets/zarr_dataset.py).

Reimplemented from scratch against the actual zarr layout in use
(square_dataset/square_image_v15.zarr: data/{action,action_mode,agentview_image,
robot0_eef_pos,robot0_eef_quat,robot0_eye_in_hand_image,robot0_gripper_qpos},
meta/episode_ends) rather than ported from any external package — only the narrow
interface ZarrSequenceDataset needs is implemented: `.episode_ends` (1D array of
per-episode cumulative end indices) and `.data[key]` (that key's full array across all
transitions, indexable). Write operations (append/add_episode) are intentionally not
implemented — read-only only.
"""

import zarr


class ReplayBuffer:
    def __init__(self, root):
        self._root = root

    @classmethod
    def create_from_path(cls, zarr_path, mode="r"):
        root = zarr.open(str(zarr_path), mode=mode)
        return cls(root)

    @property
    def data(self):
        return self._root["data"]

    @property
    def episode_ends(self):
        return self._root["meta"]["episode_ends"]
