# data_constants_chloe.py
# --------------------------------------------------------
# Dynamically load stats from npz files using stats_dir + state
# e.g.) stats_dir=/work/.../stats, state=IA
#       → reads s1_IA.npz, s2_IA.npz, ... and returns as dict
# CDL is a segmentation label, so no normalization applied
# --------------------------------------------------------
import numpy as np
from pathlib import Path

MODALITIES = ['s1', 's2', 'elevation', 'soil', 'weather']


def load_state_stats(stats_dir: str, state: str) -> dict:
    """
    stats_dir/{modality}_{state}.npz 에서 mean/std 로드.

    Returns:
        {
            'mean': {'s1': tuple, 's2': tuple, ...},
            'std':  {'s1': tuple, 's2': tuple, ...},
        }
    """
    stats_dir = Path(stats_dir)
    mean_dict, std_dict = {}, {}

    for modality in MODALITIES:
        npz_path = stats_dir / f"{modality}_{state}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Stats 파일 없음: {npz_path}\n"
                f"compute_stats.py --modality {modality} 로 먼저 생성해주세요."
            )
        d = np.load(npz_path)
        mean_dict[modality] = tuple(d['mean'].tolist())
        std_dict[modality]  = tuple(d['std'].tolist())

    return {'mean': mean_dict, 'std': std_dict}