#!/usr/bin/env python
"""
print_constants.py
stats npz 파일들에서 mean/std를 읽어서
data_constants_chloe.py에 바로 붙여넣을 수 있는 형태로 출력.

사용법:
    python print_constants.py --stats_dir /work/.../valid_list/stats
"""
import argparse
import numpy as np
from pathlib import Path


MODALITIES = ['s1', 's2', 'dem', 'soil', 'weather']

# npz 파일명 → 상수 이름 매핑
CONST_NAMES = {
    's1':      ('S1',        'IA', 'IL'),
    's2':      ('S2',        'IA', 'IL'),
    'dem':     ('ELEVATION', 'IA', 'IL'),
    'soil':    ('SOIL',      'IA', 'IL'),
    'weather': ('WEATHER',   'IA', 'IL'),
}


def fmt_tuple(arr):
    """numpy array → Python tuple 문자열"""
    vals = ', '.join(f'{v:.6g}' for v in arr.tolist())
    return f'({vals},)' if len(arr) == 1 else f'({vals})'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_dir', required=True,
                        help='stats npz 파일들이 있는 디렉토리')
    parser.add_argument('--states', default='IA-IL',
                        help='주 목록 하이픈 구분 (기본: IA-IL)')
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    states    = args.states.split('-')

    print("# " + "="*60)
    print("# data_constants_chloe.py 에 붙여넣을 상수값")
    print("# " + "="*60)
    print()

    for modality in MODALITIES:
        const_name = CONST_NAMES[modality][0]

        for state in states:
            npz_path = stats_dir / f"{modality}_{state}.npz"
            if not npz_path.exists():
                print(f"# ⚠️  {npz_path} 없음 - 스킵")
                continue

            d    = np.load(npz_path)
            mean = d['mean']
            std  = d['std']

            print(f"{const_name}_MEAN_{state} = {fmt_tuple(mean)}")
            print(f"{const_name}_STD_{state}  = {fmt_tuple(std)}")

        print()


if __name__ == '__main__':
    main()