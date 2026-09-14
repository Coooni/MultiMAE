#!/usr/bin/env python
"""
check_valid_paths.py
txt 파일 내 경로들이 실제로 존재하는지 확인.

사용법:
    python check_valid_paths.py --txt_files nova/IA/pair_S1.txt nova/IA/pair_S2.txt ...
    
    # 또는 특정 디렉토리의 모든 txt 파일
    python check_valid_paths.py --txt_dir nova/IA
    python check_valid_paths.py --txt_dir nova/IA --txt_dir nova/IL
"""
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

print_lock = threading.Lock()


VALID_PREFIX = '/work/mech-ai-scratch/bgekim'

def check_file(path: str) -> bool:
    p = path.strip()
    if not p.startswith(VALID_PREFIX):
        return False   # prefix 잘린 경로는 invalid 처리
    return Path(p).exists()


def check_txt(txt_path: str, num_workers: int = 16) -> dict:
    """txt 파일 내 경로 전체 검증. 빠른 병렬 처리."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return {'txt': str(txt_path), 'error': 'txt file not found', 'missing': [], 'total': 0}

    with open(txt_path) as f:
        paths = [l.strip() for l in f if l.strip()]

    total = len(paths)
    missing = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_file, p): p for p in paths}
        for future in as_completed(futures):
            p = futures[future]
            if not future.result():
                missing.append(p)

    return {
        'txt':    str(txt_path),
        'total':  total,
        'missing': missing,
        'ok':     total - len(missing),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_files', nargs='+', default=[], help='개별 txt 파일 경로들')
    parser.add_argument('--txt_dir',   action='append', default=[], help='디렉토리 (반복 사용 가능)')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_missing', action='store_true',
                        help='missing 경로를 missing_*.txt 파일로 저장')
    args = parser.parse_args()

    # 체크할 txt 파일 목록 수집
    txt_files = list(args.txt_files)
    for d in args.txt_dir:
        txt_files += [str(p) for p in Path(d).glob('*.txt')]

    if not txt_files:
        print('❌ 체크할 txt 파일이 없어요. --txt_files 또는 --txt_dir 지정해주세요.')
        return

    txt_files = sorted(set(txt_files))
    print(f'\n총 {len(txt_files)}개 txt 파일 검사 중...\n')

    all_ok = True
    for txt in txt_files:
        result = check_txt(txt, args.num_workers)

        if 'error' in result:
            print(f'❌ {result["txt"]} → {result["error"]}')
            all_ok = False
            continue

        n_missing = len(result['missing'])
        status = '✅' if n_missing == 0 else '❌'
        print(f'{status} {Path(result["txt"]).name:30s} | total={result["total"]:>8,} | ok={result["ok"]:>8,} | missing={n_missing:>6,}')

        if n_missing > 0:
            all_ok = False
            # 첫 5개 샘플 출력
            for p in result['missing'][:5]:
                print(f'     missing: {p}')
            if n_missing > 5:
                print(f'     ... and {n_missing - 5} more')

            # missing 경로 저장
            if args.save_missing:
                save_path = Path(txt).parent / f'missing_{Path(txt).stem}.txt'
                with open(save_path, 'w') as f:
                    f.write('\n'.join(result['missing']) + '\n')
                print(f'     💾 saved → {save_path}')

    print()
    if all_ok:
        print('✅ 모든 경로 정상!')
    else:
        print('❌ 일부 경로에 문제가 있어요. 위 내용 확인해주세요.')


if __name__ == '__main__':
    main()