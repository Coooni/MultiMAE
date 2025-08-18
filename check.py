# check.py
import argparse
import rasterio

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--modis_txt', type=str, default='/scratch/bepk/bkim2/MultiMAE/valid_list/valid_MODIS.txt')
    p.add_argument('--s2_txt', type=str, default='/scratch/bepk/bkim2/MultiMAE/valid_list/valid_S2.txt')
    return p.parse_args()

def check(txt):
    bad = []
    with open(txt) as f:
        for line in f:
            path = line.strip()
            try:
                with rasterio.open(path) as src:
                    _ = src.read(1, out_dtype='float32')
            except Exception as e:
                bad.append((path, repr(e)))
    return bad

if __name__ == "__main__":
    args = parse_args()
    bad_modis = check(args.modis_txt)
    bad_s2 = check(args.s2_txt)

    print(f"# bad MODIS: {len(bad_modis)}")
    for p, e in bad_modis[:20]:
        print(p, e)
    print(f"# bad S2: {len(bad_s2)}")
    for p, e in bad_s2[:20]:
        print(p, e)
