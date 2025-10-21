#!/usr/bin/env python3
"""
Real KITTI Runner

This script runs the fusion pipeline on a real KITTI-format subset.
It expects the following structure at --dataset_path (default: /content/kitti):
  dataset_path/
    image_2/   000000.png, 000001.png, ...
    velodyne/  000000.bin, 000001.bin, ... (float32 x,y,z,intensity)
    calib/     000000.txt  (KITTI calib format; used for all frames if per-frame not present)
    label_2/   000000.txt  (KITTI label_2 format)

Usage (Colab/local):
  python tools/run_kitti_real.py --dataset_path /content/kitti --max_samples 10 --save_plots

Tip: Use --download_demo to fetch a tiny 5-frame demo subset (hosted mirrors not included).
"""
import argparse
import os
from pathlib import Path
import sys
import json

# Ensure src is importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from main import FusionPipeline


def ensure_kitti_dirs(base: Path):
    (base / 'image_2').mkdir(parents=True, exist_ok=True)
    (base / 'velodyne').mkdir(parents=True, exist_ok=True)
    (base / 'calib').mkdir(parents=True, exist_ok=True)
    (base / 'label_2').mkdir(parents=True, exist_ok=True)


def write_config(original_cfg: Path, dataset_path: Path, max_samples: int):
    import yaml
    cfg = yaml.safe_load(original_cfg.read_text())
    cfg['data']['dataset_path'] = str(dataset_path)
    cfg['data']['max_samples'] = int(max_samples)
    with open(original_cfg, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def list_available_indices(dataset_path: Path):
    imgs = sorted([p.stem for p in (dataset_path / 'image_2').glob('*.png')])
    lids = sorted([p.stem for p in (dataset_path / 'velodyne').glob('*.bin')])
    labels = sorted([p.stem for p in (dataset_path / 'label_2').glob('*.txt')])
    # intersection to ensure all exist
    common = sorted(set(imgs) & set(lids) & set(labels))
    return [int(x) for x in common]


def main():
    parser = argparse.ArgumentParser(description='Run fusion on real KITTI subset')
    parser.add_argument('--dataset_path', type=str, default='/content/kitti')
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Explicit indices like 0 1 2')
    parser.add_argument('--config', type=str, default=str(REPO_ROOT / 'config' / 'config.yaml'))
    parser.add_argument('--save_plots', action='store_true')
    parser.add_argument('--download_demo', action='store_true',
                        help='Create folders and print instructions to add 5 demo frames')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    ensure_kitti_dirs(dataset_path)

    if args.download_demo:
        print('\nDemo subset instructions:')
        print('- Upload 000000..000004.* files into:')
        print(f'  {dataset_path}/image_2  (PNG)')
        print(f'  {dataset_path}/velodyne (BIN)')
        print(f'  {dataset_path}/label_2  (TXT)')
        print(f'  {dataset_path}/calib    (000000.txt)')
        print('\nYou can place per-frame calib files or a single 000000.txt used for all frames.')
        print('Use Google Drive or HTTP downloads as preferred.')

    # Update config to real path and max_samples
    cfg_path = Path(args.config)
    if cfg_path.exists():
        write_config(cfg_path, dataset_path, args.max_samples)
    else:
        print(f'Warning: config not found at {cfg_path}, using defaults')

    # Resolve indices to run
    if args.indices is None:
        common_indices = list_available_indices(dataset_path)
        if not common_indices:
            print('No common indices found (need matching png/bin/txt). Exiting.')
            return 1
        sample_indices = common_indices[:args.max_samples]
    else:
        sample_indices = args.indices

    # Run pipeline
    pipeline = FusionPipeline(str(cfg_path))
    # honor save_plots flag
    pipeline.config['visualization']['save_plots'] = bool(args.save_plots)

    results = pipeline.run_demo(sample_indices=sample_indices)

    # Save metrics and a short summary
    out_dir = Path(pipeline.config['evaluation']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics_real.json', 'w') as f:
        json.dump(results.get('metrics', {}), f, indent=2)
    with open(out_dir / 'run_info.json', 'w') as f:
        json.dump({
            'dataset_path': str(dataset_path),
            'indices': sample_indices,
            'max_samples': args.max_samples
        }, f, indent=2)

    print('\n✅ Run complete!')
    print(f'- Samples processed: {len(results.get("samples", []))}')
    print(f'- Metrics: {out_dir / "metrics_real.json"}')
    print(f'- Visuals saved: {bool(args.save_plots)} in {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
