#!/usr/bin/env python3
"""Fine-tune YOLO26n with the existing balloon dataset and training settings."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Balloon dataset YAML')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    model = YOLO(str(workspace / 'src/ibvs_perception/models/yolo26n.pt'))
    model.train(
        data=args.data,
        imgsz=1280,
        epochs=80,
        batch=4,
        device=args.device,
        workers=4,
        project=str(workspace / 'results/perception'),
        name='balloon_yolo26n',
        seed=2026,
        deterministic=False,
        patience=18,
        close_mosaic=10,
        mosaic=0.15,
        mixup=0.0,
        scale=0.25,
        translate=0.05,
        degrees=3.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        amp=True,
        cache=False,
        plots=True,
        verbose=True,
    )
    print(f'Best weights: {model.trainer.best}')


if __name__ == '__main__':
    main()
