#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import logging
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.optim

import mon
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullpath = args["use_fullpath"]
    verbose      = args["verbose"]
    
    config_file = args["config_file"]
    # config_file = current_dir / "sam2_configs" / config_file
    
    # Start
    console.rule(f"[bold red] {fullname}")
    console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Model
    sam2 = build_sam2(str(config_file), str(weights), device="cuda", apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model                          = sam2,
        points_per_side                = args["network"]["points_per_side"],
        points_per_batch               = args["network"]["points_per_batch"],
        pred_iou_thresh                = args["network"]["pred_iou_thresh"],
        stability_score_thresh         = args["network"]["stability_score_thresh"],
        stability_score_offset         = args["network"]["stability_score_offset"],
        mask_threshold                 = args["network"]["mask_threshold"],
        box_nms_thresh                 = args["network"]["box_nms_thresh"],
        crop_n_layers                  = args["network"]["crop_n_layers"],
        crop_nms_thresh                = args["network"]["crop_nms_thresh"],
        crop_n_points_downscale_factor = args["network"]["crop_n_points_downscale_factor"],
        min_mask_region_area           = args["network"]["min_mask_region_area"],
        output_mode                    = args["network"]["output_mode"],
        use_m2m                        = args["network"]["use_m2m"],
        multimask_output               = args["network"]["multimask_output"],
    )
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=sam2, image_size=imgsz)
        console.log(f"FLOPs : {flops:.4f}")
        console.log(f"Params: {params:.4f}")
      
    # Disable logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                meta       = datapoint["meta"]
                image_path = mon.Path(meta["path"])
                image      = datapoint["image"]
                
                # Infer
                timer.tick()
                masks = mask_generator.generate(image)
                timer.tock()
                
                # Save
                if save_image:
                    if use_fullpath:
                        relative_path   = image_path.relative_path(data_name)
                        binary_save_dir = save_dir / relative_path.parent / "binary"
                        color_save_dir  = save_dir / relative_path.parent / "color"
                    else:
                        binary_save_dir = save_dir / data_name / "binary"
                        color_save_dir  = save_dir / data_name / "color"
                    # Binary
                    for i, mask in enumerate(masks):
                        output_path = binary_save_dir / f"{image_path.stem}_mask_{i}.jpg"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(output_path), np.uint8(mask["segmentation"]) * 255)
                    # Color
                    output          = np.ones((masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 4))
                    output[:, :, 3] = 0
                    for i, mask in enumerate(masks):
                        mask_bool         = mask["segmentation"]
                        color_mask        = np.concatenate([np.random.random(3), [1.0]])  # 0.35
                        output[mask_bool] = color_mask
                    output_path = color_save_dir / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), np.uint8(output * 255))
    
    # Finish
    console.log(f"Average time: {timer.avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
