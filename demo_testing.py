import os
import time
import rasterio
import warnings
import numpy as np
import torch
import cv2
import open_earth_map as oem
from pathlib import Path

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    start = time.time()

    OEM_DATA_DIR = "OpenEarthMap_wo"
    TEST_LIST = os.path.join(OEM_DATA_DIR, "kanagawa.txt")

    N_CLASSES = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PREDS_DIR = "re_pre"
    TASK = "building"
    os.makedirs(PREDS_DIR, exist_ok=True)

    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    test_fns = [str(f) for f in fns if f.name in np.loadtxt(TEST_LIST, dtype=str)]

    print("Total samples   :", len(fns))
    print("Testing samples :", len(test_fns))


    test_data = oem.dataset.OpenEarthMapDataset(
        test_fns,
        n_classes=N_CLASSES,
        augm=None,
        testing=True,
        task = TASK

    )

    network = oem.networks.UNetFormer(in_channels=3, n_classes=N_CLASSES,  backbone_name="seresnet152d")
    network = oem.utils.load_checkpoint(
        network,
        model_name="unetformer_model_building_100.pth",
        model_dir="outputs",
    )

    network.eval().to(DEVICE)
    for idx in range(len(test_fns)):
        img, fn = test_data[idx][0], test_data[idx][2]

        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        prd = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))

        fout = os.path.join(PREDS_DIR, fn.split("/")[-1])
        with rasterio.open(fn, "r") as src:
            profile = src.profile
            prd = cv2.resize(
                prd,
                (profile["width"], profile["height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            with rasterio.open(fout, "w", **profile) as dst:
                for idx in src.indexes:
                    dst.write(prd[:, :, idx - 1], idx)
