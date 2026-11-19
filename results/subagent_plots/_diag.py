import glob
import numpy as np
import matplotlib.image as mpimg

patterns = [
    r"figs\w100_layers\depth_heatmap__*.png",
    r"figs\w100_layers\lam_local__L27*.png",
    r"figs\w100_layers\ft_rc__*.png",
    r"figs\h_last\hlast_pairdiv__*.png",
    r"figs\tok_tf\*.png",
    r"figs\lp_pos50\lp_heatmap__b0*.png",
    r"figs\margins_norms\*.png",
    r"figs\depthnpz\*.png",
    r"figs\lp_r02\lp_depth_lag__*.png",
    r"figs\swin_state\*.png",
]
for pat in patterns:
    for f in sorted(glob.glob(pat))[:2]:
        img = mpimg.imread(f)
        gray = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        h, w = gray.shape
        band = gray[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        print(f"{float((band < 0.92).mean()):6.3f}  {f}")
