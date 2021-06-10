import numpy as np

from functions import (
    get_non_sep_basis,
    get_non_sep_M_raw,
    get_pow_basis,
    get_pow_M_raw,
    get_sep_basis,
    get_sep_M_raw,
    get_smp_rec,
    show_imgs,
    sore_dice_coeff,
)
from globals import NON_SEP_B_SPLINE, POW_B_SPLINE, SEP_B_SPLINE

x, y = np.meshgrid(np.linspace(-1.5, 1.5, 513), np.linspace(-1.5, 1.5, 513))
img = (x ** 2 + y ** 2 - 1 <= 0).astype(float)

########################
# Power Reconstruction #
########################
smp, rec = get_smp_rec(
    img,
    POW_B_SPLINE,
    get_pow_M_raw(),
    get_pow_basis(),
    noise=0.1,
)
show_imgs(
    [
        [img, "image"],
        [smp, "noisy sampling"],
        [rec, "reconstruction"],
        [abs(img - rec), f"error\n{sore_dice_coeff(img, rec)}"],
    ]
)

############################
# Separable Reconstruction #
############################
smp, rec = get_smp_rec(
    img,
    SEP_B_SPLINE,
    get_sep_M_raw(),
    get_sep_basis(),
    noise=0.1,
)
show_imgs(
    [
        [img, "image"],
        [smp, "noisy sampling"],
        [rec, "reconstruction"],
        [abs(img - rec), f"error\n{sore_dice_coeff(img, rec)}"],
    ]
)

################################
# Non-Separable Reconstruction #
################################
img = np.pad(img, [[0, 512]])
smp, rec = get_smp_rec(
    img,
    NON_SEP_B_SPLINE,
    get_non_sep_M_raw(),
    get_non_sep_basis(),
    noise=0.1,
)
print(smp.shape)
show_imgs(
    [
        [img[:513, :513], "image"],
        [smp, "noisy sampling (padded)"],
        [rec[:513, :513], "reconstruction"],
        [
            abs(img - rec)[:513, :513],
            f"error\n{sore_dice_coeff(img[:513,:513], rec[:513,:513])}",
        ],
    ]
)
