import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import comb, factorial

from globals import (
    NON_SEP_B_SPLINE,
    NON_SEP_B_SPLINE_EXP_COEFF,
    POW_B_SPLINE,
    POW_B_SPLINE_EXP_COEFF,
    SEP_B_SPLINE,
    SEP_B_SPLINE_EXP_COEFF,
)

np.random.seed(seed=37)


def logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"began evaluating {func.__name__}")
        result = func(*args, **kwargs)
        print(
            f"ended evaluating {func.__name__}. took {time.time() - start_time} seconds\n"
        )
        return result

    return wrapper


@logger
def get_pow_basis():
    x = [np.linspace(0, 512, 513) ** k for k in range(5)]
    pow_basis = []
    for i in range(5):
        for j in range(5 - i):
            pow_basis.append(np.outer(x[j], x[i]))
    return pow_basis


@logger
def get_sep_basis():
    x = np.linspace(0, 512, 513)
    bernstein_basis = [
        comb(4, k) * (x / 512) ** k * (1 - (x / 512)) ** (4 - k) for k in range(5)
    ]
    return [np.outer(x, y) for x in bernstein_basis for y in bernstein_basis]


@logger
def get_non_sep_basis():
    x, y = np.meshgrid(np.linspace(0, 1024, 1025), np.linspace(0, 1024, 1025))
    non_sep_basis = []
    for i in range(5):
        for j in range(5 - i):
            non_sep_basis.append(
                comb(4, i)
                * comb(4 - i, j)
                * x ** i
                * y ** j
                * (1024 - x - y) ** (4 - i - j)
                / 1099511627776
            )
    return non_sep_basis


@logger
def get_pow_M_raw():
    pow_M_raw = []
    for r in range(3):
        for s in range(3):
            pow_M_raw.append([])
            for i in range(5):
                for j in range(5 - i):
                    pow_M_raw[-1].append(
                        (i + r)
                        * np.outer(
                            POW_B_SPLINE_EXP_COEFF[j + s],
                            POW_B_SPLINE_EXP_COEFF[i + r - 1],
                        )
                    )
            pow_M_raw.append([])
            for i in range(5):
                for j in range(5 - i):
                    pow_M_raw[-1].append(
                        (j + s)
                        * np.outer(
                            POW_B_SPLINE_EXP_COEFF[j + s - 1],
                            POW_B_SPLINE_EXP_COEFF[i + r],
                        )
                    )
    return pow_M_raw


@logger
def get_sep_M_raw():
    sep_exp_coeff = defaultdict(lambda: np.zeros(521))
    for t in range(3, 9):
        for v in range(t + 1):
            sep_exp_coeff[(t, v)] = [
                sum(
                    comb(t, v)
                    * comb(v, r)
                    * comb(t - v, s)
                    * 512 ** (-t)
                    * (-256) ** (v - r)
                    * 256 ** (t - v - s)
                    * (-1) ** (v + r + s)
                    * SEP_B_SPLINE_EXP_COEFF[r + s][k]
                    for s in range(t - v + 1)
                    for r in range(v + 1)
                )
                for k in range(521)
            ]
    sep_M_raw = []
    for r in range(5):
        for s in range(5):
            row_1 = []
            row_2 = []
            for k in range(5):
                for l in range(5):
                    q_coeff = (
                        factorial(4)
                        * factorial(k + r)
                        / factorial(k)
                        / factorial(4 + r)
                        * factorial(4)
                        * factorial(l + s)
                        / factorial(l)
                        / factorial(4 + s)
                    )
                    row_1.append(
                        q_coeff
                        * (
                            np.outer(
                                sep_exp_coeff[(4 + r - 1, k + r - 1)],
                                sep_exp_coeff[(4 + s, l + s)],
                            )
                            - np.outer(
                                sep_exp_coeff[(4 + r - 1, k + r)],
                                sep_exp_coeff[(4 + s, l + s)],
                            )
                        )
                    )
                    row_2.append(
                        q_coeff
                        * (
                            np.outer(
                                sep_exp_coeff[(4 + r, k + r)],
                                sep_exp_coeff[(4 + s - 1, l + s - 1)],
                            )
                            - np.outer(
                                sep_exp_coeff[(4 + r, k + r)],
                                sep_exp_coeff[(4 + s - 1, l + s)],
                            )
                        )
                    )
            sep_M_raw.append(row_1)
            sep_M_raw.append(row_2)
    return sep_M_raw


@logger
def get_non_sep_M_raw():
    non_sep_exp_coeff = defaultdict(lambda: np.zeros((1031, 1031)))
    for n in range(3, 8):
        for i in range(n + 1):
            for j in range(n - i + 1):
                for a in range(n - i - j + 1):
                    for b in range(n - i - j - a + 1):
                        c = n - i - j - a - b
                        non_sep_exp_coeff[(n, i, j)] += (
                            comb(n, i)
                            * comb(n - i, j)
                            * factorial(n - i - j)
                            / factorial(a)
                            / factorial(b)
                            / factorial(c)
                            * 1024 ** (a - n)
                            * (-1) ** (b + c)
                            * np.outer(
                                NON_SEP_B_SPLINE_EXP_COEFF[c + j],
                                NON_SEP_B_SPLINE_EXP_COEFF[b + i],
                            )
                        )
    non_sep_M_raw = []
    for r in range(3):
        for s in range(3):
            row_1 = []
            row_2 = []
            for i in range(5):
                for j in range(5 - i):
                    q_coeff = (
                        1024 ** (r + s)
                        * 24
                        * factorial(i + r)
                        * factorial(j + s)
                        / (factorial(i) * factorial(j) * factorial(4 + r + s))
                    )
                    row_1.append(
                        q_coeff
                        * (
                            non_sep_exp_coeff[(4 + r + s - 1, i + r - 1, j + s)]
                            - non_sep_exp_coeff[(4 + r + s - 1, i + r, j + s)]
                        )
                    )
                    row_2.append(
                        q_coeff
                        * (
                            non_sep_exp_coeff[(4 + r + s - 1, i + r, j + s - 1)]
                            - non_sep_exp_coeff[(4 + r + s - 1, i + r, j + s)]
                        )
                    )
            non_sep_M_raw.append(row_1)
            non_sep_M_raw.append(row_2)
    return non_sep_M_raw


@logger
def get_smp_rec(img, b_spline, M_raw, basis, noise=0):
    smp = fftconvolve(img, np.outer(b_spline, b_spline))
    smp += np.random.normal(0, noise, smp.shape)

    M_sum = np.einsum("ijkl,kl->ij", M_raw, smp)
    M = M_sum[:, 1:]
    b = -M_sum[:, 0]
    coeff = np.insert(np.linalg.pinv(M) @ b, 0, 1)

    rec = get_img(coeff, basis)
    rec = max(rec, 1 - rec, key=lambda x: sore_dice_coeff(img, x))

    return smp, rec


@logger
def get_img(coeff, basis):
    return (np.einsum("i,ijk->jk", coeff, basis) <= 0).astype(float)


@logger
def show_imgs(imgs):
    plt.figure()
    for i, (img, lbl) in enumerate(imgs):
        plt.subplot(int(f"{len(imgs) // 2}2{i + 1}"))
        plt.imshow(img)
        plt.title(lbl)
        plt.axis("off")
        plt.gray()
    plt.show()


@logger
def sore_dice_coeff(img, rec):
    true_pos = np.sum((img == rec) * img)
    flse_pos = np.sum((img != rec) * rec)
    flse_neg = np.sum((img != rec) * img)
    return 2 * true_pos / (2 * true_pos + flse_pos + flse_neg)


@logger
def get_stably_bounded_shape(a, b, c, d, height, width):
    M = np.random.randn(3, 3)
    M = M @ M.T
    while not all(np.linalg.eig(M)[0]) > 0:
        M = np.random.randn(3, 3)
        M = M @ M.T
    x, y = np.meshgrid(np.linspace(a, b, height), -np.linspace(c, d, width))
    z = np.array([x ** i * y ** (2 - i) for i in range(3)])
    return (
        np.einsum("ikl,ij,jkl->kl", z, M, z)
        + np.einsum(
            "i,ijk->jk",
            np.random.randn(10),
            np.array([x ** i * y ** j for i in range(4) for j in range(4 - i)]),
        )
        < 0
    ).astype(float)
