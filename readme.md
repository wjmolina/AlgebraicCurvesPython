# AlgebraicCurvesPython
This repo is a continuation of the original [AlgebraicCurves](https://github.com/wjmolina/AlgebraicCurves) repo, which we developed using MATLAB.

We chose Python due to its powerful packages, relevance, and open-source nature.

Only a tiny part of the functionality in the original repo is present in this one.

This repo comprises four files:

 - `functions.py`: This file contains functions to reconstruct images of algebraic curves using the power, separable Bernstein, and non-separable Bernstein methods. It also contains other useful functions.
 - `globals.py`: This file contains global variables (B-splines and their expansion coefficients) that were obtained using functions in the original repo. As a result, the aforementioned functions reconstruct only power and separable Bernstein images of size 513 by 513 and non-separable Bernstein images of size 1025 by 1025, all up to degree 4.
 - `examples.py`: This file contains one reconstruction of each of the three methods for illustration purposes.
 - `requirements`: This file contains the repo dependencies. We recommend
     - creating a virtual environment `$ python -m venv .venv`,
     - activating it `$ source .venv/bin/activate` (in bash/zsh), and
     - installing the repo dependencies `$ pip install -r requirements.txt`.

The work in this repo is summarized in [this work](https://www.dropbox.com/s/y6v2d0bd87qibbv/wjm_dissertation_revised.pdf?dl=0).