"""Multi-pass anchor & inflection-point ball possession inference package."""

import numpy as np

H, A, D, X = 0, 1, 2, 3
UNDETERMINED = -1
NOBODY = -1

LABEL_STRINGS = np.array(["H", "A", "D", "X"], dtype="<U1")

