"""
Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387–411 — reference centreline profiles.

Table I: u along the vertical centreline (x = 0.5) at 17 y positions.
Table II: v along the horizontal centreline (y = 0.5) at 17 x positions.
Their grid was 129×129 (Re ≤ 1000) with a multigrid solver; the numbers are the accepted
benchmark for the lid-driven cavity.

Transcribed from the paper and cross-checked digit for digit against two independent
reproductions (gist.github.com/ivan-pi). A slip would show up as a single outlier against an
otherwise-matching solver profile — and one does: see SUSPECT below.
"""
import numpy as np

# Table I — y positions and u(x = 0.5)
Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
              0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
U = {
    100:  np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                    -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                    -0.03717, 0.00000]),
    400:  np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
                    -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
                    -0.08186, 0.00000]),
    1000: np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
                    -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
                    -0.18109, 0.00000]),
}

# Table II — x positions and v(y = 0.5)
X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
              0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
V = {
    100:  np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
                    -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
                    0.09233, 0.00000]),
    400:  np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993,
                    -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713,
                    0.18360, 0.00000]),
    1000: np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
                    -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012,
                    0.27485, 0.00000]),
}

# Points where the PRINTED table is inconsistent with itself. Kept verbatim above (never
# "corrected"); the comparison reports the verdict with and without them.
#   Re = 400, v at x = 0.9063 = -0.23827: non-monotone against its neighbours (-0.22847 at
#   x = 0.9453, -0.44993 at x = 0.8594) and 15% off a grid-converged solver that matches the
#   other 16 points to 0.2%. Reproduced identically in every copy of the table found, so it is
#   in the original paper, not a copying error. The solver's value there is -0.388.
SUSPECT = {400: {"v": [0.9063]}}

# Primary-vortex centre (x, y) and streamfunction minimum, from the paper's Table IV
VORTEX = {
    100:  (0.6172, 0.7344, -0.103423),
    400:  (0.5547, 0.6055, -0.113909),
    1000: (0.5313, 0.5625, -0.117929),
}
