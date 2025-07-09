"""
Warning: almost entirely vibecoded

Sorry, but the pipes don't render right.  The opacity difference is completely bugging it - if any low opacity pipe gets drawn over a full opacity pipe then it all becomes low opacity.  Any combination of depth-testing type stuff won't work; I just don't know how to fix it.  So I actually just put the lo and hi animations over each other separate with mode=lighten in Resolve.

Simplex‑style 3‑D attention diagram
––––––––––––––––––––––––––––––––––––
Key change
~~~~~~~~~~
• Background pipes (“lo”) keep depth testing so they occlude each other normally.
• Focus pipes (“hi”) neither test nor write depth *and* are drawn last, so their
  0.8‑opacity teal is composited on top of everything even where pipes overlap.
"""

import numpy as np
from itertools import product
from manim_imports_ext import *


# ----------------------------------------------------------------------
# Re‑usable mini‑dial (needle only, clock‑face optional)
# ----------------------------------------------------------------------
class Dial(VGroup):
    def __init__(self, s, radius=0.375, color=YELLOW, **kwargs):
        super().__init__(**kwargs)
        needle = Line(ORIGIN, radius * RIGHT, stroke_width=4, stroke_color=color)
        needle.rotate(interpolate(-PI / 2, PI / 2, s))
        self.add(needle)


# ----------------------------------------------------------------------
# Main scene
# ----------------------------------------------------------------------
class SimplexAttentionFlow(ThreeDScene):
    def construct(self):

        # ---------------- base data ----------------
        words        = ["The", "quick", "brown", "fox", "jumps"]
        n, focus     = len(words), len(words) - 1                  # focus = «jumps»
        k_sim        = np.array([.15, .48, .25, .60, .50])         # 1‑D key
        q_sim        = np.array([.20, .60, .90, .30, .75])         # 1‑D query
        k_outer      = np.outer(k_sim, k_sim)                      # key × key′
        row_y        = lambda r: (n / 2 - r) * 1.2                 # y‑layout helper

        # ---------------- layout constants ----------
        DIAL_R, GAP_AD, WORD_GAP, ARR   = 0.375, 0.126, 0.60, 0.576
        X_KEY, X_Q                      = -2, 2
        X_KARR = X_KEY - (DIAL_R + GAP_AD + ARR + WORD_GAP)        # word anchors
        X_QARR = X_Q   + (DIAL_R + GAP_AD + ARR + WORD_GAP)
        GRID_GAP, X_GRID                = 0.60, X_KEY
        CELL                            = 0.75 * DIAL_R

        # ---------------- tiny helper ---------------
        def connector(word, dial, buff=0.10, stroke_width=4):
            """Billboarded arrow from word → dial."""
            arr = Arrow(word, dial, buff=buff, stroke_width=stroke_width).set_stroke(width=0)
            arr.always.set_perpendicular_to_camera(self.camera.frame)
            return arr

        # ---------------- query column --------------
        R_w, R_a, R_d = VGroup(), VGroup(), VGroup()
        for i, w in enumerate(words):
            y    = row_y(i)
            word = Text(w, weight=BOLD if i == focus else NORMAL).move_to([X_QARR, y, 0])
            dial = Dial(q_sim[i], color=YELLOW).move_to([X_Q, y, 0])
            R_w.add(word); R_d.add(dial); R_a.add(connector(word, dial))
        self.add(R_w, R_a, R_d)

        # ---------------- K′ header row (RED) -------
        top_y = row_y(n)                                           # 6th row
        Kp_w, Kp_a, Kp_d = VGroup(), VGroup(), VGroup()
        for j, w in enumerate(words[::-1]):
            z    = (j - (n - 1) / 2) * GRID_GAP
            word = Text(w, color=RED ).move_to([X_KARR, top_y, z])
            dial = Dial(k_sim[j], color=RED).move_to([X_KEY, top_y, z])
            Kp_w.add(word); Kp_d.add(dial); Kp_a.add(connector(word, dial))
        for grp in (Kp_w, Kp_a, Kp_d):
            grp.set_opacity(0.5)
        self.add(Kp_w, Kp_a, Kp_d)

        # ---------------- K footer column (BLUE) ----
        bot_z = (n + 1) / 2 * GRID_GAP                             # 6th column
        K_w, K_a, K_d = VGroup(), VGroup(), VGroup()
        for i, w in enumerate(words):
            y    = row_y(i)
            word = Text(w, color=BLUE).move_to([X_KARR, y, bot_z])
            dial = Dial(k_sim[i], color=BLUE).move_to([X_KEY, y, bot_z])
            K_w.add(word); K_d.add(dial); K_a.add(connector(word, dial))
        for grp in (K_w, K_a, K_d):
            grp.set_opacity(0.5)
        self.add(K_w, K_a, K_d)

        # ---------------- 5×5 K×K′ dial sheet -------
        prod_dials = VGroup()
        for i, j in product(range(n), repeat=2):
            s    = float(k_outer[i, j])
            dial = Dial(s, color=PURPLE)
            dial.rotate(PI / 2, axis=RIGHT)
            dial.move_to([X_GRID, row_y(i), (j - (n - 1) / 2) * GRID_GAP])
            prod_dials.add(dial)
        self.add(prod_dials)

        # ---------------- sparse pipes -------------- #
        TAU   = 0.05                     # soft-max temperature (↓ sharper)
        THR   = 0.10                     # drop pipes whose weight < THR
        MIN_W, MAX_W, DYN_W = 0.20,  9, 1.2   # width mapping
        hi, lo = VGroup(), VGroup()      # focus / background groups

        # 1) soft-max over entire 5×5 grid (toy visual, not per-query)
        soft_w = np.exp(k_outer / TAU)
        soft_w /= soft_w.sum()           # ⇒ weights ∈ (0,1), sum = 1
        soft_w /= soft_w.max()           # rescale so max = 1 for width map

        def make_pipe(q, i, j, w):
            y, z  = row_y(i), (j - (n - 1) / 2) * GRID_GAP
            src   = [X_GRID + CELL / 2, y, z]
            dst   = [X_Q   - 0.30     , row_y(q), 0]
            width = MIN_W + (MAX_W - MIN_W) * (w ** DYN_W)
            op    = 0.80 if q == focus else 0.024
            return Line(
                src, dst, buff=0,
                stroke_color=TEAL, stroke_width=width, stroke_opacity=op
            ).insert_n_curves(20)

        # 2) build the pipe groups
        for q in range(n):                        # query index
            for i, j in product(range(n), repeat=2):
                w = soft_w[i, j]
                if w < THR:
                    continue                      # skip faint pipes
                (hi if q == focus else lo).add(make_pipe(q, i, j, w))

        # 3) depth-buffer strategy
        for mob in lo.family_members_with_points():
            mob.apply_depth_test()                # bg pipes occlude normally
        for mob in hi.family_members_with_points():
            mob.deactivate_depth_test()           # focus pipes ignore Z-buffer

        # for some reason hi has to be added first - probably camera shifted from original z axis
        self.add(hi)
        self.add(lo)

        pipes = VGroup(*lo, *hi)

        # ---------------- flowing dots -------------
        def flowing_dots():
            t = self.time * 0.25
            dots = VGroup()
            for k, p in enumerate(pipes):
                dot = Dot(radius=0.01 * p.get_stroke_width(), fill_color=TEAL, stroke_width=0)
                dot.set_fill(opacity=p.get_stroke_opacity())
                dot.move_to(p.point_from_proportion((t + 0.05 * k) % 1))
                dots.add(dot)
            return dots
        self.add(always_redraw(flowing_dots))

        # ---------------- camera -------------------
        self.camera.frame.set_euler_angles(phi=55 * DEGREES, theta=20 * DEGREES)
        self.wait(8)
