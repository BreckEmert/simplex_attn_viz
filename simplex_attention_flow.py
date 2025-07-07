import numpy as np
from manim_imports_ext import *
from itertools import product

class Dial(VGroup):
    """Simple needle dial."""
    def __init__(self, s, radius=0.375, color=YELLOW, **kwargs):
        super().__init__(**kwargs)
        needle = Line(ORIGIN, radius * RIGHT, stroke_width=4, stroke_color=color)
        needle.rotate(interpolate(-PI / 2, PI / 2, s))
        self.add(needle)

class SelfAttentionFlow(ThreeDScene):
    def construct(self):
        # ---------------- base data ----------------
        words = ["The", "quick", "brown", "fox", "jumps"]
        n, focus = len(words), len(words) - 1                     # focus = “jumps”
        k_sim  = np.array([.15, .48, .25, .60, .50])              # 1-D key
        q_sim  = np.array([.20, .60, .90, .30, .75])              # 1-D query
        k_outer = np.outer(k_sim, k_sim)                          # 2-D key×key′

        # ---------------- layout constants ---------
        DIAL_R, GAP_AD, WORD_GAP, ARR = 0.375, 0.126, 0.60, 0.576
        X_KEY, X_Q  = -2, 2
        X_KARR = X_KEY - (DIAL_R + GAP_AD + ARR + WORD_GAP)       # K & K′ word anchors
        X_QARR = X_Q  + (DIAL_R + GAP_AD + ARR + WORD_GAP)        # Q   word anchors
        GRID_GAP, X_GRID = 0.60, X_KEY
        CELL   = 0.75 * DIAL_R
        row_y  = lambda r: (n / 2 - r) * 1.2

        # ---------------- helper: Grant-style arrow3D ------------------
        def make_arrow3d(start_mob, end_mob, thickness=0.015, height=0.30, color=WHITE):
            start, end = start_mob.get_center(), end_mob.get_center()
            arrow = Arrow3D(start, end, thickness=thickness, height=height, color=color)
            arrow.apply_depth_test()
            arrow.set_flat_stroke(False)
            return arrow

        # ---------------- QUERY column -------------
        R_w, R_a, R_d = VGroup(), VGroup(), VGroup()
        for i, w in enumerate(words):
            y = row_y(i)
            word = Text(w, weight=BOLD if i == focus else NORMAL)
            word.move_to([X_QARR, y, 0])
            dial = Dial(q_sim[i], color=YELLOW).move_to([X_Q, y, 0])
            R_w.add(word)
            R_d.add(dial)
            R_a.add(make_arrow3d(word, dial))
        self.add(R_w, R_a, R_d)

        # ---------------- K′ header row (RED) ------
        top_y = row_y(n)                              # 6th row (just below grid)
        Kp_w, Kp_a, Kp_d = VGroup(), VGroup(), VGroup()
        for j, w in enumerate(words):
            z = (j - (n - 1) / 2) * GRID_GAP
            word = Text(w, color=RED).move_to([X_KARR, top_y, z])
            dial = Dial(k_sim[j], color=RED).move_to([X_KEY, top_y, z])
            Kp_w.add(word)
            Kp_d.add(dial)
            Kp_a.add(make_arrow3d(word, dial))
        for g in (Kp_w, Kp_a, Kp_d):
            g.set_opacity(0.5)
        self.add(Kp_w, Kp_a, Kp_d)

        # ---------------- K footer column (BLUE) ---
        bot_z = (n + 1) / 2 * GRID_GAP                # 6th col (just beyond grid)
        K_w, K_a, K_d = VGroup(), VGroup(), VGroup()
        for i, w in enumerate(words):
            y = row_y(i)
            word = Text(w, color=BLUE).move_to([X_KARR, y, bot_z])
            dial = Dial(k_sim[i], color=BLUE).move_to([X_KEY, y, bot_z])
            K_w.add(word)
            K_d.add(dial)
            K_a.add(make_arrow3d(word, dial))
        for g in (K_w, K_a, K_d):
            g.set_opacity(0.5)
        self.add(K_w, K_a, K_d)

        # ---------------- 5×5 K×K′ dial sheet ------
        prod_dials = VGroup()
        for i, j in product(range(n), repeat=2):
            s = float(k_outer[i, j])                            # scalar value
            dial = Dial(s, color=PURPLE)                        # violet = product
            dial.rotate(PI / 2, axis=RIGHT)
            dial.move_to([X_GRID, row_y(i), (j - (n - 1) / 2) * GRID_GAP])
            dial.scalar = s                                     # store scalar
            prod_dials.add(dial)
        self.add(prod_dials)

        # ---------------- sparse pipes (Line3D) -------------
        THR, MIN_W, MAX_W = 0.15, 0.6, 145            # 1.8 × dynamic range
        pipes = VGroup()
        for q in range(n):
            for i, j in product(range(n), repeat=2):
                w = k_outer[i, j]
                if w < THR:
                    continue
                y, z = row_y(i), (j - (n - 1) / 2) * GRID_GAP
                s = [X_GRID + CELL / 2, y, z]
                e = [X_Q - 0.30, row_y(q), 0]
                width = MIN_W + (MAX_W - MIN_W) * (w ** 2.8)
                op    = 0.80 if q == focus else 0.024  # 80 % / 2.4 %
                pipe = Line3D(s, e, thickness=0.015, stroke_color=TEAL,
                             stroke_width=width, stroke_opacity=op)
                pipe.apply_depth_test()
                pipe.set_flat_stroke(False)
                pipe.insert_n_curves(20)
                pipes.add(pipe)
        self.add(pipes)

        # ---------------- flowing dots -------------
        def dots():
            t = self.time * 0.25
            vg = VGroup()
            for k, p in enumerate(pipes):
                d = Dot(radius=0.01 * p.get_stroke_width(),
                        fill_color=TEAL, stroke_width=0)
                d.set_fill(opacity=[float(p.get_stroke_opacity())])
                d.move_to(p.point_from_proportion((t + 0.05 * k) % 1))
                vg.add(d)
            return vg
        self.add(always_redraw(dots))

        # ---------------- camera -------------------
        self.camera.frame.set_euler_angles(phi=55 * DEGREES, theta=20 * DEGREES)
        self.wait(8)
