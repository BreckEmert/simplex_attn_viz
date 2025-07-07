from manim_imports_ext import *
import numpy as np


class Dial(VGroup):
    """Yellow needle at angle determined by s ∈ [0, 1]."""
    def __init__(self, s: float, radius: float = 0.375, **kwargs):
        super().__init__(**kwargs)
        needle = Line(ORIGIN, radius * RIGHT, stroke_width=4, stroke_color=YELLOW)
        needle.rotate(interpolate(-PI / 2, PI / 2, s))
        self.add(needle)


class SelfAttentionFlow(Scene):
    def construct(self):
        # ---------- sentence & layout ----------
        words = ["The", "quick", "brown", "fox", "jumps"]
        n = len(words)
        highlight_j = n - 1          # focus on last word

        # dial/arrow geometry
        DIAL_R, GAP_AD, WORD_GAP, ARR_LEN = 0.375, 0.126, 0.25, 0.576
        X_KEY_DIAL, X_QUERY_DIAL = -2, 2
        X_KEY_ARROW = X_KEY_DIAL - (DIAL_R + GAP_AD + ARR_LEN)
        X_QUERY_ARROW = X_QUERY_DIAL + (DIAL_R + GAP_AD + ARR_LEN)
        PIPE_OFF = 0.30

        # pipe‐width parameters
        BASE_W, MIN_W, TEMP = 45, 0.6, 8.0

        # dial angles
        key_sims   = np.array([.15, 0.48, 0.25, 0.6, 0.5])
        query_sims = np.array([0.20, 0.6, 0.9, 0.30, 0.75])

        # ---------- helpers ----------
        def mk_arrow(x0, x1, y):
            return Arrow([x0, y, 0], [x1, y, 0], buff=0, stroke_width=4)

        def mk_dial(s, x, y):
            return Dial(s).move_to([x, y, 0])

        # ---------- build left/right columns ----------
        row_y = lambda r: (n / 2 - r) * 1.2
        Lw = VGroup(); La = VGroup(); Ld = VGroup()
        Rw = VGroup(); Ra = VGroup(); Rd = VGroup()

        for i, w in enumerate(words):
            y = row_y(i)
            # keys
            Ld.add(mk_dial(key_sims[i], X_KEY_DIAL, y))
            La.add(mk_arrow(X_KEY_ARROW, X_KEY_DIAL - (DIAL_R + GAP_AD), y))
            Lw.add(Text(w).next_to(La[-1], LEFT, WORD_GAP).set_y(y))
            # queries
            Rd.add(mk_dial(query_sims[i], X_QUERY_DIAL, y))
            Ra.add(mk_arrow(X_QUERY_ARROW, X_QUERY_DIAL + (DIAL_R + GAP_AD), y))
            style = BOLD if i == highlight_j else NORMAL
            Rw.add(Text(w, weight=style).next_to(Ra[-1], RIGHT, WORD_GAP).set_y(y))

        self.add(Lw, La, Ld, Rd, Ra, Rw)

        # ---------- soft-max attention weights ----------
        θk = np.array([interpolate(-PI / 2, PI / 2, t) for t in key_sims])
        θq = np.array([interpolate(-PI / 2, PI / 2, t) for t in query_sims])
        cos_mtx = (np.cos(θk[:, None] - θq[None, :]) + 1) / 2
        logits = cos_mtx * TEMP
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        attn = exp / exp.sum(axis=0, keepdims=True)    # shape (n, n)

        # ---------- pipes ----------
        pipes = VGroup()
        for i in range(n):
            for j in range(n):
                y_i, y_j = row_y(i), row_y(j)
                s = [X_KEY_DIAL + PIPE_OFF, y_i, 0]
                e = [X_QUERY_DIAL - PIPE_OFF, y_j, 0]
                width = MIN_W + (BASE_W - MIN_W) * attn[i, j]
                pipes.add(Line(
                    s, e,
                    stroke_width=width,
                    stroke_color=TEAL,
                    stroke_opacity=1.0 if j == highlight_j else 0.07,
                ))
        self.add(pipes)

        # ---------- flowing dots ----------
        def flowing_dots():
            t = self.time * 0.25
            dots = VGroup()
            for k, p in enumerate(pipes):
                d = Dot(radius=0.01 * p.get_stroke_width(),
                        fill_color=p.get_stroke_color(),
                        stroke_width=0)
                d.set_fill(opacity=float(p.get_stroke_opacity()))
                d.move_to(p.point_from_proportion((t + 0.06 * k) % 1))
                dots.add(d)
            return dots
        self.add(always_redraw(flowing_dots))

        # ---------- styling ----------
        VGroup(*Lw, *Rw[:-1]).set_fill(opacity=0.25)

        # ---------- one-shot downward shift ----------
        self.camera.frame.shift(DOWN * -0.5)

        self.wait(7.97)
