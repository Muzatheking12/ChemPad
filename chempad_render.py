import math
import re

import numpy as np
from PySide6 import QtCore, QtGui

Qt = QtCore.Qt

BOND_LEN = 14.4
LINE_W = 0.6
BOLD_W = 2.0
MARGIN_W = 1.3
HASH_SP = 2.5
BOND_SP = 0.18 * BOND_LEN
WEDGE_W = 3.6
LABEL_PT = 10.0
SUB_PT = 7.0
SUB_DROP = 2.0
SUP_RAISE = 3.6
LINE_SPACING = 1.2

HEAD_LEN = 12.0
HEAD_HW = 3.4
HEAD_NOTCH = 0.90
CURVE_HEAD_LEN = 7.0
CURVE_HEAD_HW = 2.1
ARROW_LEN = 72.0
PLUS_HALF = 3.4

FONT_FAMILIES = ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"]

GENERIC = ("R", "R1", "R2", "R3", "R4", "X", "Y", "Z")

STEREO_COLOR = QtGui.QColor(255, 178, 51)
SEL_COLOR = QtGui.QColor(235, 51, 51)
HOVER_COLOR = QtGui.QColor(255, 191, 0)

LABEL_COLORS = {
    "N": (26, 51, 242), "O": (230, 13, 13), "S": (217, 173, 0), "F": (0, 184, 204),
    "Cl": (13, 173, 13), "Br": (153, 77, 13), "I": (128, 26, 191), "P": (255, 128, 0),
}
LABEL_COLORS_DARK = {
    "N": (89, 128, 255), "O": (255, 64, 64), "S": (255, 230, 26), "F": (51, 242, 255),
    "Cl": (51, 230, 51), "Br": (217, 128, 51), "I": (191, 102, 255), "P": (255, 153, 51),
}


def split_el(el):
    m = re.match(r"^([A-Z][a-z]?)([+-])(\d?)$", el or "")
    if m:
        q = int(m.group(3)) if m.group(3) else 1
        return m.group(1), (q if m.group(2) == "+" else -q)
    return el, 0


def base(el):
    return split_el(el)[0]


def el_str(sym, charge):
    if charge == 0:
        return sym
    return sym + ("+" if charge > 0 else "-") + (str(abs(charge)) if abs(charge) > 1 else "")


def display_el(el):
    sym, q = split_el(el)
    if q == 0:
        return sym
    return sym + (str(abs(q)) if abs(q) > 1 else "") + ("+" if q > 0 else "-")


def is_generic(el):
    return base(el) in GENERIC


_FONT_CACHE = {}
_FM = {}


def make_font(px, bold=False):
    key = (int(round(px)), bool(bold))
    f = _FONT_CACHE.get(key)
    if f is None:
        f = QtGui.QFont()
        f.setFamilies(FONT_FAMILIES)
        f.setPixelSize(max(1, key[0]))
        f.setBold(bool(bold))
        f.setHintingPreference(QtGui.QFont.PreferNoHinting)
        f.setStyleStrategy(QtGui.QFont.PreferAntialias)
        _FONT_CACHE[key] = f
    return f


def _fm(bold=False):
    m = _FM.get(bold)
    if m is None:
        m = QtGui.QFontMetricsF(make_font(1000, bold))
        _FM[bold] = m
    return m


def text_w(s, size, bold=False):
    return _fm(bold).horizontalAdvance(s) * size / 1000.0


def cap_h(size, bold=False):
    c = _fm(bold).capHeight()
    if c <= 0:
        c = 716.0
    return c * size / 1000.0


class Label:
    """Laid-out atom label. All numbers are relative to the atom centre (model units)."""
    __slots__ = ("runs", "rect", "circ", "boxes")

    def __init__(self):
        self.runs = []
        self.boxes = []
        self.rect = (0.0, 0.0, 0.0, 0.0)
        self.circ = None


def make_label(el, nh, nbr, orient=None, circled=False):
    """nbr: list of (dx, dy) vectors to the bonded neighbours."""
    sym, q = split_el(el)
    deg = len(nbr)
    if sym == "C" and deg > 0:
        return None
    F, S = LABEL_PT, SUB_PT
    cap = cap_h(F)
    base_txt, digits = sym, ""
    m = re.match(r"^([A-Z][a-z]?)(\d+)$", sym)
    if m and sym in GENERIC:
        base_txt, digits = m.group(1), m.group(2)
    heavy = [(base_txt, F, 0.0, True)]
    if digits:
        heavy.append((digits, S, SUB_DROP, True))
    hs = []
    if nh > 0 and sym not in GENERIC and sym != "H":
        hs = [("H", F, 0.0, False)]
        if nh > 1:
            hs.append((str(nh), S, SUB_DROP, False))
    base_y = cap / 2.0
    lab = Label()
    plain_chg = None
    if q:
        plain_chg = ("+" if q > 0 else "−")
        if abs(q) > 1:
            plain_chg = str(abs(q)) + plain_chg
    use_circle = bool(circled and abs(q) == 1)

    runs = []
    if hs and orient == "v":
        up = (sum(dy for _, dy in nbr) / deg > 0.5) if deg else False
        lh = cap + 1.9
        hy = base_y - lh if up else base_y + lh
        x = -sum(text_w(t, s) for t, s, _, _ in heavy) / 2.0
        hx0 = x
        for t, s, dy, _ in heavy:
            runs.append((t, s, x, base_y + dy))
            x += text_w(t, s)
        x_end = x
        if plain_chg and not use_circle:
            runs.append((plain_chg, S, x, base_y - SUP_RAISE))
            x_end = x + text_w(plain_chg, S)
        xh = -sum(text_w(t, s) for t, s, _, _ in hs) / 2.0
        for t, s, dy, _ in hs:
            runs.append((t, s, xh, hy + dy))
            xh += text_w(t, s)
    else:
        if hs:
            if deg:
                prefix = (sum(dx for dx, _ in nbr) / deg) > 0.5
            else:
                prefix = sym in ("O", "S", "Se", "F", "Cl", "Br", "I")
            seq = (hs + heavy) if prefix else (heavy + hs)
        else:
            seq = heavy
        x, hx0, hx1 = 0.0, None, None
        pos = []
        for t, s, dy, is_h in seq:
            w = text_w(t, s)
            if is_h:
                hx0 = x if hx0 is None else hx0
                hx1 = x + w
            pos.append((t, s, x, base_y + dy))
            x += w
        shift = -(hx0 + hx1) / 2.0
        runs = [(t, s, px + shift, py) for t, s, px, py in pos]
        x_end = x + shift
        if plain_chg and not use_circle:
            runs.append((plain_chg, S, x_end, base_y - SUP_RAISE))
            x_end += text_w(plain_chg, S)
    x0 = min(r[2] for r in runs)
    x1 = max(r[2] + text_w(r[0], r[1]) for r in runs)
    y0 = min(r[3] - cap_h(r[1]) for r in runs)
    y1 = max(r[3] for r in runs)
    lab.runs = runs
    lab.rect = (x0, y0, x1, y1)
    for t, sz, rx, ry in runs:
        tb = _fm().tightBoundingRect(t)
        k = sz / 1000.0
        lab.boxes.append((rx + tb.left() * k, ry + tb.top() * k, rx + tb.right() * k, ry + tb.bottom() * k))
    if use_circle:
        r = 2.7
        lab.circ = (x1 + r + 0.6, -cap / 2.0 - 0.7, r, 1 if q > 0 else -1)
    return lab


def ray_exit(rect, u, pad=0.0):
    """Distance from the origin (inside rect) to the padded rect boundary along unit vector u."""
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    t = 1e9
    if u[0] > 1e-9:
        t = min(t, x1 / u[0])
    elif u[0] < -1e-9:
        t = min(t, x0 / u[0])
    if u[1] > 1e-9:
        t = min(t, y1 / u[1])
    elif u[1] < -1e-9:
        t = min(t, y0 / u[1])
    return max(t, 0.0)


def clip_dist(boxes, u, margin=MARGIN_W):
    """Largest distance along u (from the atom centre) that is still within `margin` of a letter box."""
    ux, uy = float(u[0]), float(u[1])
    tmax = max(max(abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3])) for b in boxes) * 1.5 + margin + 2.0
    best, t = 0.0, 0.0
    m2 = margin * margin
    while t <= tmax:
        px, py = t * ux, t * uy
        for x0, y0, x1, y1 in boxes:
            dx = max(x0 - px, 0.0, px - x1)
            dy = max(y0 - py, 0.0, py - y1)
            if dx * dx + dy * dy <= m2:
                best = t
                break
        t += 0.1
    return best


def parse_markup(text, chem=True):
    """'H2O' / 'x_{2}' / 'y^{+}' -> lines of (text, mode) with mode in n / sub / sup."""
    lines = []
    for raw in str(text).split("\n"):
        runs, buf, i = [], "", 0

        def flush():
            nonlocal buf
            if buf:
                runs.append((buf, "n"))
                buf = ""

        n = len(raw)
        while i < n:
            ch = raw[i]
            if ch in "_^" and i + 1 < n:
                mode = "sub" if ch == "_" else "sup"
                if raw[i + 1] == "{":
                    j = raw.find("}", i + 2)
                    j = n if j < 0 else j
                    seg, i = raw[i + 2:j], j + 1
                else:
                    seg, i = raw[i + 1], i + 2
                flush()
                runs.append((seg, mode))
                continue
            if chem and ch.isdigit() and i > 0 and (raw[i - 1].isalpha() or raw[i - 1] == ")"
                                                      or (runs and runs[-1][1] == "sub" and not buf)):
                j = i
                while j < n and raw[j].isdigit():
                    j += 1
                flush()
                runs.append((raw[i:j], "sub"))
                i = j
                continue
            if chem and ch in "+-" and i > 0 and (raw[i - 1].isalnum() or raw[i - 1] == ")") \
                    and (i + 1 == n or raw[i + 1] in " ,;)") and not raw[i - 1].isspace():
                flush()
                runs.append(("−" if ch == "-" else "+", "sup"))
                i += 1
                continue
            buf += ch
            i += 1
        flush()
        lines.append(runs)
    return lines


def text_layout(text, size, chem=True, anchor="c", bold=False):
    """Returns (lines, bbox) relative to the annotation anchor.
    lines: list of list of (text, px size, x, baseline)."""
    parsed = parse_markup(text, chem)
    cap = cap_h(size, bold)
    lh = size * LINE_SPACING
    out, widths = [], []
    for runs in parsed:
        w = 0.0
        for t, mode in runs:
            w += text_w(t, size if mode == "n" else size * 0.7, bold)
        widths.append(w)
    nl = len(parsed)
    block_h = cap + (nl - 1) * lh
    if anchor == "b":
        top = -block_h
    elif anchor == "t":
        top = 0.0
    else:
        top = -block_h / 2.0
    for k, runs in enumerate(parsed):
        base_y = top + cap + k * lh
        x = -widths[k] / 2.0
        row = []
        for t, mode in runs:
            s = size if mode == "n" else size * 0.7
            dy = 0.0 if mode == "n" else (size * 0.2 if mode == "sub" else -size * 0.36)
            row.append((t, s, x, base_y + dy))
            x += text_w(t, s, bold)
        out.append(row)
    wmax = max(widths) if widths else 0.0
    return out, (-wmax / 2.0, top - size * 0.1, wmax / 2.0, top + block_h + size * 0.25)


def arrow_geometry(kind, p0, p1, pts=None, flip=False, lw=LINE_W):
    """Return (lines, polys, dashed) with lines = [(P, Q)] / polylines, polys = filled polygons."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1e-6:
        return [], [], False
    u = d / L
    n = np.array([-u[1], u[0]])
    lines, polys = [], []

    def head(tip, uu, nn, hl=HEAD_LEN, hw=HEAD_HW, half=0):
        notch = tip - uu * hl * HEAD_NOTCH
        if half == 0:
            return [tip, tip - uu * hl + nn * hw, notch, tip - uu * hl - nn * hw], notch
        return [tip, tip - uu * hl + nn * hw * half, notch], notch

    if kind in ("arrow", "dashed"):
        poly, notch = head(p1, u, n)
        polys.append(poly)
        lines.append([p0, notch])
        return lines, polys, kind == "dashed"
    if kind == "dbl":
        poly1, n1 = head(p1, u, n)
        poly2, n2 = head(p0, -u, n)
        polys += [poly1, poly2]
        lines.append([n2, n1])
        return lines, polys, False
    if kind == "eq":
        g = 1.9
        up = np.array([u[1], -u[0]])
        a0, a1 = p0 + up * g, p1 + up * g
        b0, b1 = p0 - up * g, p1 - up * g
        poly1, n1 = head(a1, u, up, 9.0, 3.2, half=1)
        poly2, n2 = head(b0, -u, -up, 9.0, 3.2, half=1)
        polys += [poly1, poly2]
        lines.append([a0, n1])
        lines.append([b1, n2])
        return lines, polys, False
    if kind == "retro":
        o = 1.7
        hl, hw = 9.0, 4.6
        lines.append([p0 + n * o, p1 - u * hl * 0.15 + n * o - u * hl * 0.0])
        lines.append([p0 - n * o, p1 - u * hl * 0.15 - n * o])
        lines.append([p1 - u * hl + n * hw, p1, p1 - u * hl - n * hw])
        return lines, polys, False
    if kind in ("curve", "curve1"):
        c = np.asarray(pts[1], float) if pts is not None and len(pts) > 2 else (p0 + p1) / 2 + n * L * 0.35
        hl, hw = CURVE_HEAD_LEN, CURVE_HEAD_HW
        tang = p1 - c
        tl = float(np.hypot(*tang))
        tang = tang / tl if tl > 1e-6 else u
        tn = np.array([-tang[1], tang[0]])
        half = 0
        if kind == "curve1":
            half = -1 if flip else 1
        poly, notch = head(p1, tang, tn, hl, hw, half=half)
        polys.append(poly)
        ts = np.linspace(0.0, 1.0, 48)
        curve = [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t * t * p1 for t in ts]
        cut = len(curve)
        for k in range(len(curve) - 1, -1, -1):
            if np.hypot(*(curve[k] - p1)) >= hl * HEAD_NOTCH:
                cut = k + 1
                break
        shaft = curve[:cut] + [notch]
        lines.append(shaft)
        return lines, polys, False
    return [[p0, p1]], [], False


def bezier_points(p0, c, p1, n=24):
    p0, c, p1 = (np.asarray(v, float) for v in (p0, c, p1))
    return [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t * t * p1 for t in np.linspace(0, 1, n)]


def annot_bbox(rec, text_bold=False):
    """(x0, y0, x1, y1) of an annotation in model units."""
    if rec.kind == "text":
        _, bb = text_layout(rec.text, LABEL_PT * rec.scale, rec.chem, rec.anchor, text_bold)
        x, y = rec.pts[0]
        return (x + bb[0], y + bb[1], x + bb[2], y + bb[3])
    if rec.kind == "plus":
        x, y = rec.pts[0]
        h = PLUS_HALF * rec.scale
        return (x - h, y - h, x + h, y + h)
    pts = rec.pts
    if rec.kind.startswith("curve") and len(pts) > 2:
        pts = bezier_points(pts[0], pts[1], pts[2])
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    m = HEAD_HW + 2.0
    return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)


def _qc(c):
    if isinstance(c, QtGui.QColor):
        return c
    if isinstance(c, str):
        return QtGui.QColor(c)
    return QtGui.QColor(*c)


class SceneRenderer:
    """Paints a Sketch (atoms, bonds, annotations) with ACS-1996 proportions."""

    def __init__(self, sk, ui=False, dark=False, atom_ids=None, annot_ids=None):
        self.sk = sk
        self.ui = ui
        self.dark = dark
        self.fg = QtGui.QColor(255, 255, 255) if dark else QtGui.QColor(0, 0, 0)
        self.atom_ids = set(sk.atoms) if atom_ids is None else (set(atom_ids) & set(sk.atoms))
        self.annot_ids = set(sk.annots) if annot_ids is None else (set(annot_ids) & set(sk.annots))
        self.lw = LINE_W * sk.bond_scale
        self.prepare()

    def prepare(self):
        sk = self.sk
        self.pos = {i: np.array([sk.atoms[i].x, sk.atoms[i].y], float) for i in self.atom_ids}
        self.adj = {i: [] for i in self.atom_ids}
        self.bonds = {}
        for (a, b), o in sk.bonds.items():
            if a in self.pos and b in self.pos:
                self.bonds[(a, b)] = o
                self.adj[a].append(b)
                self.adj[b].append(a)
        self.labels = {}
        for i in self.atom_ids:
            at = sk.atoms[i]
            nbr = [tuple(self.pos[k] - self.pos[i]) for k in self.adj[i]]
            lab = make_label(at.el, sk.nh.get(i, 0), nbr, at.orient, sk.circled)
            if lab is not None:
                self.labels[i] = lab

    def atom_color(self, i):
        at = self.sk.atoms[i]
        if at.color:
            return _qc(at.color)
        if i in self.labels and not self.sk.mono_letters:
            tab = LABEL_COLORS_DARK if self.dark else LABEL_COLORS
            c = tab.get(base(at.el))
            if c:
                return QtGui.QColor(*c)
        return self.fg

    def bond_color(self, i):
        at = self.sk.atoms[i]
        return _qc(at.color) if at.color else self.fg

    def _clip(self, i, u):
        lab = self.labels.get(i)
        if lab is None:
            return 0.0
        return clip_dist(lab.boxes, u, MARGIN_W)

    def _free_angle(self, i, extra=()):
        occ = [math.atan2(*(self.pos[k] - self.pos[i])[::-1]) for k in self.adj[i]] + list(extra)
        if not occ:
            return -math.pi / 2
        occ = sorted(occ)
        best, best_gap = None, -1.0
        for k, a in enumerate(occ):
            b = occ[(k + 1) % len(occ)]
            gap = (b - a) % (2 * math.pi) if len(occ) > 1 else 2 * math.pi
            if gap > best_gap:
                best_gap, best = gap, a + gap / 2
        return best

    def lone_pair_angles(self, i, k):
        occ = [math.atan2(*(self.pos[j] - self.pos[i])[::-1]) for j in self.adj[i]]
        lab = self.labels.get(i)
        if lab is not None and lab.circ is not None:
            occ += [math.radians(-58), math.radians(-12)]
        if k <= 0:
            return []
        if len(occ) == 0:
            return [-math.pi / 2 + j * 2 * math.pi / k for j in range(k)] if k > 1 else [-math.pi / 2]
        if len(occ) == 1:
            return [occ[0] + 2 * math.pi * (j + 1) / (k + 1) for j in range(k)]
        out = []
        for _ in range(k):
            occ_all = sorted(occ + out)
            best, best_gap = None, -1.0
            for q, v in enumerate(occ_all):
                w = occ_all[(q + 1) % len(occ_all)]
                gap = (w - v) % (2 * math.pi)
                if gap > best_gap:
                    best_gap, best = gap, v + gap / 2
            out.append(best)
        return out

    def ring_of(self, a, b):
        best = None
        for r in self.sk.rings:
            if a in r and b in r and all(x in self.pos for x in r):
                if best is None or len(r) < len(best):
                    best = r
        return best

    def _sides(self, x, other, fwd, n):
        out = []
        for k in self.adj[x]:
            if k == other:
                continue
            w = self.pos[k] - self.pos[x]
            wl = float(np.hypot(*w))
            if wl < 1e-6:
                continue
            w = w / wl
            s = float(np.dot(w, n))
            if abs(s) < 0.12:
                continue
            phi = math.acos(max(-1.0, min(1.0, float(np.dot(w, fwd)))))
            out.append((1 if s > 0 else -1, phi))
        return out

    def bond_lines(self, a, b, order):
        """[(P, Q, t0, t1)] drawing segments in model coordinates."""
        A, B = self.pos[a], self.pos[b]
        d = B - A
        L = float(np.hypot(*d))
        if L < 1e-6:
            return []
        u = d / L
        n = np.array([-u[1], u[0]])
        la, lb = a in self.labels, b in self.labels
        ea = self._clip(a, u) if la else 0.0
        eb = self._clip(b, -u) if lb else 0.0

        def seg(off, ta, tb):
            s0 = ea if la else ta
            s1 = L - (eb if lb else tb)
            if s1 - s0 < 0.3:
                return None
            return (A + u * s0 + n * off, A + u * s1 + n * off)

        out = []
        if order == 1:
            out.append(seg(0.0, 0.0, 0.0))
        elif order == 3:
            for off in (-BOND_SP, 0.0, BOND_SP):
                out.append(seg(off, 0.0, 0.0))
        else:
            sa = self._sides(a, b, u, n)
            sb = self._sides(b, a, -u, n)
            ring = self.ring_of(a, b)
            style, sign = "c", 1
            if ring is not None:
                cen = np.mean([self.pos[x] for x in ring], axis=0)
                sign = 1 if float(np.dot(cen - (A + B) / 2, n)) > 0 else -1
                style = "s"
            elif len(self.adj[a]) == 1 or len(self.adj[b]) == 1:
                other = sb if len(self.adj[a]) == 1 else sa
                if other and len({s for s, _ in other}) == 1:
                    style, sign = "s", other[0][0]
                else:
                    style = "c"
            else:
                cnt = {1: 0, -1: 0}
                for s, _ in sa + sb:
                    cnt[s] += 1
                both_a = {s for s, _ in sa} == {1, -1}
                both_b = {s for s, _ in sb} == {1, -1}
                single = sb if both_a else sa
                if (both_a and both_b) or cnt[1] + cnt[-1] == 0:
                    style = "c"
                elif (both_a or both_b) and single:
                    style, sign = "s", single[0][0]
                else:
                    style = "s"
                    if cnt[1] != cnt[-1]:
                        sign = 1 if cnt[1] > cnt[-1] else -1
                    else:
                        first = sa if (sa and a < b) or not sb else sb
                        sign = first[0][0] if first else 1
            if style == "s":
                d_off = sign * BOND_SP
                out.append(seg(0.0, 0.0, 0.0))

                def bis(sides):
                    ph = [p for s, p in sides if s == sign]
                    if not ph:
                        return 0.0
                    t = BOND_SP / math.tan(min(ph) / 2.0)
                    return max(0.0, min(t, 0.4 * L))

                out.append(seg(d_off, bis(sa), bis(sb)))
            else:
                h = BOND_SP / 2.0

                def miter(sides, s):
                    if {sd for sd, _ in sides} != {1, -1}:
                        return 0.0
                    ph = [p for sd, p in sides if sd == s]
                    if not ph:
                        return 0.0
                    ph = min(ph)
                    t = h / math.tan(ph) if abs(math.tan(ph)) > 1e-6 else 0.0
                    return max(-2.5, min(t, 0.4 * L))

                for s in (1, -1):
                    out.append(seg(s * h, miter(sa, s), miter(sb, s)))
        return [o for o in out if o is not None]

    def draw(self, p):
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
                         | QtGui.QPainter.SmoothPixmapTransform)
        if self.ui:
            self._underlay(p)
        self._draw_bonds(p)
        self._draw_labels(p)
        self._draw_charges_and_pairs(p)
        self._draw_stereo(p)
        for i in sorted(self.annot_ids):
            self._draw_annot(p, self.sk.annots[i])
        if self.ui:
            self._overlay(p)

    def _pen(self, color, w, cap=Qt.RoundCap):
        pen = QtGui.QPen(_qc(color), w)
        pen.setCapStyle(cap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def _line(self, p, P, Q, ca, cb, w):
        if ca == cb:
            p.setPen(self._pen(ca, w))
            p.drawLine(QtCore.QPointF(*P), QtCore.QPointF(*Q))
        else:
            M = (P + Q) / 2
            p.setPen(self._pen(ca, w))
            p.drawLine(QtCore.QPointF(*P), QtCore.QPointF(*M))
            p.setPen(self._pen(cb, w))
            p.drawLine(QtCore.QPointF(*M), QtCore.QPointF(*Q))

    def _draw_bonds(self, p):
        sk = self.sk
        items = sorted(self.bonds.items())
        z = {i: sk.atoms[i].z for i in self.atom_ids}
        three = [i for i in self.atom_ids if sk.atoms[i].g3 is not None]
        if three:
            zs = [z[i] for i in three]
            zmin, span = min(zs), max(max(zs) - min(zs), 1e-6)
            items.sort(key=lambda it: -(z[it[0][0]] + z[it[0][1]]) / 2.0)
        bg = QtGui.QColor(0, 0, 0) if self.dark else QtGui.QColor(255, 255, 255)
        for (a, b), o in items:
            ca, cb = self.bond_color(a), self.bond_color(b)
            is3 = sk.atoms[a].g3 is not None or sk.atoms[b].g3 is not None
            wi = sk.wedge_info.get((a, b)) if sk.stereo_mode == 2 else None
            if wi and o == 1 and not is3:
                self._draw_wedge(p, a, b, wi)
                continue
            w = self.lw
            lines = self.bond_lines(a, b, o)
            if is3:
                t = 0.5 - ((z[a] + z[b]) / 2.0 - zmin) / span
                w = self.lw * (1.05 + 0.5 * t)
                for P, Q in lines:
                    d = Q - P
                    L = float(np.hypot(*d))
                    if L > 3.4:
                        u = d / L
                        p.setPen(self._pen(bg, self.lw * 4.0, Qt.FlatCap))
                        p.drawLine(QtCore.QPointF(*(P + u * 1.7)), QtCore.QPointF(*(Q - u * 1.7)))
            for P, Q in lines:
                self._line(p, P, Q, ca, cb, w)

    def _draw_wedge(self, p, a, b, wi):
        kind, beg = wi
        end = b if beg == a else a
        A, B = self.pos[beg], self.pos[end]
        d = B - A
        L = float(np.hypot(*d))
        if L < 1e-6:
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        s0 = self._clip(beg, u) if beg in self.labels else 0.0
        s1 = L - (self._clip(end, -u) if end in self.labels else 0.0)
        if s1 - s0 < 0.5:
            return
        W = WEDGE_W * max(self.sk.bond_scale, 0.6) ** 0.5
        col = self.bond_color(beg)
        if kind == "wedge":
            w0, w1 = W * s0 / L, W * s1 / L
            poly = QtGui.QPolygonF([QtCore.QPointF(*(A + u * s0 + n * w0 / 2)),
                                    QtCore.QPointF(*(A + u * s1 + n * w1 / 2)),
                                    QtCore.QPointF(*(A + u * s1 - n * w1 / 2)),
                                    QtCore.QPointF(*(A + u * s0 - n * w0 / 2))])
            p.setPen(self._pen(col, 0.2, Qt.FlatCap))
            p.setBrush(col)
            p.drawPolygon(poly)
            p.setBrush(Qt.NoBrush)
        else:
            p.setPen(self._pen(col, self.lw, Qt.FlatCap))
            s = max(s0, HASH_SP * 0.6)
            while s <= s1 + 1e-6:
                w = W * s / L
                c = A + u * s
                p.drawLine(QtCore.QPointF(*(c + n * w / 2)), QtCore.QPointF(*(c - n * w / 2)))
                s += HASH_SP

    def _draw_labels(self, p):
        for i, lab in self.labels.items():
            c = self.atom_color(i)
            ax, ay = self.pos[i]
            p.setPen(c)
            for t, s, x, y in lab.runs:
                p.setFont(make_font(s))
                p.drawText(QtCore.QPointF(ax + x, ay + y), t)

    def _draw_charges_and_pairs(self, p):
        sk = self.sk
        for i in self.atom_ids:
            at = sk.atoms[i]
            sym, q = split_el(at.el)
            lab = self.labels.get(i)
            col = self.atom_color(i)
            ax, ay = self.pos[i]
            used = []
            if lab is not None and lab.circ is not None:
                cx, cy, r, sg = lab.circ
                self._circled(p, ax + cx, ay + cy, r, sg, col)
            elif lab is None and q:
                ang = self._free_angle(i)
                used.append(ang)
                dist = 5.4
                cx, cy = ax + dist * math.cos(ang), ay + dist * math.sin(ang)
                if sk.circled and abs(q) == 1:
                    self._circled(p, cx, cy, 2.7, 1 if q > 0 else -1, col)
                else:
                    txt = ("+" if q > 0 else "−")
                    if abs(q) > 1:
                        txt = str(abs(q)) + txt
                    p.setPen(col)
                    p.setFont(make_font(SUB_PT))
                    w = text_w(txt, SUB_PT)
                    p.drawText(QtCore.QPointF(cx - w / 2, cy + cap_h(SUB_PT) / 2), txt)
            if sk.show_lp and sk.lp.get(i, 0) > 0 and not is_generic(at.el):
                for ang in self.lone_pair_angles(i, sk.lp[i]):
                    u = np.array([math.cos(ang), math.sin(ang)])
                    dist = (ray_exit(lab.rect, u, 0.0) + 2.6) if lab is not None else 4.6
                    c = self.pos[i] + u * dist
                    perp = np.array([-u[1], u[0]]) * 1.25
                    p.setPen(Qt.NoPen)
                    p.setBrush(col)
                    for s in (1, -1):
                        q2 = c + perp * s
                        p.drawEllipse(QtCore.QPointF(*q2), 0.65, 0.65)
                    p.setBrush(Qt.NoBrush)

    def _circled(self, p, cx, cy, r, sign, col):
        p.setPen(self._pen(col, 0.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        k = r * 0.58
        p.drawLine(QtCore.QPointF(cx - k, cy), QtCore.QPointF(cx + k, cy))
        if sign > 0:
            p.drawLine(QtCore.QPointF(cx, cy - k), QtCore.QPointF(cx, cy + k))

    def _draw_stereo(self, p):
        sk = self.sk
        if sk.stereo_mode != 1:
            return
        col = STEREO_COLOR if self.ui else self.fg
        p.setFont(make_font(8))
        p.setPen(col)
        cap = cap_h(8)
        for aid, c in sk.centers.items():
            if aid not in self.pos or c not in ("R", "S", "?"):
                continue
            txt = "(?)" if c == "?" else f"({c})"
            ang = self._free_angle(aid)
            lab = self.labels.get(aid)
            dist = (ray_exit(lab.rect, np.array([math.cos(ang), math.sin(ang)])) if lab else 0.0) + 8.0
            cx = self.pos[aid][0] + dist * math.cos(ang)
            cy = self.pos[aid][1] + dist * math.sin(ang)
            w = text_w(txt, 8)
            p.drawText(QtCore.QPointF(cx - w / 2, cy + cap / 2), txt)
        for (a, b), c in sk.ez.items():
            if a not in self.pos or b not in self.pos or (a, b) not in self.bonds:
                continue
            A, B = self.pos[a], self.pos[b]
            d = B - A
            L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            u = d / L
            n = np.array([-u[1], u[0]])
            sa = self._sides(a, b, u, n)
            sb = self._sides(b, a, -u, n)
            cnt = sum(s for s, _ in sa + sb)
            side = -1 if cnt > 0 else 1
            offs = [float(np.dot((ln[0] + ln[1]) / 2 - (A + B) / 2, n)) for ln in self.bond_lines(a, b, self.bonds[(a, b)])]
            if offs and abs(sum(offs) / len(offs)) > 0.5:
                side = -1 if sum(offs) > 0 else 1
            txt = f"({c})"
            w, h = text_w(txt, 8), cap
            reach = max([abs(o) for o in offs] + [0.0])
            ext = abs(n[0]) * w / 2 + abs(n[1]) * h / 2
            mid = (A + B) / 2 + n * side * (reach + ext + 2.4)
            p.drawText(QtCore.QPointF(mid[0] - w / 2, mid[1] + cap / 2), txt)

    def annot_color(self, rec):
        if self.ui and rec.id in self.sk.sel_annots:
            return SEL_COLOR
        return _qc(rec.color) if rec.color else self.fg

    def _draw_annot(self, p, rec):
        col = self.annot_color(rec)
        lw = self.lw
        if rec.kind == "text":
            lines, _ = text_layout(rec.text, LABEL_PT * rec.scale, rec.chem, rec.anchor, self.sk.text_bold)
            p.setPen(col)
            for row in lines:
                for t, s, x, y in row:
                    p.setFont(make_font(s, self.sk.text_bold))
                    p.drawText(QtCore.QPointF(rec.pts[0][0] + x, rec.pts[0][1] + y), t)
        elif rec.kind == "plus":
            c = rec.pts[0]
            h = PLUS_HALF * rec.scale
            p.setPen(self._pen(col, max(lw, 0.8)))
            p.drawLine(QtCore.QPointF(c[0] - h, c[1]), QtCore.QPointF(c[0] + h, c[1]))
            p.drawLine(QtCore.QPointF(c[0], c[1] - h), QtCore.QPointF(c[0], c[1] + h))
        else:
            lines, polys, dashed = arrow_geometry(rec.kind, rec.pts[0], rec.pts[-1], rec.pts, rec.flip)
            pen = self._pen(col, lw, Qt.FlatCap)
            if dashed:
                pen.setStyle(Qt.CustomDashLine)
                pen.setDashPattern([4.0, 2.5])
            p.setPen(pen)
            for pl in lines:
                path = QtGui.QPainterPath(QtCore.QPointF(*pl[0]))
                for q in pl[1:]:
                    path.lineTo(QtCore.QPointF(*q))
                p.drawPath(path)
            p.setPen(self._pen(col, 0.15, Qt.FlatCap))
            p.setBrush(col)
            for poly in polys:
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*v) for v in poly]))
            p.setBrush(Qt.NoBrush)

    def _underlay(self, p):
        sk = self.sk
        p.setPen(Qt.NoPen)
        for (a, b) in self.bonds:
            if a in sk.selected and b in sk.selected:
                p.setPen(self._pen(QtGui.QColor(235, 51, 51, 90), 5.0))
                p.drawLine(QtCore.QPointF(*self.pos[a]), QtCore.QPointF(*self.pos[b]))
        p.setPen(Qt.NoPen)
        for i in sk.selected:
            if i in self.pos:
                p.setBrush(QtGui.QColor(235, 51, 51, 105))
                p.drawEllipse(QtCore.QPointF(*self.pos[i]), 5.4, 5.4)
        if sk.hover_id in self.pos:
            p.setBrush(QtGui.QColor(255, 191, 0, 110))
            p.drawEllipse(QtCore.QPointF(*self.pos[sk.hover_id]), 6.2, 6.2)
        if sk.hover_bond in self.bonds:
            a, b = sk.hover_bond
            p.setPen(self._pen(QtGui.QColor(255, 191, 0, 120), 5.0))
            p.drawLine(QtCore.QPointF(*self.pos[a]), QtCore.QPointF(*self.pos[b]))
        p.setBrush(Qt.NoBrush)

    def _overlay(self, p):
        sk = self.sk
        pv = sk.preview
        if pv:
            P0, P1, bad, el = pv
            p.setPen(self._pen(QtGui.QColor(230, 90, 40) if bad else QtGui.QColor(51, 102, 230), 1.0))
            p.drawLine(QtCore.QPointF(*P0), QtCore.QPointF(*P1))
            if el:
                p.setBrush(QtGui.QColor(230, 40, 40, 90) if bad else QtGui.QColor(120, 120, 120, 90))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(*P1), 4.5, 4.5)
                p.setBrush(Qt.NoBrush)
        if sk.rband:
            x0, y0, x1, y1 = sk.rband
            pen = QtGui.QPen(QtGui.QColor(255, 150, 25), 0.0)
            pen.setStyle(Qt.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawRect(QtCore.QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))
        if len(sk.sel_annots) == 1:
            rec = sk.annots.get(next(iter(sk.sel_annots)))
            if rec is not None and rec.kind not in ("text", "plus"):
                p.setBrush(QtGui.QColor(255, 255, 255))
                p.setPen(self._pen(SEL_COLOR, 0.6))
                for k, q in enumerate(rec.pts):
                    r = 1.9 if (k == 1 and rec.kind.startswith("curve")) else 2.4
                    p.drawRect(QtCore.QRectF(q[0] - r, q[1] - r, 2 * r, 2 * r))
                p.setBrush(Qt.NoBrush)

    def annot_bbox(self, rec):
        return annot_bbox(rec, self.sk.text_bold)

    def bounds(self):
        xs0, ys0, xs1, ys1 = [], [], [], []
        for i in self.atom_ids:
            x, y = self.pos[i]
            lab = self.labels.get(i)
            r = lab.rect if lab else (-0.5, -0.5, 0.5, 0.5)
            xs0.append(x + r[0]); ys0.append(y + r[1]); xs1.append(x + r[2]); ys1.append(y + r[3])
            ext = 0.0
            if lab is not None and lab.circ is not None:
                ext = max(ext, lab.circ[0] + lab.circ[2] - r[2])
            sk = self.sk
            if sk.show_lp and sk.lp.get(i, 0) > 0:
                ext = max(ext, 6.0)
            if split_el(sk.atoms[i].el)[1] and lab is None:
                ext = max(ext, 8.0)
            if ext:
                xs0.append(x - ext - 2); ys0.append(y - ext - 2); xs1.append(x + ext + 2); ys1.append(y + ext + 2)
        for i in self.annot_ids:
            bb = self.annot_bbox(self.sk.annots[i])
            xs0.append(bb[0]); ys0.append(bb[1]); xs1.append(bb[2]); ys1.append(bb[3])
        if not xs0:
            return None
        return (min(xs0), min(ys0), max(xs1), max(ys1))
