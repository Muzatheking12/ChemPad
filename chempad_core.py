import math
import re
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import vtk

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

PRIMARY_ELEMENTS = ["C", "N", "O", "S"]
MORE_ELEMENTS = ["H", "P", "F", "Cl", "Br", "I"]

COLORS = {
    "H": (0.78, 0.78, 0.82), "C": (0.35, 0.35, 0.38), "N": (0.25, 0.4, 1.0),
    "O": (0.95, 0.15, 0.15), "S": (1.0, 0.9, 0.1), "P": (1.0, 0.5, 0.0),
    "F": (0.0, 0.9, 0.9), "Cl": (0.1, 0.85, 0.1), "Br": (0.65, 0.35, 0.12),
    "I": (0.6, 0.15, 0.85), "B": (1.0, 0.7, 0.7), "Si": (0.8, 0.65, 0.5),
    "Se": (1.0, 0.63, 0.0), "Li": (0.8, 0.5, 1.0), "Na": (0.67, 0.36, 0.95),
    "K": (0.56, 0.25, 0.83), "Mg": (0.54, 1.0, 0.0), "Ca": (0.24, 1.0, 0.0),
    "Fe": (0.88, 0.4, 0.2), "Zn": (0.49, 0.5, 0.69), "Cu": (0.78, 0.5, 0.2),
}
RADII = {
    "H": 0.25, "C": 0.38, "N": 0.36, "O": 0.34, "S": 0.45, "P": 0.45,
    "F": 0.32, "Cl": 0.45, "Br": 0.5, "I": 0.55,
}
BOND_NAMES = {1: "Single bond", 2: "Double bond", 3: "Triple bond"}
BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
BOND_SCALE = {1: 1.0, 2: 0.88, 3: 0.79}
BOND_RADIUS = {1: 0.09, 2: 0.06, 3: 0.05}
BOND_OFFSETS = {1: [0.0], 2: [-0.1, 0.1], 3: [-0.18, 0.0, 0.18]}
H_RADIUS = 0.2
DRAG_THRESHOLD = 5
LETTERS_2D = True
LETTERS_3D = False
LABEL_GAP = 0.32
JOINT_R = 0.1
BOND_PICK_PX = 8
STEREO_COLOR = (1.0, 0.7, 0.2)

RING_GROUPS = {
    "Carbocycles": [
        ("Cyclopropane", "C1CC1"), ("Cyclobutane", "C1CCC1"), ("Cyclopentane", "C1CCCC1"),
        ("Cyclohexane", "C1CCCCC1"), ("Cycloheptane", "C1CCCCCC1"), ("Cyclooctane", "C1CCCCCCC1"),
        ("Cyclopentadiene", "C1=CC=CC1"), ("Cyclohexene", "C1=CCCCC1"), ("Benzene", "c1ccccc1"),
    ],
    "Aromatic 5-membered": [
        ("Pyrrole", "c1cc[nH]c1"), ("Furan", "c1ccoc1"), ("Thiophene", "c1ccsc1"),
        ("Imidazole", "c1c[nH]cn1"), ("Pyrazole", "c1cn[nH]c1"), ("Oxazole", "c1cocn1"),
        ("Thiazole", "c1cscn1"), ("Isoxazole", "c1cnoc1"), ("1,2,4-Triazole", "c1nc[nH]n1"),
        ("Tetrazole", "c1nnn[nH]1"),
    ],
    "Aromatic 6-membered": [
        ("Pyridine", "c1ccncc1"), ("Pyrimidine", "c1cncnc1"), ("Pyrazine", "c1cnccn1"),
        ("Pyridazine", "c1ccnnc1"), ("1,3,5-Triazine", "c1ncncn1"),
    ],
    "Saturated heterocycles": [
        ("Oxirane", "C1CO1"), ("Aziridine", "C1CN1"), ("Tetrahydrofuran", "C1CCOC1"),
        ("Pyrrolidine", "C1CCNC1"), ("Tetrahydropyran", "C1CCOCC1"), ("Piperidine", "C1CCNCC1"),
        ("Piperazine", "C1CNCCN1"), ("Morpholine", "C1COCCN1"), ("1,4-Dioxane", "C1COCCO1"),
    ],
    "Fused rings": [
        ("Naphthalene", "c1ccc2ccccc2c1"), ("Anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("Indole", "c1ccc2[nH]ccc2c1"), ("Benzofuran", "c1ccc2occc2c1"),
        ("Benzothiophene", "c1ccc2sccc2c1"), ("Benzimidazole", "c1ccc2[nH]cnc2c1"),
        ("Quinoline", "c1ccc2ncccc2c1"), ("Isoquinoline", "c1ccc2cnccc2c1"),
        ("Purine", "c1ncc2[nH]cnc2n1"), ("Decalin", "C1CCC2CCCCC2C1"),
    ],
}
RING_SMILES = {n: s for grp in RING_GROUPS.values() for n, s in grp}

GROUP_CATEGORIES = {
    "Alkyl / aryl": [
        ("Methyl", "*C"), ("Ethyl", "*CC"), ("n-Propyl", "*CCC"), ("Isopropyl", "*C(C)C"),
        ("n-Butyl", "*CCCC"), ("tert-Butyl", "*C(C)(C)C"), ("Vinyl", "*C=C"),
        ("Allyl", "*CC=C"), ("Ethynyl", "*C#C"), ("Phenyl", "*c1ccccc1"),
        ("Benzyl", "*Cc1ccccc1"), ("Trifluoromethyl", "*C(F)(F)F"),
        ("Trichloromethyl", "*C(Cl)(Cl)Cl"), ("Methylene (=CH2)", "*=C"),
        ("Trimethylsilyl", "*[Si](C)(C)C"),
    ],
    "Oxygen": [
        ("Hydroxyl (-OH)", "*O"), ("Methoxy (-OMe)", "*OC"), ("Ethoxy (-OEt)", "*OCC"),
        ("Phenoxy (-OPh)", "*Oc1ccccc1"), ("Acetoxy (-OAc)", "*OC(C)=O"),
        ("Oxo (=O)", "*=O"), ("Phosphate (-OPO3H2)", "*OP(=O)(O)O"),
    ],
    "Carbonyl": [
        ("Aldehyde (-CHO)", "*C=O"), ("Acetyl (-COMe)", "*C(C)=O"),
        ("Carboxylic acid (-COOH)", "*C(=O)O"), ("Carboxylate (-COO-)", "*C(=O)[O-]"),
        ("Methyl ester (-COOMe)", "*C(=O)OC"), ("Ethyl ester (-COOEt)", "*C(=O)OCC"),
        ("Amide (-CONH2)", "*C(N)=O"), ("N-Methylamide (-CONHMe)", "*C(=O)NC"),
        ("Acyl chloride (-COCl)", "*C(Cl)=O"), ("Anhydride", "*C(=O)OC(C)=O"),
        ("Boc ester (-COOtBu)", "*C(=O)OC(C)(C)C"),
    ],
    "Nitrogen": [
        ("Amino (-NH2)", "*N"), ("Methylamino (-NHMe)", "*NC"),
        ("Dimethylamino (-NMe2)", "*N(C)C"), ("Acetamido (-NHAc)", "*NC(C)=O"),
        ("Ammonium (-NH3+)", "*[NH3+]"), ("Nitro (-NO2)", "*[N+](=O)[O-]"),
        ("Nitroso (-NO)", "*N=O"), ("Cyano (-CN)", "*C#N"), ("Isocyanide (-NC)", "*[N+]#[C-]"),
        ("Azide (-N3)", "*N=[N+]=[N-]"), ("Imine (=NH)", "*=N"), ("Isocyanate (-NCO)", "*N=C=O"),
        ("Isothiocyanate (-NCS)", "*N=C=S"), ("Urea (-NHCONH2)", "*NC(N)=O"),
        ("Hydrazine (-NHNH2)", "*NN"), ("Oxime (-CH=NOH)", "*C=NO"),
    ],
    "Sulfur / P / B": [
        ("Thiol (-SH)", "*S"), ("Methylthio (-SMe)", "*SC"), ("Thione (=S)", "*=S"),
        ("Thiocyanate (-SCN)", "*SC#N"), ("Sulfoxide (-S(O)Me)", "*S(C)=O"),
        ("Methylsulfonyl (-SO2Me)", "*S(C)(=O)=O"), ("Sulfonic acid (-SO3H)", "*S(=O)(=O)O"),
        ("Sulfonamide (-SO2NH2)", "*S(N)(=O)=O"), ("Phosphonic acid (-PO3H2)", "*P(=O)(O)O"),
        ("Boronic acid (-B(OH)2)", "*B(O)O"),
    ],
    "Halogens": [
        ("Fluoro (-F)", "*F"), ("Chloro (-Cl)", "*Cl"), ("Bromo (-Br)", "*Br"), ("Iodo (-I)", "*I"),
    ],
}
GROUP_SMILES = {n: s for grp in GROUP_CATEGORIES.values() for n, s in grp}


def _prow(left, right=()):
    left, right = list(left), list(right)
    return left + [""] * (18 - len(left) - len(right)) + right


PT_ROWS = [
    _prow(["H"], ["He"]),
    _prow(["Li", "Be"], ["B", "C", "N", "O", "F", "Ne"]),
    _prow(["Na", "Mg"], ["Al", "Si", "P", "S", "Cl", "Ar"]),
    _prow("K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()),
    _prow("Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe".split()),
    _prow("Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()),
    [""] * 18,
    [""] * 3 + "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split() + [""],
]

_RING_CACHE = {}
_GROUP_CACHE = {}


_PT = Chem.GetPeriodicTable()


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


def color_of(el):
    return COLORS.get(base(el), (0.62, 0.62, 0.72))


LABEL_COLORS = {
    "N": (0.10, 0.20, 0.95), "O": (0.90, 0.05, 0.05), "S": (0.85, 0.68, 0.0),
    "F": (0.0, 0.72, 0.80), "Cl": (0.05, 0.68, 0.05), "Br": (0.60, 0.30, 0.05),
    "I": (0.50, 0.10, 0.75),
}


LABEL_COLORS_DARK = {
    "N": (0.35, 0.50, 1.00), "O": (1.00, 0.25, 0.25), "S": (1.00, 0.90, 0.10),
    "F": (0.20, 0.95, 1.00), "Cl": (0.20, 0.90, 0.20), "Br": (0.85, 0.50, 0.20),
    "I": (0.75, 0.40, 1.00),
}


def label_color(el):
    return LABEL_COLORS.get(base(el), (0.0, 0.0, 0.0))


def radius_of(el):
    s = base(el)
    if s in RADII:
        return RADII[s]
    try:
        return min(0.7, 0.2 + 0.25 * _PT.GetRcovalent(_PT.GetAtomicNumber(s)))
    except Exception:
        return 0.45


def ideal_bond_length(e1, e2, order):
    r = (_PT.GetRcovalent(_PT.GetAtomicNumber(base(e1)))
         + _PT.GetRcovalent(_PT.GetAtomicNumber(base(e2))))
    return r * BOND_SCALE[order]


def bkey(a, b):
    return (a, b) if a < b else (b, a)


def _apply_stereo(mol, idx, pos, chiral):
    try:
        if pos:
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.Set3D(False)
            for i, (x, y) in pos.items():
                conf.SetAtomPosition(idx[i], (float(x), float(y), 0.0))
            mol.AddConformer(conf, assignId=True)
            Chem.DetectBondStereochemistry(mol)
        if chiral:
            for aid, want in chiral.items():
                if aid not in idx:
                    continue
                atom = mol.GetAtomWithIdx(idx[aid])
                found = False
                for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                    atom.SetChiralTag(tag)
                    t = Chem.Mol(mol)
                    Chem.AssignStereochemistry(t, cleanIt=True, force=True)
                    ta = t.GetAtomWithIdx(idx[aid])
                    if ta.HasProp("_CIPCode") and ta.GetProp("_CIPCode") == want:
                        found = True
                        break
                if not found:
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass


def build_mol(elems, bonds, pos=None, chiral=None):
    rw = Chem.RWMol()
    idx = {}
    for i in sorted(elems):
        sym, q = split_el(elems[i])
        a = Chem.Atom(sym)
        a.SetFormalCharge(q)
        idx[i] = rw.AddAtom(a)
    for (a, b), order in sorted(bonds.items()):
        rw.AddBond(idx[a], idx[b], BOND_TYPES[order])
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, str(exc).split("\n")[0], idx
    if pos or chiral:
        _apply_stereo(mol, idx, pos, chiral)
    return mol, "", idx


def _template_from_mol(m, a0_pref=None):
    pass


def ring_template(name):
    if name in _RING_CACHE:
        return _RING_CACHE[name]
    from rdkit.Chem import rdDepictor
    m = Chem.MolFromSmiles(RING_SMILES[name])
    Chem.Kekulize(m, clearAromaticFlags=True)
    rdDepictor.Compute2DCoords(m)
    xy = np.array(m.GetConformer().GetPositions())[:, :2]
    xy = xy - xy.mean(axis=0)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(round(b.GetBondTypeAsDouble())))
             for b in m.GetBonds()]
    a0 = syms.index("C")
    cv = -xy[a0]
    t = dict(syms=syms, bonds=bonds, xy=xy, a0=a0, din=cv / np.linalg.norm(cv), order=1)
    _RING_CACHE[name] = t
    return t


def group_template(name):
    if name in _GROUP_CACHE:
        return _GROUP_CACHE[name]
    from rdkit.Chem import rdDepictor
    m = Chem.MolFromSmiles(GROUP_SMILES[name])
    Chem.Kekulize(m, clearAromaticFlags=True)
    rdDepictor.Compute2DCoords(m)
    xy_all = np.array(m.GetConformer().GetPositions())[:, :2]
    dummy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0][0]
    dn = m.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
    order = int(round(m.GetBondBetweenAtoms(dummy, dn).GetBondTypeAsDouble()))
    keep = [i for i in range(m.GetNumAtoms()) if i != dummy]
    remap = {old: new for new, old in enumerate(keep)}
    syms = [el_str(m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(i).GetFormalCharge())
            for i in keep]
    bonds = [(remap[b.GetBeginAtomIdx()], remap[b.GetEndAtomIdx()],
              int(round(b.GetBondTypeAsDouble())))
             for b in m.GetBonds() if dummy not in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())]
    din = xy_all[dn] - xy_all[dummy]
    t = dict(syms=syms, bonds=bonds, xy=xy_all[keep], a0=remap[dn],
             din=din / np.linalg.norm(din), order=order)
    _GROUP_CACHE[name] = t
    return t


def kabsch2d(new, old):
    cn, co = new.mean(axis=0), old.mean(axis=0)
    P, Q = new - cn, old - co
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return P @ R.T + co


def fit_planar(elems, bonds, pos, movable=None):
    if not bonds:
        return None
    mol, err, idx = build_mol(elems, bonds)
    if mol is None:
        return None
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y) in pos.items():
        conf.SetAtomPosition(idx[i], (float(x), float(y), 0.0))
    mol.AddConformer(conf, assignId=True)
    try:
        if not AllChem.UFFHasAllMoleculeParams(mol):
            return None
        ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is None:
            return None
        if movable is not None:
            for i in pos:
                if i not in movable:
                    ff.AddFixedPoint(idx[i])
        ff.Initialize()
        ff.Minimize(maxIts=2000)
    except Exception:
        return None
    p = np.array(mol.GetConformer().GetPositions())
    if not np.all(np.isfinite(p)):
        return None
    out = {i: (float(p[idx[i], 0]), float(p[idx[i], 1])) for i in pos}
    if movable is None:
        id_of = {v: k for k, v in idx.items()}
        for frag in Chem.GetMolFrags(mol):
            if len(frag) < 2:
                continue
            ids = [id_of[k] for k in frag]
            old = np.array([pos[i] for i in ids], float)
            new = np.array([out[i] for i in ids], float)
            for i, q in zip(ids, kabsch2d(new, old)):
                out[i] = (float(q[0]), float(q[1]))
    return out


def _forcefield(mh):
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            return AllChem.MMFFGetMoleculeForceField(mh, AllChem.MMFFGetMoleculeProperties(mh))
        if AllChem.UFFHasAllMoleculeParams(mh):
            return AllChem.UFFGetMoleculeForceField(mh)
    except Exception:
        pass
    return None


def embed_3d(elems, bonds, pos, chiral=None):
    mol, err, idx = build_mol(elems, bonds, pos, chiral)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    n = mol.GetNumAtoms()
    mh = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 7
    ok = AllChem.EmbedMolecule(mh, ps) == 0
    if not ok:
        ps.useRandomCoords = True
        ok = AllChem.EmbedMolecule(mh, ps) == 0
    if not ok:
        rng = np.random.default_rng(1)
        c = Chem.Conformer(n)
        for i, (x, y) in pos.items():
            c.SetAtomPosition(idx[i], (float(x), float(y), float(rng.normal(0, 0.15))))
        m2 = Chem.Mol(mol)
        m2.RemoveAllConformers()
        m2.AddConformer(c, assignId=True)
        mh = Chem.AddHs(m2, addCoords=True)
    ff = _forcefield(mh)
    if ff is not None:
        try:
            ff.Initialize()
            ff.Minimize(maxIts=5000)
        except Exception:
            pass
    try:
        ref = Chem.Mol(mol)
        ref.RemoveAllConformers()
        c = Chem.Conformer(n)
        for i, (x, y) in pos.items():
            c.SetAtomPosition(idx[i], (float(x), float(y), 0.0))
        ref.AddConformer(c, assignId=True)
        rdMolAlign.AlignMol(mh, ref, atomMap=[(k, k) for k in range(n)])
    except Exception:
        pass
    return mh, idx


def minimize_3d(mh, idx, movable=None):
    ff = _forcefield(mh)
    if ff is None:
        return False
    n = len(idx)
    if movable is not None:
        id_of = {v: k for k, v in idx.items()}
        for a in mh.GetAtoms():
            k = a.GetIdx()
            parent = k if k < n else a.GetNeighbors()[0].GetIdx()
            if id_of[parent] not in movable:
                ff.AddFixedPoint(k)
    ff.Initialize()
    ff.Minimize(maxIts=5000)
    return True


class TextLabel(vtk.vtkTextActor):

    def __init__(self):
        super().__init__()
        self.world = np.zeros(3)
        self.base_size = 0.0
        self.hshift = 0

    def SetPosition(self, x, y, z=0.0):
        self.world = np.array([x, y, z], float)


class AtomRec:
    def __init__(self, aid, element, pos, actor, src, label):
        self.id, self.element, self.actor, self.src, self.label = aid, element, actor, src, label
        self.pos = np.array(pos, float)
        self.pos2 = self.pos.copy()
        self.has_label = False

    def sync(self):
        self.actor.SetPosition(*self.pos)
        self.label.SetPosition(*self.pos)


ARROW_LEN = 3.0
DEFAULT_BOND_SCALE = 0.5
REF_PXW = 1.0 / 60.0
ANNOT_NAMES = {"arrow": "reaction arrow", "eq": "equilibrium arrow", "dbl": "double-headed arrow",
               "plus": "plus sign", "text": "text label"}


def arrow_polylines(kind, L=ARROW_LEN, h=0.36, g=0.13, angle=0.0):
    lines, segs = _arrow_polylines(kind, L, h, g)
    if angle:
        c, sn = math.cos(angle), math.sin(angle)
        rot = lambda p: (p[0] * c - p[1] * sn, p[0] * sn + p[1] * c)
        lines = [[rot(p) for p in pl] for pl in lines]
        segs = [(rot(a), rot(b)) for a, b in segs]
    return lines, segs


def _arrow_polylines(kind, L, h, g):
    s = h * 0.6
    if kind == "arrow":
        return ([[(-L / 2, 0), (L / 2, 0)], [(L / 2 - h, s), (L / 2, 0), (L / 2 - h, -s)]],
                [((-L / 2, 0), (L / 2, 0))])
    if kind == "dbl":
        return ([[(-L / 2, 0), (L / 2, 0)], [(L / 2 - h, s), (L / 2, 0), (L / 2 - h, -s)],
                 [(-L / 2 + h, s), (-L / 2, 0), (-L / 2 + h, -s)]], [((-L / 2, 0), (L / 2, 0))])
    return ([[(-L / 2, g), (L / 2, g)], [(L / 2 - h, g + s), (L / 2, g)],
             [(L / 2, -g), (-L / 2, -g)], [(-L / 2 + h, -g - s), (-L / 2, -g)]],
            [((-L / 2, g), (L / 2, g)), ((-L / 2, -g), (L / 2, -g))])


class AnnotRec:

    def __init__(self, aid, kind, pos, text, actors, segs=(), size=0, scale=1.0, angle=0.0):
        self.id, self.kind, self.text, self.actors, self.size = aid, kind, text, actors, size
        self.angle = angle
        self.scale, self.base_size = scale, (size / scale if scale else size)
        self.pos = np.array(pos, float)
        self.base = self.pos.copy()
        self.segs = [(np.array([*a, 0.0], float), np.array([*b, 0.0], float)) for a, b in segs]

    def describe(self):
        return f'text label "{self.text}"' if self.kind == "text" else ANNOT_NAMES[self.kind]


class TextLabelDialog(QtWidgets.QDialog):
    SYMBOLS = ["\u00b0", "\u00b0C", "\u0394", "\u00b1", "\u2192", "\u21cc", "\u00d7", "\u00b5"]

    def __init__(self, text="", title="Text label", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Label text:"))
        self.edit = QtWidgets.QLineEdit(text)
        self.edit.setMinimumWidth(320)
        lay.addWidget(self.edit)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Insert:"))
        for sym in self.SYMBOLS:
            b = QtWidgets.QPushButton(sym)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.setFixedWidth(40)
            b.clicked.connect(lambda checked=False, t=sym: self._insert(t))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.edit.setFocus()
        self.edit.selectAll()

    def _insert(self, t):
        self.edit.insert(t)
        self.edit.setFocus()

    def value(self):
        return self.edit.text().strip()


class ChargeDialog(QtWidgets.QDialog):
    def __init__(self, element, charge, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Formal charge of {element}")
        lay = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("font-size:16px;font-weight:600;")
        lay.addWidget(self.label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(-4, 4)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setPageStep(1)
        self.slider.setMinimumWidth(280)
        self.slider.setValue(int(charge))
        lay.addWidget(self.slider)
        ticks = QtWidgets.QHBoxLayout()
        for v in range(-4, 5):
            t = QtWidgets.QLabel(f"{v:+d}" if v else "0")
            t.setAlignment(QtCore.Qt.AlignCenter)
            ticks.addWidget(t)
        lay.addLayout(ticks)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.slider.valueChanged.connect(self._upd)
        self._upd(self.slider.value())

    def _upd(self, v):
        self.label.setText(f"Charge: {v:+d}" if v else "Charge: 0 (neutral)")

    def value(self):
        return self.slider.value()


class PeriodicTableDialog(QtWidgets.QDialog):
    def __init__(self, current, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Periodic table")
        self.chosen = None
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(3)
        for r, row in enumerate(PT_ROWS):
            for c, sym in enumerate(row):
                if not sym:
                    continue
                btn = QtWidgets.QPushButton(sym)
                btn.setFixedSize(44, 40)
                btn.setFocusPolicy(QtCore.Qt.NoFocus)
                border = "3px solid #ff9800" if sym == current else "1px solid #888"
                btn.setStyleSheet(
                    f"QPushButton{{background:#f0f0f0;color:#111;font-weight:700;border:{border};"
                    f"border-radius:4px;}}QPushButton:hover{{background:#dbe8ff;border:2px solid #3366cc;}}")
                btn.clicked.connect(lambda checked=False, s=sym: self._pick(s))
                grid.addWidget(btn, r, c)
        grid.setRowMinimumHeight(6, 10)

    def _pick(self, sym):
        self.chosen = sym
        self.accept()


class Sketch(QtCore.QObject):
    smilesChanged = QtCore.Signal(str)
    toolChanged = QtCore.Signal()
    statusMessage = QtCore.Signal(str)
    dockRequested = QtCore.Signal(list)
    tableRequested = QtCore.Signal(list)

    def __init__(self, renderer, widget, parent=None):
        super().__init__(parent)
        self.ren = renderer
        self.widget = widget
        self.rw = widget.GetRenderWindow()
        self.element = "C"
        self.ring = None
        self.group = None
        self.bond_order = 1
        self.draw2d = True
        self.editing = False
        self.attached = False
        self.dark = True
        self.fg = (1.0, 1.0, 1.0)
        self.z0 = 0.0
        self._framed = False
        self._gap_px = None

        self.atoms, self.bonds = {}, {}
        self.bond_actors = {}
        self.actor2id = {}
        self.next_id = 0
        self.mol = None
        self.smiles = ""
        self.mol3, self.idx3 = None, None
        self.chiral = {}
        self.centers = {}
        self.ez = {}
        self.stereo_actors = []
        self.history = []

        self.selected = set()
        self.sel_halos = {}
        self.hover_id = None
        self.hover_bond = None
        self.click_bond = None
        self.mode = None
        self.press_pos = (0, 0)
        self.pan_last = (0, 0)
        self.first_click = False
        self.click_annot = None
        self.scaling = None
        self.rotating = None
        self.mono_letters = False
        self.stereo_mode = 1
        self.wedge_info = {}
        self._mol_idx = {}
        self.bond_scale = DEFAULT_BOND_SCALE
        self.text_bold = False
        self.annots, self.annot_next, self.sel_annots = {}, 0, set()
        self.moved = False
        self.drag_src = None
        self.right_atom = None
        self.right_bond = None

        self.bond_hover_line = vtk.vtkLineSource()
        mb = vtk.vtkPolyDataMapper()
        mb.SetInputConnection(self.bond_hover_line.GetOutputPort())
        self.bond_hover_actor = vtk.vtkActor()
        self.bond_hover_actor.SetMapper(mb)
        self.bond_hover_actor.GetProperty().SetColor(1.0, 0.75, 0.0)
        self.bond_hover_actor.GetProperty().SetLineWidth(7)
        self.bond_hover_actor.GetProperty().SetOpacity(0.55)
        self.bond_hover_actor.PickableOff()
        self.bond_hover_actor.VisibilityOff()

        self.hover_halo = self._make_halo((1.0, 0.75, 0.0), 0.35)
        self.ghost = self._make_halo((0.5, 0.5, 0.5), 0.45)
        self.preview_line = vtk.vtkLineSource()
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(self.preview_line.GetOutputPort())
        self.preview_actor = vtk.vtkActor()
        self.preview_actor.SetMapper(m)
        self.preview_actor.GetProperty().SetColor(1.0, 0.85, 0.2)
        self.preview_actor.GetProperty().SetLineWidth(3)
        self.preview_actor.PickableOff()
        self.preview_actor.VisibilityOff()

        self.rb_pts = vtk.vtkPoints()
        for _ in range(5):
            self.rb_pts.InsertNextPoint(0, 0, 0)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(5)
        for i in range(5):
            cells.InsertCellPoint(i)
        self.rb_poly = vtk.vtkPolyData()
        self.rb_poly.SetPoints(self.rb_pts)
        self.rb_poly.SetLines(cells)
        m2 = vtk.vtkPolyDataMapper2D()
        m2.SetInputData(self.rb_poly)
        self.rb_actor = vtk.vtkActor2D()
        self.rb_actor.SetMapper(m2)
        self.rb_actor.GetProperty().SetColor(1.0, 0.6, 0.1)
        self.rb_actor.GetProperty().SetLineWidth(1.5)
        self.rb_actor.PickableOff()
        self.rb_actor.VisibilityOff()
        self._fixed = [self.bond_hover_actor, self.hover_halo, self.ghost,
                       self.preview_actor, self.rb_actor]

        self.ren.AddObserver("StartEvent", self._on_render_start)

    def reattach(self, z0=None):
        if z0 is not None and not self.atoms:
            self.z0 = float(z0)
        self.attached = True
        for a in self._fixed:
            self.ren.AddViewProp(a)
        for r in self.atoms.values():
            self.ren.AddActor(r.actor)
            self.ren.AddActor(r.label)
        for rec in self.annots.values():
            for a in rec.actors:
                self.ren.AddViewProp(a)
        self._gap_px = None
        self._apply_style()
        self._rebuild_all_bonds()
        self._refresh_stereo_labels()
        self._refresh_halos()
        self._set_theme_colors()

    def detach(self):
        self.attached = False
        self.mode = None
        self.stereo_actors = []
        self.sel_halos = {}
        self.bond_actors = {}

    def clear(self):
        if self.attached:
            for r in self.atoms.values():
                self.ren.RemoveActor(r.actor)
                self.ren.RemoveActor(r.label)
            for a in list(self.bond_actors.values()) + self.stereo_actors + list(self.sel_halos.values()):
                self.ren.RemoveActor(a)
        for rec in self.annots.values():
            for a in rec.actors:
                self.ren.RemoveViewProp(a)
        self.annots, self.annot_next, self.sel_annots = {}, 0, set()
        self.wedge_info = {}
        self.atoms, self.bonds = {}, {}
        self.bond_actors, self.stereo_actors, self.sel_halos = {}, [], {}
        self.chiral, self.centers, self.ez = {}, {}, {}
        self.selected, self.history = set(), []
        self.next_id = 0
        self.mol, self.smiles = None, ""
        self.hover_id = self.hover_bond = None
        self.mode = None
        self._framed = False
        for a in self._fixed:
            a.VisibilityOff()

    def set_editing(self, on):
        self.editing = bool(on)
        self.mode = None
        self._clear_preview()
        self.rb_actor.VisibilityOff()
        if not on:
            self.hover_id = self.hover_bond = None
            self.hover_halo.VisibilityOff()
            self.bond_hover_actor.VisibilityOff()
            self.selected = set()
            self.sel_annots = set()
            self._refresh_annot_colors()
            if self.attached:
                self._refresh_halos()
        if self.attached:
            self.render()

    def set_dark(self, dark):
        self.dark = bool(dark)
        self.fg = (1.0, 1.0, 1.0) if self.dark else (0.0, 0.0, 0.0)
        self._set_theme_colors()
        self._refresh_annot_colors()
        if self.attached and self.atoms:
            for r in self.atoms.values():
                r.label.GetTextProperty().SetColor(*self._lcol(r.element))
            self._apply_style()
            self._rebuild_all_bonds()
            self._refresh_stereo_labels()
            self.render()

    def _set_theme_colors(self):
        self.preview_actor.GetProperty().SetColor(*((1.0, 0.85, 0.2) if self.dark else (0.2, 0.4, 0.9)))

    def _lcol(self, el):
        if self.mono_letters:
            return self.fg
        if self.dark:
            return LABEL_COLORS_DARK.get(base(el), (1.0, 1.0, 1.0))
        return label_color(el)

    def enter_draw(self, center, radius):
        cam = self.ren.GetActiveCamera()
        dop = np.array(cam.GetDirectionOfProjection())
        up = np.array(cam.GetViewUp())
        aligned = dop[2] < -0.999 and abs(up[0]) < 0.02 and up[1] > 0.99
        if not self.atoms and not self._framed:
            self._frame(center, radius)
            self._framed = True
        elif not aligned:
            pos = np.array(cam.GetPosition())
            if abs(dop[2]) > 1e-3:
                t = (self.z0 - pos[2]) / dop[2]
                q = pos + t * dop
                dist = abs(t)
            else:
                q = np.array(cam.GetFocalPoint())
                q[2] = self.z0
                dist = float(np.linalg.norm(pos - q))
            cam.SetFocalPoint(*q)
            cam.SetPosition(q[0], q[1], self.z0 + max(dist, 5.0))
            cam.SetViewUp(0, 1, 0)
        cam.ParallelProjectionOff()
        self.ren.ResetCameraClippingRange()

    def _frame(self, center, radius):
        cam = self.ren.GetActiveCamera()
        cx, cy = float(center[0]), float(center[1])
        R = max(float(radius), 4.0)
        f = np.array([cx + 0.9 * R, cy, self.z0])
        w, h = self.ren.GetSize()
        aspect = max(w, 1) / max(h, 1)
        half_h = max(1.5 * R, 2.2 * R / aspect)
        dist = half_h / math.tan(math.radians(cam.GetViewAngle()) / 2.0)
        cam.ParallelProjectionOff()
        cam.SetFocalPoint(*f)
        cam.SetPosition(f[0], f[1], f[2] + dist)
        cam.SetViewUp(0, 1, 0)

    def _world_at(self, x, y):
        p0 = self.display_to_world(x, y, 0.0)
        p1 = self.display_to_world(x, y, 1.0)
        d = p1 - p0
        if abs(d[2]) < 1e-9:
            return np.array([p0[0], p0[1], self.z0])
        return p0 + ((self.z0 - p0[2]) / d[2]) * d

    def _px_world(self):
        cam = self.ren.GetActiveCamera()
        h = max(self.ren.GetSize()[1], 1)
        if cam.GetParallelProjection():
            return 2.0 * cam.GetParallelScale() / h
        dop = np.array(cam.GetDirectionOfProjection())
        pos = np.array(cam.GetPosition())
        if abs(dop[2]) > 1e-3:
            depth = abs((self.z0 - pos[2]) / dop[2])
        else:
            depth = float(np.linalg.norm(pos - np.array(cam.GetFocalPoint())))
        return 2.0 * depth * math.tan(math.radians(cam.GetViewAngle()) / 2.0) / h

    def label_gap(self):
        return float(np.clip(18.0 * REF_PXW, 0.05, 2.0))

    def _letters(self):
        return True

    def _editable(self):
        if not self.editing:
            self.say("Switch on Draw mode to edit the drawing.")
            return False
        return True

    def render(self):
        self.rw.Render()

    def _on_render_start(self, obj=None, ev=None):
        if not self.attached or (not self.atoms and not self.stereo_actors and not self.annots):
            return
        pxw = max(self._px_world(), 1e-9)
        props = self.ren.GetViewProps()
        props.InitTraversal()
        for _ in range(props.GetNumberOfItems()):
            pr = props.GetNextProp()
            if isinstance(pr, TextLabel):
                d = self.world_to_display(pr.world)
                fs = pr.GetTextProperty().GetFontSize()
                if pr.base_size > 0:
                    fs = int(round(float(np.clip(pr.base_size * REF_PXW / pxw, 6, 200))))
                    tp = pr.GetTextProperty()
                    if tp.GetFontSize() != fs:
                        tp.SetFontSize(fs)
                vtk.vtkTextActor.SetPosition(pr, d[0] - pr.hshift * 0.3 * fs, d[1])

    def picture_records(self):
        out = []
        if not self.attached:
            return out
        items = [(r.label, 22) for r in self.atoms.values() if r.has_label]
        items += [(a, 15) for a in self.stereo_actors]
        for lab, size in items:
            if not lab.GetVisibility():
                continue
            text = lab.GetInput()
            d = self.world_to_display(lab.world)
            col = lab.GetTextProperty().GetColor()
            out.append((text, d[0] - 0.3 * size * len(text), d[1] - 0.35 * size, tuple(col), size))
        return out

    def hide_labels(self, hide):
        if not self.attached:
            return
        if hide:
            for r in self.atoms.values():
                r.label.VisibilityOff()
            for a in self.stereo_actors:
                a.VisibilityOff()
        else:
            self._apply_style()
            for a in self.stereo_actors:
                a.VisibilityOn()

    def _minimize(self, movable):
        what = "selection" if movable is not None else "all atoms"
        if self._fit2d(movable):
            self._rebuild_all_bonds()
            self._refresh_halos()
            self._sync()
            self.say(f"Energy-minimized {what}.")
        else:
            self.say("Minimization failed (no force-field parameters?).")

    def press(self, button, x, y, shift=False):
        self.press_pos, self.moved = (x, y), False
        if button == "left":
            if shift and (self.selected or self.sel_annots):
                self.mode = "move"
                self.move_start = self._world_at(x, y)
                self.move_atoms = {i: self.atoms[i].pos2.copy() for i in self.selected if i in self.atoms}
                self.move_annots = {i: self.annots[i].pos.copy() for i in self.sel_annots if i in self.annots}
                return
            aid = self.pick_atom(x, y)
            if aid is not None:
                self.mode, self.drag_src = "bond", aid
                self.hover_id = aid
                self._update_hover()
                self.render()
                return
            bk = self.pick_bond(x, y)
            if bk is not None:
                self.mode, self.click_bond = "bondclick", bk
            else:
                an = self.pick_annot(x, y)
                if an is not None:
                    self.mode, self.click_annot = "annotclick", an
                else:
                    self.mode = "empty"
        else:
            self.right_atom = self.pick_atom(x, y)
            self.right_bond = self.pick_bond(x, y) if self.right_atom is None else None
            self.mode = "right"

    def move(self, x, y, left_down=False, right_down=False):
        if self.mode in ("empty", "annotclick"):
            self._check_moved(x, y)
            return
        if self.mode == "move":
            self._check_moved(x, y)
            if self.moved:
                d = self._world_at(x, y) - self.move_start
                d[2] = 0.0
                for i, p0 in self.move_atoms.items():
                    r = self.atoms[i]
                    r.pos = p0 + d
                    r.pos2 = r.pos.copy()
                    r.sync()
                for i, p0 in self.move_annots.items():
                    rec = self.annots[i]
                    rec.pos = p0 + d
                    for a in rec.actors:
                        if isinstance(a, TextLabel):
                            a.SetPosition(*rec.pos)
                        else:
                            a.SetPosition(*(rec.pos - rec.base))
                self._rebuild_all_bonds()
                self._refresh_halos()
                self.render()
            return
        if self.mode in ("right", "rband"):
            self._check_moved(x, y)
            if self.moved:
                self.mode = "rband"
                x0, y0 = self.press_pos
                for i, (px, py) in enumerate([(x0, y0), (x, y0), (x, y), (x0, y), (x0, y0)]):
                    self.rb_pts.SetPoint(i, px, py, 0)
                self.rb_pts.Modified()
                self.rb_poly.Modified()
                self.rb_actor.VisibilityOn()
                self.render()
            return
        if self.mode == "bondclick":
            self._check_moved(x, y)
            if self.moved:
                self.mode = "empty"
            return
        if self.mode == "bond":
            self._check_moved(x, y)
            self._update_preview(x, y)
            return
        if left_down or right_down:
            return
        aid = self.pick_atom(x, y)
        bk = self.pick_bond(x, y) if aid is None else None
        if aid != self.hover_id or bk != self.hover_bond:
            self.hover_id, self.hover_bond = aid, bk
            self._update_hover()
            self._update_bond_hover()
            self.render()

    def _pan_to(self, x, y):
        w_old = self._world_at(*self.pan_last)
        w_new = self._world_at(x, y)
        d = w_old - w_new
        cam = self.ren.GetActiveCamera()
        cam.SetPosition(*(np.array(cam.GetPosition()) + d))
        cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) + d))
        self.pan_last = (x, y)
        self.ren.ResetCameraClippingRange()
        self.render()

    def release(self, button, x, y):
        mode, self.mode = self.mode, None
        if button == "left":
            if mode == "empty":
                if not self.moved:
                    if self.selected or self.sel_annots:
                        self.clear_selection()
                    else:
                        self.place_first(*self.press_pos)
            elif mode == "move":
                if self.moved:
                    self._commit(fit=False)
                    self.notify("Moved selection.")
            elif mode == "annotclick":
                if not self.moved and self.click_annot in self.annots:
                    self.delete_annots([self.click_annot])
                self.click_annot = None
            elif mode == "bondclick":
                if not self.moved and self.click_bond in self.bonds:
                    self.cycle_bond(self.click_bond)
                self.click_bond = None
            elif mode == "bond":
                self._clear_preview()
                src = self.drag_src
                if src in self.atoms:
                    if not self.moved:
                        if self.group:
                            self.attach_group(src, self.group)
                        elif self.ring:
                            self.attach_ring(src, self.ring)
                        elif self.atoms[src].element != self.element:
                            self.replace_atom(src, self.element)
                    else:
                        tgt = self.pick_atom(x, y)
                        if tgt is not None and tgt != src:
                            self.connect(src, tgt, self.bond_order)
                        elif tgt is None:
                            pos, _ = self.ghost_position(src, x, y)
                            self.add_atom_bonded(src, self.element, pos, self.bond_order)
                self.drag_src = None
                self.hover_id = self.pick_atom(x, y)
                self._update_hover()
                self.render()
        else:
            if mode == "rband":
                self.rb_actor.VisibilityOff()
                x0, y0 = self.press_pos
                xa, xb = sorted((x0, x))
                ya, yb = sorted((y0, y))
                ids = []
                for i, r in self.atoms.items():
                    d = self.world_to_display(r.pos)
                    if xa <= d[0] <= xb and ya <= d[1] <= yb:
                        ids.append(i)
                ann = []
                for i, rec in self.annots.items():
                    d = self.world_to_display(rec.pos)
                    if xa <= d[0] <= xb and ya <= d[1] <= yb:
                        ann.append(i)
                self.set_selection(ids, ann)
                if ids or ann:
                    self.say(f"{len(ids)} atom(s) + {len(ann)} label(s) selected: "
                             "Ctrl+X delete, Ctrl+M minimize, Esc cancel.")
            elif mode == "right":
                aid, bk, xy = self.right_atom, self.right_bond, self.press_pos
                QtCore.QTimer.singleShot(0, lambda: self.show_menu(aid, bk, xy))


    def _make_halo(self, color, opacity):
        s = vtk.vtkSphereSource()
        s.SetRadius(1.0)
        s.SetThetaResolution(28)
        s.SetPhiResolution(28)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(s.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*color)
        a.GetProperty().SetOpacity(opacity)
        a.PickableOff()
        a.VisibilityOff()
        self.ren.AddActor(a)
        return a

    def world_to_display(self, p):
        self.ren.SetWorldPoint(p[0], p[1], p[2], 1.0)
        self.ren.WorldToDisplay()
        return self.ren.GetDisplayPoint()

    def display_to_world(self, x, y, z):
        self.ren.SetDisplayPoint(x, y, z)
        self.ren.DisplayToWorld()
        w = self.ren.GetWorldPoint()
        return np.array(w[:3]) / w[3]

    def cam_axes(self):
        cam = self.ren.GetActiveCamera()
        dop = np.array(cam.GetDirectionOfProjection())
        up = np.array(cam.GetViewUp())
        right = np.cross(dop, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, dop)
        up /= np.linalg.norm(up)
        return right, up, dop

    def pick_atom(self, x, y):
        right, _, _ = self.cam_axes()
        best, best_key = None, None
        for i, r in self.atoms.items():
            d0 = self.world_to_display(r.pos)
            dist = math.hypot(d0[0] - x, d0[1] - y)
            d1 = self.world_to_display(r.pos + right * radius_of(r.element))
            rad = math.hypot(d1[0] - d0[0], d1[1] - d0[1])
            rad = max(rad, 14.0 if self._letters() else 6.0)
            if dist <= rad:
                key = (round(d0[2], 5), dist)
                if best_key is None or key < best_key:
                    best, best_key = i, key
        return best

    def pick_bond(self, x, y):
        best, best_d = None, BOND_PICK_PX
        p = np.array([x, y], float)
        for k in self.bonds:
            a = np.array(self.world_to_display(self.atoms[k[0]].pos)[:2])
            b = np.array(self.world_to_display(self.atoms[k[1]].pos)[:2])
            ab = b - a
            t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
            d = float(np.linalg.norm(p - (a + t * ab)))
            if d < best_d:
                best, best_d = k, d
        return best

    def _update_bond_hover(self):
        k = self.hover_bond
        if k is None or k not in self.bonds:
            self.bond_hover_actor.VisibilityOff()
            return
        self.bond_hover_line.SetPoint1(*self.atoms[k[0]].pos)
        self.bond_hover_line.SetPoint2(*self.atoms[k[1]].pos)
        self.bond_hover_actor.VisibilityOn()

    def _apply_style(self):
        letters = self._letters()
        deg, nbrs = {}, {}
        for a, b in self.bonds:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
            nbrs.setdefault(a, []).append(b)
            nbrs.setdefault(b, []).append(a)
        hs = self._implicit_h()
        for r in self.atoms.values():
            sym, q = split_el(r.element)
            show = letters and (sym != "C" or q != 0 or deg.get(r.id, 0) == 0)
            r.has_label = show
            r.label.SetVisibility(show)
            self._set_label_text(r, hs.get(r.id, 0), [self.atoms[n].pos2[0] - r.pos2[0] for n in nbrs.get(r.id, []) if n in self.atoms])
            prop = r.actor.GetProperty()
            if letters:
                if show:
                    r.actor.VisibilityOff()
                else:
                    r.src.SetRadius(JOINT_R * self.bond_scale)
                    prop.SetColor(*self.fg)
                    r.actor.VisibilityOn()
            else:
                r.src.SetRadius(radius_of(r.element))
                prop.SetColor(*color_of(r.element))
                r.actor.VisibilityOn()

    def _implicit_h(self):
        if not self.atoms:
            return {}
        mol, _, idx = build_mol(self._elems(), self.bonds)
        if mol is None:
            return {}
        return {i: mol.GetAtomWithIdx(k).GetTotalNumHs() for i, k in idx.items()}

    def _set_label_text(self, r, nh, dxs):
        sym, q = split_el(r.element)
        tp = r.label.GetTextProperty()
        text, shift = display_el(r.element), 0
        if nh > 0 and sym not in ("C", "H"):
            chg = display_el(r.element)[len(sym):]
            hh = "H" if nh == 1 else f"H{nh}"
            if dxs:
                prefix = (sum(dxs) / len(dxs)) > 0.05
            else:
                prefix = sym in ("O", "S", "Se", "F", "Cl", "Br", "I")
            text, shift = (f"{hh}{sym}{chg}", -1) if prefix else (f"{sym}{hh}{chg}", 1)
        if r.label.GetInput() != text:
            r.label.SetInput(text)
        r.label.hshift = shift
        if shift > 0:
            tp.SetJustificationToLeft()
        elif shift < 0:
            tp.SetJustificationToRight()
        else:
            tp.SetJustificationToCentered()

    def notify(self, msg):
        self.statusMessage.emit(msg)

    def say(self, msg):
        low = msg.lower()
        if low.startswith("rejected") or "valence" in low or "valency" in low:
            self.statusMessage.emit(msg)

    def set_element(self, el):
        self.element, self.ring, self.group = el, None, None
        self.toolChanged.emit()

    def set_ring(self, name):
        self.ring, self.group = name, None
        self.toolChanged.emit()

    def set_group(self, name):
        self.group, self.ring = name, None
        self.toolChanged.emit()

    def set_bond(self, order):
        self.bond_order = order
        self.toolChanged.emit()

    def open_periodic_table(self, place_xy=None):
        dlg = PeriodicTableDialog(self.element if not (self.ring or self.group) else "",
                                  self.widget)
        if dlg.exec() and dlg.chosen:
            self._choose_element(dlg.chosen, place_xy)

    def _elems(self):
        return {i: r.element for i, r in self.atoms.items()}

    def _check(self, elems, bonds):
        mol, err, _ = build_mol(elems, bonds)
        if mol is None:
            self.say("Rejected: " + err)
            return False
        return True

    def _sync(self, push=True):
        pos = {i: (r.pos2[0], r.pos2[1]) for i, r in self.atoms.items()}
        mol, err, idx = build_mol(self._elems(), self.bonds, pos, self.chiral)
        self.mol = mol
        self._mol_idx = idx
        smi = ""
        self.centers, self.ez = {}, {}
        if mol is not None and mol.GetNumAtoms():
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                smi = Chem.MolToSmiles(mol)
            id_of = {v: k for k, v in idx.items()}
            try:
                for i, c in Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True):
                    self.centers[id_of[i]] = c
            except Exception:
                pass
            for b in mol.GetBonds():
                if b.GetBondType() != Chem.BondType.DOUBLE:
                    continue
                st = b.GetStereo()
                code = ("E" if st == Chem.BondStereo.STEREOE
                        else "Z" if st == Chem.BondStereo.STEREOZ else None)
                if code:
                    self.ez[bkey(id_of[b.GetBeginAtomIdx()], id_of[b.GetEndAtomIdx()])] = code
        self.smiles = smi
        self.mol3, self.idx3 = None, None
        self._refresh_stereo_labels()
        if push:
            self._push_history()
        self.smilesChanged.emit(smi)
        self.render()

    def _snapshot(self):
        return dict(atoms={i: (r.element, r.pos2.copy()) for i, r in self.atoms.items()},
                    bonds=dict(self.bonds), chiral=dict(self.chiral), next_id=self.next_id,
                    annots={i: (r.kind, r.pos.copy(), r.text, r.scale, r.angle) for i, r in self.annots.items()},
                    annot_next=self.annot_next)

    def _push_history(self):
        self.history.append(self._snapshot())
        del self.history[:-100]

    def undo(self):
        if not self.editing:
            self.say("Switch on Draw mode to undo.")
            return
        if not self.history:
            self.say("Nothing to undo.")
            return
        self.history.pop()
        prev = self.history[-1] if self.history else dict(atoms={}, bonds={}, chiral={}, next_id=0)
        for rec in self.atoms.values():
            self.ren.RemoveActor(rec.actor)
            self.ren.RemoveActor(rec.label)
        self.atoms.clear()
        for aid, (el, p) in prev["atoms"].items():
            self.next_id = aid
            self._create_atom(el, p)
        self.next_id = prev["next_id"]
        self.bonds = dict(prev["bonds"])
        self.chiral = dict(prev["chiral"])
        self._restore_annots(prev.get("annots", {}), prev.get("annot_next", 0))
        self.selected, self.hover_id = set(), None
        self._apply_style()
        self._rebuild_all_bonds()
        self._refresh_halos()
        self._sync(push=False)
        self.say("Undone.")

    def _refresh_stereo_labels(self):
        for a in self.stereo_actors:
            self.ren.RemoveActor(a)
        self.stereo_actors = []
        new = self._compute_wedges()
        if new != self.wedge_info:
            self.wedge_info = new
            self._rebuild_all_bonds()
        if not self.draw2d or self.stereo_mode != 1:
            return
        for aid, c in self.centers.items():
            if c in ("R", "S", "?") and aid in self.atoms:
                t = self._make_label("(?)" if c == "?" else f"({c})",
                                     size=18 if c != "?" else 15,
                                     color=STEREO_COLOR if c != "?" else (0.75, 0.75, 0.75))
                t.SetPosition(*(self.atoms[aid].pos2 + np.array([0.5, 0.45, 0.3])))
                self.stereo_actors.append(t)
        for k, c in self.ez.items():
            if k in self.bonds:
                mid = (self.atoms[k[0]].pos2 + self.atoms[k[1]].pos2) / 2
                t = self._make_label(f"({c})", size=18, color=STEREO_COLOR)
                t.SetPosition(*(mid + np.array([0.0, 0.42, 0.3])))
                self.stereo_actors.append(t)

    def _fit2d(self, movable=None):
        pos = {i: (r.pos[0], r.pos[1]) for i, r in self.atoms.items()}
        out = fit_planar(self._elems(), self.bonds, pos, movable)
        if out is None:
            return False
        for i, (x, y) in out.items():
            r = self.atoms[i]
            r.pos = np.array([x, y, self.z0])
            r.pos2 = r.pos.copy()
            r.sync()
        return True

    def _commit(self, fit=True):
        if fit:
            self._fit2d()
        self._apply_style()
        self._rebuild_all_bonds()
        self._refresh_halos()
        self._sync()

    def _create_atom(self, element, pos):
        aid = self.next_id
        self.next_id += 1
        src = vtk.vtkSphereSource()
        src.SetRadius(radius_of(element))
        src.SetThetaResolution(36)
        src.SetPhiResolution(36)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(src.GetOutputPort())
        act = vtk.vtkActor()
        act.SetMapper(m)
        p = act.GetProperty()
        p.SetColor(*color_of(element))
        p.SetSpecular(0.5)
        p.SetSpecularPower(50)
        act.PickableOff()
        self.ren.AddActor(act)
        label = self._make_label(display_el(element), color=self._lcol(element))
        rec = AtomRec(aid, element, pos, act, src, label)
        rec.sync()
        self.atoms[aid] = rec
        return aid

    def _make_label(self, text, size=26, color=None, bold=True):
        color = self.fg if color is None else color
        t = TextLabel()
        t.SetInput(text)
        tp = t.GetTextProperty()
        tp.SetFontSize(int(round(size)))
        t.base_size = float(size)
        tp.SetBold(bold)
        tp.SetColor(*color)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        t.PickableOff()
        self.ren.AddActor(t)
        return t

    def _bond_ends(self, p1, p2, gap1, gap2):
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        d = p2 - p1
        L = float(np.linalg.norm(d))
        gap = self.label_gap()
        g1 = gap if gap1 else 0.0
        g2 = gap if gap2 else 0.0
        if L <= g1 + g2 + 0.15:
            return p1, p2
        u = d / L
        return p1 + u * g1, p2 - u * g2

    def _make_bond_actor(self, p1, p2, order, radius=None, inward=None):
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        d = p2 - p1
        d = d / np.linalg.norm(d)
        _, _, dop = self.cam_axes()
        perp = np.cross(d, dop)
        if np.linalg.norm(perp) < 1e-3:
            perp = np.cross(d, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp) < 1e-3:
                perp = np.cross(d, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        offs, trims = list(BOND_OFFSETS[order]), [0.0] * len(BOND_OFFSETS[order])
        if order == 2 and inward is not None and self.bond_scale < 0.99:
            dot = float(np.dot(perp, np.asarray(inward, float) - (p1 + p2) / 2))
            if abs(dot) > 0.15:
                offs, trims = [0.0, (1.0 if dot > 0 else -1.0) * 0.2], [0.0, 0.2]
        L = float(np.linalg.norm(p2 - p1))
        for k, off in enumerate(offs):
            t = trims[k] if L > 2 * trims[k] + 0.3 else 0.0
            pts.InsertNextPoint(*(p1 + d * t + perp * off))
            pts.InsertNextPoint(*(p2 - d * t + perp * off))
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2 * k)
            lines.InsertCellPoint(2 * k + 1)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(poly)
        base_r = radius or (BOND_RADIUS[1] if self.bond_scale < 0.99 else BOND_RADIUS[order])
        tube.SetRadius(base_r * self.bond_scale)
        tube.SetNumberOfSides(18)
        tube.CappingOn()
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(tube.GetOutputPort())
        act = vtk.vtkActor()
        act.SetMapper(m)
        act.GetProperty().SetColor(*self.fg)
        act.GetProperty().SetSpecular(0.1)
        act.PickableOff()
        self.ren.AddActor(act)
        return act

    def _make_wedge_actor(self, p1, p2, kind):
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        d = p2 - p1
        L = float(np.linalg.norm(d))
        d = d / max(L, 1e-9)
        _, _, dop = self.cam_axes()
        perp = np.cross(d, dop)
        n = float(np.linalg.norm(perp))
        perp = perp / n if n > 1e-3 else np.array([0.0, 1.0, 0.0])
        hw = 0.30 * max(self.bond_scale, 0.6)
        pts, cells, pd = vtk.vtkPoints(), vtk.vtkCellArray(), vtk.vtkPolyData()
        if kind == "wedge":
            for q in (p1, p2 + perp * hw, p2 - perp * hw):
                pts.InsertNextPoint(*q)
            cells.InsertNextCell(3)
            for i in range(3):
                cells.InsertCellPoint(i)
            pd.SetPoints(pts)
            pd.SetPolys(cells)
        else:
            nd = 8
            for k in range(nd):
                t = (k + 1) / nd
                c, w = p1 + d * L * t, hw * t + 0.01
                i0 = pts.InsertNextPoint(*(c + perp * w))
                i1 = pts.InsertNextPoint(*(c - perp * w))
                cells.InsertNextCell(2)
                cells.InsertCellPoint(i0)
                cells.InsertCellPoint(i1)
            pd.SetPoints(pts)
            pd.SetLines(cells)
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        act = vtk.vtkActor()
        act.SetMapper(m)
        act.GetProperty().SetColor(*self.fg)
        act.GetProperty().SetLighting(False)
        if kind != "wedge":
            act.GetProperty().SetLineWidth(max(2.0, 4.0 * self.bond_scale))
        act.PickableOff()
        self.ren.AddActor(act)
        return act

    def _compute_wedges(self):
        if self.stereo_mode != 2 or self.mol is None or not self.mol.GetNumConformers():
            return {}
        try:
            m = Chem.Mol(self.mol)
            Chem.WedgeMolBonds(m, m.GetConformer())
        except Exception:
            return {}
        id_of = {v: k for k, v in self._mol_idx.items()}
        out = {}
        for b in m.GetBonds():
            d = b.GetBondDir()
            if d in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
                a0, a1 = id_of[b.GetBeginAtomIdx()], id_of[b.GetEndAtomIdx()]
                out[bkey(a0, a1)] = ("wedge" if d == Chem.BondDir.BEGINWEDGE else "dash", a0)
        return out

    def _rebuild_all_bonds(self):
        for act in self.bond_actors.values():
            self.ren.RemoveActor(act)
        self.bond_actors.clear()
        inward = {}
        if self.bond_scale < 0.99 and 2 in self.bonds.values():
            adj = {}
            for (x, y) in self.bonds:
                adj.setdefault(x, []).append(y)
                adj.setdefault(y, []).append(x)
            mol, _, idx = build_mol(self._elems(), self.bonds)
            if mol is not None:
                id_of = {v: k for k, v in idx.items()}
                rings = [[id_of[i] for i in ring] for ring in mol.GetRingInfo().AtomRings()]
                for (a, b), o in self.bonds.items():
                    cands = [r for r in rings if a in r and b in r] if o == 2 else []
                    if cands:
                        r = min(cands, key=len)
                        inward[(a, b)] = np.mean([self.atoms[i].pos for i in r], axis=0)
                    elif o == 2:
                        v, mid = np.zeros(3), (self.atoms[a].pos + self.atoms[b].pos) / 2
                        for e, other in ((a, b), (b, a)):
                            for n in adj.get(e, []):
                                if n != other:
                                    u = self.atoms[n].pos - self.atoms[e].pos
                                    v += u / max(float(np.linalg.norm(u)), 1e-9)
                        if np.linalg.norm(v) > 1e-6:
                            inward[(a, b)] = mid + v
        for (a, b), o in self.bonds.items():
            ra, rb = self.atoms[a], self.atoms[b]
            wi = self.wedge_info.get((a, b)) if self.stereo_mode == 2 else None
            if wi and o == 1:
                kind, beg = wi
                end = b if beg == a else a
                rbeg, rend = self.atoms[beg], self.atoms[end]
                p1, p2 = self._bond_ends(rbeg.pos, rend.pos, rbeg.has_label, rend.has_label)
                self.bond_actors[(a, b)] = self._make_wedge_actor(p1, p2, kind)
                continue
            p1, p2 = self._bond_ends(ra.pos, rb.pos, ra.has_label, rb.has_label)
            self.bond_actors[(a, b)] = self._make_bond_actor(p1, p2, o, inward=inward.get((a, b)))
        self.hover_bond = None
        self._update_bond_hover()

    def place_first(self, x, y):
        if not self.draw2d:
            return
        c = self._world_at(x, y)
        c[2] = self.z0
        if self.ring:
            self.place_template(ring_template(self.ring), c)
        elif self.group:
            self.place_template(group_template(self.group), c)
        else:
            self._create_atom(self.element, c)
            self._commit(fit=False)

    def place_template(self, t, center):
        xy = t["xy"] - t["xy"].mean(axis=0)
        ids = [self._create_atom(s, (center[0] + p[0], center[1] + p[1], self.z0))
               for s, p in zip(t["syms"], xy)]
        for i, j, o in t["bonds"]:
            self.bonds[bkey(ids[i], ids[j])] = o
        self._commit()

    def place_ring(self, name, center):
        self.place_template(ring_template(name), center)

    def attach_ring(self, aid, name):
        self.attach_template(aid, ring_template(name))

    def attach_group(self, aid, name):
        self.attach_template(aid, group_template(name))

    def attach_template(self, aid, t):
        if not self._editable():
            return
        A = self.atoms[aid]
        vecs = []
        for k in self.bonds:
            if aid in k:
                o = self.atoms[k[0] if k[1] == aid else k[1]]
                v = o.pos[:2] - A.pos[:2]
                vecs.append(v / np.linalg.norm(v))
        if vecs:
            m = -np.sum(vecs, axis=0)
            if np.linalg.norm(m) < 0.3:
                m = np.array([-vecs[0][1], vecs[0][0]])
            d0 = m / np.linalg.norm(m)
        else:
            d0 = np.array([math.cos(math.pi / 6), math.sin(math.pi / 6)])
        ang0 = math.atan2(d0[1], d0[0])
        a0, xy, order = t["a0"], t["xy"], t.get("order", 1)
        L = ideal_bond_length(A.element, t["syms"][a0], order)
        rel = xy - xy[a0]
        cv = t["din"]
        th = math.atan2(cv[1], cv[0])
        others = np.array([r.pos[:2] for r in self.atoms.values()])
        pts = None
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6):
            ang = ang0 + k * math.pi / 6
            phi = ang - th
            R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
            cand = A.pos[:2] + L * np.array([math.cos(ang), math.sin(ang)]) + rel @ R.T
            if np.min(np.linalg.norm(cand[:, None, :] - others[None, :, :], axis=2)) >= 0.9:
                pts = cand
                break
        if pts is None:
            self.say("No room to attach it there.")
            return
        base_id = self.next_id
        elems = self._elems()
        for j, s in enumerate(t["syms"]):
            elems[base_id + j] = s
        bonds = dict(self.bonds)
        for i, j, o in t["bonds"]:
            bonds[bkey(base_id + i, base_id + j)] = o
        bonds[bkey(aid, base_id + a0)] = order
        if not self._check(elems, bonds):
            return
        for s, p in zip(t["syms"], pts):
            self._create_atom(s, (p[0], p[1], self.z0))
        self.bonds = bonds
        self._commit()

    def ghost_position(self, src, x, y):
        rec = self.atoms[src]
        sd = self.world_to_display(rec.pos)
        step = math.pi / 6
        ang = round(math.atan2(y - sd[1], x - sd[0]) / step) * step
        right, up, _ = self.cam_axes()
        d = math.cos(ang) * right + math.sin(ang) * up
        L = ideal_bond_length(rec.element, self.element, self.bond_order)
        return rec.pos + L * d, L

    def clashes(self, pos, ignore, L):
        return any(np.linalg.norm(r.pos - pos) < 0.6 * L for i, r in self.atoms.items() if i != ignore)

    def add_atom_bonded(self, src, element, pos, order):
        L = ideal_bond_length(self.atoms[src].element, element, order)
        if self.clashes(pos, src, L):
            self.say("Position overlaps another atom - drag in another direction.")
            return
        new_id = self.next_id
        elems = self._elems()
        elems[new_id] = element
        bonds = dict(self.bonds)
        bonds[bkey(src, new_id)] = order
        if not self._check(elems, bonds):
            return
        self._create_atom(element, pos)
        self.bonds = bonds
        self._commit()

    def connect(self, a, b, order):
        k = bkey(a, b)
        if self.bonds.get(k) == order:
            self.say("Those atoms are already joined by that bond.")
            return
        bonds = dict(self.bonds)
        bonds[k] = order
        if not self._check(self._elems(), bonds):
            return
        self.bonds = bonds
        self._commit()

    def replace_atom(self, aid, element):
        self._set_element(aid, element)

    def _set_element(self, aid, element):
        elems = self._elems()
        elems[aid] = element
        if not self._check(elems, self.bonds):
            return False
        rec = self.atoms[aid]
        rec.element = element
        rec.label.SetInput(display_el(element))
        rec.label.GetTextProperty().SetColor(*self._lcol(element))
        self._commit()
        return True

    def set_charge(self, aid, q):
        if not self._editable() or aid not in self.atoms:
            return
        sym = base(self.atoms[aid].element)
        new = el_str(sym, q)
        if new == self.atoms[aid].element:
            return
        if self._set_element(aid, new):
            self.say(f"{sym} formal charge set to {q:+d}." if q else f"{sym} made neutral.")

    def charge_dialog(self, aid):
        if not self._editable() or aid not in self.atoms:
            return
        sym, q = split_el(self.atoms[aid].element)
        dlg = ChargeDialog(sym, q, self.widget)
        if dlg.exec():
            self.set_charge(aid, dlg.value())

    def delete_atoms(self, ids):
        if not self._editable():
            return
        ids = [i for i in ids if i in self.atoms]
        if not ids:
            return
        idset = set(ids)
        self.bonds = {k: o for k, o in self.bonds.items() if k[0] not in idset and k[1] not in idset}
        for i in ids:
            rec = self.atoms.pop(i)
            self.ren.RemoveActor(rec.actor)
            self.ren.RemoveActor(rec.label)
            self.chiral.pop(i, None)
        self.selected -= idset
        if self.hover_id in idset:
            self.hover_id = None
        self._commit()
        self.say(f"Deleted {len(ids)} atom(s).")

    def delete_selected(self):
        if not (self.selected or self.sel_annots) or not self._editable():
            return
        ids, ann = list(self.selected), list(self.sel_annots)
        if ann:
            self._remove_annots(ann)
        if ids:
            self.delete_atoms(ids)
        elif ann:
            self._push_history()
            self.render()
            self.notify(f"Deleted {len(ann)} label(s).")

    def set_chirality(self, aid, want):
        if not self._editable() or aid not in self.atoms:
            return
        if want is None:
            self.chiral.pop(aid, None)
        else:
            self.chiral[aid] = want
        self._commit(fit=False)
        got = self.centers.get(aid, "?")
        if want and got != want:
            self.say(f"Could not set {want} on that atom (result: {got}).")
        else:
            self.say("Chirality cleared." if want is None else f"Set {want}.")

    def set_ez(self, k, want):
        if not self._editable() or k not in self.bonds:
            return
        if self.ez.get(k) == want:
            return
        a, b = k
        adj = {}
        for (u, v) in self.bonds:
            if (u, v) == k:
                continue
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        side, stack = {b}, [b]
        while stack:
            n = stack.pop()
            for m in adj.get(n, []):
                if m not in side:
                    side.add(m)
                    stack.append(m)
        if a in side:
            self.say("This double bond is in a ring - E/Z cannot be flipped here.")
            return
        pa = self.atoms[a].pos2[:2].copy()
        pb = self.atoms[b].pos2[:2].copy()
        d = (pb - pa) / np.linalg.norm(pb - pa)
        for i in side:
            r = self.atoms[i]
            v = r.pos2[:2] - pa
            q = pa + 2 * np.dot(v, d) * d - v
            r.pos = np.array([q[0], q[1], self.z0])
            r.pos2 = r.pos.copy()
            r.sync()
        self._commit(fit=False)
        self.say(f"Double bond set to {self.ez.get(k, want)}.")

    def minimize_selection(self):
        if not self.selected:
            self.say("Select atoms first (right-drag a box).")
            return
        self._minimize(set(self.selected))

    def minimize_all(self):
        self._minimize(None)

    def _update_hover(self):
        if self.hover_id is None or self.hover_id not in self.atoms:
            self.hover_halo.VisibilityOff()
            return
        r = self.atoms[self.hover_id]
        self.hover_halo.SetPosition(*r.pos)
        s = radius_of(r.element) * 1.35
        self.hover_halo.SetScale(s, s, s)
        self.hover_halo.VisibilityOn()

    def _refresh_halos(self):
        for h in self.sel_halos.values():
            self.ren.RemoveActor(h)
        self.sel_halos.clear()
        self.selected &= set(self.atoms)
        for i in self.selected:
            r = self.atoms[i]
            h = self._make_halo((1.0, 0.2, 0.2), 0.45)
            h.SetPosition(*r.pos)
            s = radius_of(r.element) * 1.3
            h.SetScale(s, s, s)
            h.VisibilityOn()
            self.sel_halos[i] = h
        self._update_hover()

    def components(self):
        adj = {i: set() for i in self.atoms}
        for a, b in self.bonds:
            if a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)
        seen, comps = set(), []
        for i in sorted(adj):
            if i in seen:
                continue
            stack, comp = [i], set()
            while stack:
                n = stack.pop()
                if n in comp:
                    continue
                comp.add(n)
                stack.extend(adj[n] - comp)
            seen |= comp
            comps.append(comp)
        return comps

    def select_all(self):
        self.set_selection(list(self.atoms), list(self.annots))

    def add_selected_to_table(self):
        if not self.atoms:
            return
        if not self.selected:
            self.select_all()
        comps = self.selected_compounds()
        if comps:
            self.tableRequested.emit(comps)

    def dock_selected(self):
        if not self.atoms:
            return
        if not self.selected:
            self.select_all()
        comps = self.selected_compounds()
        if comps:
            self.dockRequested.emit(comps)

    def selected_compounds(self):
        if not self.selected:
            return []
        out = []
        covered = set()
        for comp in self.components():
            if comp & self.selected:
                if not comp <= self.selected:
                    return []
                covered |= comp
                elems = {i: self.atoms[i].element for i in comp}
                bonds = {k: v for k, v in self.bonds.items() if k[0] in comp and k[1] in comp}
                pos = {i: (self.atoms[i].pos2[0], self.atoms[i].pos2[1]) for i in comp}
                chiral = {i: v for i, v in self.chiral.items() if i in comp}
                mol, err, _ = build_mol(elems, bonds, pos, chiral)
                if mol is None:
                    return []
                try:
                    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
                except Exception:
                    smi = Chem.MolToSmiles(mol)
                out.append({"smiles": smi})
        return out if covered == self.selected else []

    def set_selection(self, ids, annots=()):
        self.selected = set(ids)
        self.sel_annots = set(annots) & set(self.annots)
        self._refresh_halos()
        self._refresh_annot_colors()
        self.render()

    def clear_selection(self):
        if self.selected or self.sel_annots:
            self.set_selection([])

    def _annot_color(self, rec):
        return (1.0, 0.2, 0.2) if rec.id in self.sel_annots else self.fg

    def _refresh_annot_colors(self):
        for rec in self.annots.values():
            col = self._annot_color(rec)
            for a in rec.actors:
                if isinstance(a, TextLabel):
                    a.GetTextProperty().SetColor(*col)
                else:
                    a.GetProperty().SetColor(*col)

    def _make_annot(self, aid, kind, pos, text, scale=1.0, angle=0.0):
        pos = np.array([pos[0], pos[1], self.z0], float)
        if kind in ("text", "plus"):
            text = "+" if kind == "plus" else text
            size = (40 if kind == "plus" else 24) * scale
            t = self._make_label(text, size=int(round(size)), bold=self.text_bold)
            t.SetPosition(*pos)
            rec = AnnotRec(aid, kind, pos, text, [t], size=size, scale=scale)
        else:
            lines, segs = arrow_polylines(kind, L=ARROW_LEN * scale, angle=angle)
            pts, cells = vtk.vtkPoints(), vtk.vtkCellArray()
            for pl in lines:
                cells.InsertNextCell(len(pl))
                for (dx, dy) in pl:
                    cells.InsertCellPoint(pts.InsertNextPoint(pos[0] + dx, pos[1] + dy, pos[2]))
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetLines(cells)
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(pd)
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.PickableOff()
            a.GetProperty().SetLineWidth(3.0)
            a.GetProperty().SetLighting(False)
            self.ren.AddActor(a)
            rec = AnnotRec(aid, kind, pos, "", [a], segs=segs, scale=scale, angle=angle)
        self.annots[aid] = rec
        for a in rec.actors:
            if isinstance(a, TextLabel):
                a.GetTextProperty().SetColor(*self._annot_color(rec))
            else:
                a.GetProperty().SetColor(*self._annot_color(rec))
        return rec

    def _restore_annots(self, data, next_id):
        for rec in self.annots.values():
            for a in rec.actors:
                self.ren.RemoveViewProp(a)
        self.annots, self.sel_annots = {}, set()
        for aid, item in data.items():
            kind, pos, text = item[:3]
            self._make_annot(aid, kind, pos, text, item[3] if len(item) > 3 else 1.0,
                             item[4] if len(item) > 4 else 0.0)
        self.annot_next = next_id

    def add_annot(self, kind, xy, text="", scale=1.0):
        if not self._editable():
            return None
        pos = self._world_at(*xy)
        aid = self.annot_next
        self.annot_next += 1
        self._make_annot(aid, kind, pos, text, scale)
        self._push_history()
        self.render()
        return aid

    def add_text_label(self, xy):
        dlg = TextLabelDialog("", "Text label", self.widget)
        if dlg.exec() and dlg.value():
            self.add_annot("text", xy, dlg.value())

    def _apply_text_weight(self):
        for rec in self.annots.values():
            for a in rec.actors:
                if isinstance(a, TextLabel):
                    a.GetTextProperty().SetBold(self.text_bold)

    def set_text_thickness(self, bold):
        self.text_bold = bool(bold)
        self._apply_text_weight()
        self.render()
        self.notify("Text labels: " + ("bold" if self.text_bold else "thin"))

    def set_bond_thickness(self, factor=None, scale_by=None):
        f = self.bond_scale if factor is None else factor
        if scale_by is not None:
            f *= scale_by
        self.bond_scale = float(np.clip(f, 0.25, 3.0))
        self._apply_style()
        self._rebuild_all_bonds()
        self.render()
        self.notify(f"Bond thickness x{self.bond_scale:.2f}")

    def scene_dict(self):
        cam = self.ren.GetActiveCamera()
        return {
            "format": "pad", "version": 1,
            "atoms": [{"id": i, "element": r.element, "x": float(r.pos2[0]), "y": float(r.pos2[1])}
                      for i, r in sorted(self.atoms.items())],
            "bonds": [[a, b, o] for (a, b), o in sorted(self.bonds.items())],
            "chiral": {str(i): c for i, c in self.chiral.items()},
            "annots": [{"id": i, "kind": r.kind, "x": float(r.pos[0]), "y": float(r.pos[1]),
                        "text": r.text, "scale": r.scale, "angle": r.angle}
                       for i, r in sorted(self.annots.items())],
            "next_id": self.next_id, "annot_next": self.annot_next,
            "bond_scale": self.bond_scale, "text_bold": self.text_bold, "dark": self.dark, "mono_letters": self.mono_letters, "stereo_mode": self.stereo_mode,
            "smiles": self.smiles,
            "camera": {"position": list(cam.GetPosition()), "focal": list(cam.GetFocalPoint()),
                       "viewup": list(cam.GetViewUp()), "angle": cam.GetViewAngle()},
        }

    def load_scene_dict(self, d):
        if d.get("format") not in ("pad", "ca"):
            raise ValueError("not a .pad scene file")
        self.clear()
        self.set_dark(bool(d.get("dark", self.dark)))
        self.bond_scale = float(d.get("bond_scale", 1.0))
        self.text_bold = bool(d.get("text_bold", self.bond_scale >= 0.99))
        self.stereo_mode = int(d.get("stereo_mode", 1 if d.get("show_stereo", True) else 0))
        self.mono_letters = bool(d.get("mono_letters", False))
        for a in d.get("atoms", []):
            self.next_id = int(a["id"])
            self._create_atom(a["element"], (a["x"], a["y"], self.z0))
        self.next_id = int(d.get("next_id", max([int(a["id"]) for a in d.get("atoms", [])] + [-1]) + 1))
        self.bonds = {bkey(int(a), int(b)): int(o) for a, b, o in d.get("bonds", [])}
        self.chiral = {int(i): c for i, c in d.get("chiral", {}).items()}
        for n in d.get("annots", []):
            self._make_annot(int(n["id"]), n["kind"], (n["x"], n["y"]), n.get("text", ""),
                             n.get("scale", 1.0), n.get("angle", 0.0))
        self.annot_next = int(d.get("annot_next", max([int(n["id"]) for n in d.get("annots", [])] + [-1]) + 1))
        self._commit(fit=False)
        cam, c = self.ren.GetActiveCamera(), d.get("camera")
        if c:
            cam.SetPosition(*c["position"])
            cam.SetFocalPoint(*c["focal"])
            cam.SetViewUp(*c["viewup"])
            cam.SetViewAngle(c.get("angle", cam.GetViewAngle()))
            self.ren.ResetCameraClippingRange()
        self.render()
        return bool(c)

    STEREO_MODES = ("No stereo marks", "Stereo labels (R) (S) (E) (Z)", "Wedge / dash bonds at R/S centres")

    def toggle_mono_letters(self):
        self.mono_letters = not self.mono_letters
        for r in self.atoms.values():
            r.label.GetTextProperty().SetColor(*self._lcol(r.element))
        self.render()
        self.notify("Atom letters: " + ("black" if self.mono_letters else "coloured") + "  (Ctrl+G toggles)")

    def cycle_stereo_display(self):
        self.stereo_mode = (self.stereo_mode + 1) % 3
        self._refresh_stereo_labels()
        self.render()
        self.notify(f"Stereo display: {self.STEREO_MODES[self.stereo_mode]}  (Ctrl+H cycles)")

    def toggle_stereo_labels(self):
        self.cycle_stereo_display()

    def _set_annot_scale(self, rec, scale):
        scale = float(np.clip(scale, 0.2, 8.0))
        if rec.kind in ("text", "plus"):
            rec.scale = scale
            rec.size = rec.base_size * scale
            rec.actors[0].base_size = rec.size
        else:
            for a in rec.actors:
                self.ren.RemoveViewProp(a)
            self._make_annot(rec.id, rec.kind, rec.pos, rec.text, scale, rec.angle)

    def begin_scale(self, x, y):
        if self.scaling is not None:
            return True
        if not (self.selected or self.sel_annots) or not self._editable():
            return False
        ids = [i for i in self.selected if i in self.atoms]
        centre = np.mean([self.atoms[i].pos2 for i in ids], axis=0) if ids else None
        self.scaling = dict(x0=x, y0=y, f=1.0, centre=centre,
                            atoms={i: self.atoms[i].pos2.copy() for i in ids},
                            annots={i: self.annots[i].scale for i in self.sel_annots if i in self.annots})
        return True

    def scale_to(self, x, y):
        sc = self.scaling
        if sc is None:
            return
        f = float(np.clip(math.exp(0.004 * ((x - sc["x0"]) + (y - sc["y0"]))), 0.2, 6.0))
        sc["f"] = f
        for i, p0 in sc["atoms"].items():
            r = self.atoms[i]
            r.pos = sc["centre"] + (p0 - sc["centre"]) * f
            r.pos2 = r.pos.copy()
            r.sync()
        for i, s0 in sc["annots"].items():
            if i in self.annots:
                self._set_annot_scale(self.annots[i], s0 * f)
        if sc["atoms"]:
            self._rebuild_all_bonds()
        self._refresh_halos()
        self.render()

    def end_scale(self):
        sc, self.scaling = self.scaling, None
        if sc is not None and abs(sc["f"] - 1.0) > 1e-3:
            self._commit(fit=False)
            self.notify("Resized selection.")

    def edit_text_label(self, aid):
        rec = self.annots.get(aid)
        if rec is None or rec.kind != "text" or not self._editable():
            return
        dlg = TextLabelDialog(rec.text, "Edit text label", self.widget)
        text = dlg.value() if dlg.exec() else ""
        if text and text != rec.text:
            rec.text = text
            rec.actors[0].SetInput(text)
            self._push_history()
            self.render()

    def begin_rotate(self, x, y):
        if self.rotating is not None:
            return True
        if not (self.selected or self.sel_annots) or not self._editable():
            return False
        ids = [i for i in self.selected if i in self.atoms]
        centre = np.mean([self.atoms[i].pos2 for i in ids], axis=0) if ids else None
        arrows = {i: self.annots[i].angle for i in self.sel_annots
                  if i in self.annots and self.annots[i].kind not in ("text", "plus")}
        if not ids and not arrows:
            return False
        self.rotating = dict(x0=x, y0=y, a=0.0, centre=centre, arrows=arrows,
                             atoms={i: self.atoms[i].pos2.copy() for i in ids})
        return True

    def rotate_to(self, x, y):
        rt = self.rotating
        if rt is None:
            return
        a = 0.01 * ((x - rt["x0"]) + (y - rt["y0"]))
        rt["a"] = a
        c, sn = math.cos(a), math.sin(a)
        for i, p0 in rt["atoms"].items():
            d = p0 - rt["centre"]
            r = self.atoms[i]
            r.pos = rt["centre"] + np.array([d[0] * c - d[1] * sn, d[0] * sn + d[1] * c, 0.0])
            r.pos2 = r.pos.copy()
            r.sync()
        for i, a0 in rt["arrows"].items():
            rec = self.annots.get(i)
            if rec is not None:
                for act in rec.actors:
                    self.ren.RemoveViewProp(act)
                self._make_annot(i, rec.kind, rec.pos, rec.text, rec.scale, a0 + a)
        if rt["atoms"]:
            self._rebuild_all_bonds()
        self._refresh_halos()
        self.render()

    def end_rotate(self):
        rt, self.rotating = self.rotating, None
        if rt is not None and abs(rt["a"]) > 1e-3:
            self._commit(fit=False)
            self.notify("Rotated selection.")

    def duplicate_selected(self):
        if not (self.selected or self.sel_annots) or not self._editable():
            return
        d = np.array([2.0, -1.5, 0.0])
        old = sorted(i for i in self.selected if i in self.atoms)
        new = {}
        for i in old:
            r = self.atoms[i]
            new[i] = self._create_atom(r.element, r.pos2 + d)
        for (a, b), o in list(self.bonds.items()):
            if a in new and b in new:
                self.bonds[bkey(new[a], new[b])] = o
        for i, c in list(self.chiral.items()):
            if i in new:
                self.chiral[new[i]] = c
        new_ann = []
        for i in sorted(self.sel_annots):
            rec = self.annots.get(i)
            if rec is None:
                continue
            aid = self.annot_next
            self.annot_next += 1
            self._make_annot(aid, rec.kind, rec.pos + d, rec.text, rec.scale, rec.angle)
            new_ann.append(aid)
        self.selected = set(new.values())
        self.sel_annots = set(new_ann)
        self._commit(fit=False)
        self._refresh_annot_colors()
        self.render()
        self.notify(f"Duplicated {len(new)} atom(s) and {len(new_ann)} label(s).")

    def _remove_annots(self, ids):
        for i in ids:
            rec = self.annots.pop(i, None)
            if rec is not None:
                for a in rec.actors:
                    self.ren.RemoveViewProp(a)
            self.sel_annots.discard(i)

    def delete_annots(self, ids):
        if not self._editable():
            return
        ids = [i for i in ids if i in self.annots]
        if not ids:
            return
        self._remove_annots(ids)
        self._push_history()
        self.render()
        self.notify(f"Deleted {len(ids)} label(s).")

    def pick_annot(self, x, y):
        best, best_d = None, 1e9
        p = np.array([x, y], float)
        for i, r in self.annots.items():
            if r.kind in ("text", "plus"):
                d = self.world_to_display(r.pos)
                fs = r.actors[0].GetTextProperty().GetFontSize()
                hw = 0.33 * fs * max(len(r.text), 1) + 8
                hh = 0.45 * fs + 6
                if abs(x - d[0]) <= hw and abs(y - d[1]) <= hh:
                    dist = math.hypot(x - d[0], y - d[1])
                    if dist < best_d:
                        best, best_d = i, dist
            else:
                for a, b in r.segs:
                    da = np.array(self.world_to_display(r.pos + a)[:2])
                    db = np.array(self.world_to_display(r.pos + b)[:2])
                    ab = db - da
                    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - da, ab) / np.dot(ab, ab), 0, 1))
                    dist = float(np.linalg.norm(p - (da + t * ab)))
                    if dist < 10.0 and dist < best_d:
                        best, best_d = i, dist
        return best

    def _check_moved(self, x, y):
        px, py = self.press_pos
        if math.hypot(x - px, y - py) > DRAG_THRESHOLD:
            self.moved = True

    def cycle_bond(self, k):
        cur = self.bonds.get(k)
        if cur is None:
            return
        for nxt in (cur % 3 + 1, (cur + 1) % 3 + 1):
            if nxt == cur:
                continue
            bonds = dict(self.bonds)
            bonds[k] = nxt
            mol, _, _ = build_mol(self._elems(), bonds)
            if mol is not None:
                self.bonds = bonds
                self._commit()
                self.say(f"{BOND_NAMES[cur]} -> {BOND_NAMES[nxt]}.")
                return
        self.say("Bond order can't change: it would exceed an atom's valence.")

    def _update_preview(self, x, y):
        src = self.drag_src
        if src not in self.atoms or not self.moved:
            return
        sp = self.atoms[src].pos
        tgt = self.pick_atom(x, y)
        if tgt is not None and tgt != src:
            end = self.atoms[tgt].pos
            self.ghost.VisibilityOff()
            self.hover_id = tgt
        else:
            self.hover_id = src
            if tgt == src:
                self._clear_preview()
                self._update_hover()
                self.rw.Render()
                return
            end, L = self.ghost_position(src, x, y)
            bad = self.clashes(end, src, L)
            self.ghost.GetProperty().SetColor(*((1.0, 0.1, 0.1) if bad else color_of(self.element)))
            s = radius_of(self.element)
            self.ghost.SetPosition(*end)
            self.ghost.SetScale(s, s, s)
            self.ghost.VisibilityOn()
        self.preview_line.SetPoint1(*sp)
        self.preview_line.SetPoint2(*end)
        self.preview_actor.VisibilityOn()
        self._update_hover()
        self.rw.Render()

    def _clear_preview(self):
        self.preview_actor.VisibilityOff()
        self.ghost.VisibilityOff()

    def show_menu(self, atom_id, bond_key, xy):
        menu = QtWidgets.QMenu(self.widget)
        n_sel = len(self.selected)

        if atom_id is not None and atom_id in self.atoms:
            r = self.atoms[atom_id]
            act = menu.addAction(f"Delete atom ({r.element}{atom_id + 1})")
            act.triggered.connect(lambda checked=False, i=atom_id: self.delete_atoms([i]))
            q = split_el(r.element)[1]
            ch = menu.addAction(f"Set formal charge... (now {q:+d})")
            ch.triggered.connect(lambda checked=False, i=atom_id: self.charge_dialog(i))
        if self.atoms:
            a = menu.addAction("Select all drawn atoms")
            a.triggered.connect(lambda checked=False: self.select_all())
        comps = self.selected_compounds() if n_sel else []
        if comps:
            n = len(comps)
            word = "compound" if n == 1 else "compounds"
            menu.addSection("Docking")
            a = menu.addAction(f"Dock {n} {word}   Ctrl+R")
            a.triggered.connect(lambda checked=False, c=comps: self.dockRequested.emit(c))
            a = menu.addAction(f"Add {n} {word} to the docking table (not docked)   Ctrl+A")
            a.triggered.connect(lambda checked=False, c=comps: self.tableRequested.emit(c))
            menu.addSeparator()
        if n_sel:
            a = menu.addAction(f"Delete {n_sel} selected   Ctrl+X")
            a.triggered.connect(lambda checked=False: self.delete_selected())
            b = menu.addAction(f"Energy minimize {n_sel} selected   Ctrl+M")
            b.triggered.connect(lambda checked=False: self.minimize_selection())

        if self.selected or self.sel_annots:
            a = menu.addAction("Duplicate selection   Ctrl+D")
            a.triggered.connect(lambda checked=False: self.duplicate_selected())
            if not n_sel:
                a = menu.addAction(f"Delete {len(self.sel_annots)} selected label(s)   Ctrl+X")
                a.triggered.connect(lambda checked=False: self.delete_selected())

        an = self.pick_annot(*xy)
        if an is not None:
            if self.annots[an].kind == "text":
                a = menu.addAction(f'Edit text label "{self.annots[an].text}"...')
                a.triggered.connect(lambda checked=False, i=an: self.edit_text_label(i))
            a = menu.addAction(f"Delete {self.annots[an].describe()}")
            a.triggered.connect(lambda checked=False, i=an: self.delete_annots([i]))
        menu.addSection("Reaction scheme")
        for label, kind in (("arrow  \u2192", "arrow"), ("equilibrium arrow  \u21cc", "eq"),
                            ("double-headed arrow  \u2194", "dbl"), ("plus sign  +", "plus")):
            a = menu.addAction(f"Add {label}")
            a.triggered.connect(lambda checked=False, k=kind: self.add_annot(k, xy))
        a = menu.addAction("Add heat symbol  \u0394")
        a.triggered.connect(lambda checked=False: self.add_annot("text", xy, "\u0394", 1.7))
        a = menu.addAction("Add text label...")
        a.triggered.connect(lambda checked=False: self.add_text_label(xy))

        if atom_id is not None and atom_id in self.atoms:
            if atom_id in self.centers:
                cur = self.centers[atom_id]
                menu.addSection(f"R/S - stereocentre (now {cur if cur in ('R', 'S') else 'unspecified'})")
                for lab in ("R", "S"):
                    a = menu.addAction(f"Set {lab}")
                    a.setCheckable(True)
                    a.setChecked(cur == lab)
                    a.triggered.connect(lambda checked=False, i=atom_id, l=lab: self.set_chirality(i, l))
                if atom_id in self.chiral:
                    a = menu.addAction("Clear chirality")
                    a.triggered.connect(lambda checked=False, i=atom_id: self.set_chirality(i, None))
            else:
                menu.addSection("R/S")
                a = menu.addAction("Not a stereocentre (needs 4 different groups)")
                a.setEnabled(False)
        if bond_key is not None and bond_key in self.bonds and self.bonds[bond_key] == 2:
            if bond_key in self.ez:
                cur = self.ez[bond_key]
                menu.addSection(f"E/Z - double bond (now {cur})")
                for lab in ("E", "Z"):
                    a = menu.addAction(f"Set {lab}")
                    a.setCheckable(True)
                    a.setChecked(cur == lab)
                    a.triggered.connect(lambda checked=False, k=bond_key, l=lab: self.set_ez(k, l))
            else:
                menu.addSection("E/Z")
                a = menu.addAction("Not a stereogenic double bond")
                a.setEnabled(False)

        place_xy = xy if (atom_id is None and bond_key is None) else None
        grp = QtGui.QActionGroup(menu)
        grp.setExclusive(True)

        menu.addSection("Atom")

        def add_el(m, el):
            a = m.addAction(el)
            a.setCheckable(True)
            a.setChecked(self.ring is None and self.group is None and el == self.element)
            grp.addAction(a)
            a.triggered.connect(lambda checked=False, e=el: self._choose_element(e, place_xy))

        for el in PRIMARY_ELEMENTS:
            add_el(menu, el)
        more = menu.addMenu("More atoms")
        for el in MORE_ELEMENTS:
            add_el(more, el)
        pt = menu.addAction("Periodic table...")
        pt.triggered.connect(lambda checked=False: self.open_periodic_table(place_xy))

        menu.addSection("Rings")
        rings = menu.addMenu("Rings")
        for gname, items in RING_GROUPS.items():
            sub = rings.addMenu(gname)
            for name, _ in items:
                a = sub.addAction(name)
                a.setCheckable(True)
                a.setChecked(self.ring == name)
                a.triggered.connect(
                    lambda checked=False, n=name: self._choose_ring(n, place_xy, atom_id))

        menu.addSection("Functional groups")
        groups = menu.addMenu("Functional groups")
        for gname, items in GROUP_CATEGORIES.items():
            sub = groups.addMenu(gname)
            for name, _ in items:
                a = sub.addAction(name)
                a.triggered.connect(
                    lambda checked=False, n=name: self._choose_group(n, place_xy, atom_id, bond_key, xy))

        menu.addSection("Bond thickness")
        tm = menu.addMenu(f"Bond thickness (now x{self.bond_scale:.2f})")
        for label, kw in (("Thinner", dict(scale_by=0.8)), ("Thicker", dict(scale_by=1.25)),
                          ("Default (thin)  x0.5", dict(factor=DEFAULT_BOND_SCALE)),
                          ("Normal  x1.0", dict(factor=1.0)), ("Bold  x1.6", dict(factor=1.6))):
            a = tm.addAction(label)
            a.triggered.connect(lambda checked=False, k=kw: self.set_bond_thickness(**k))

        tt = menu.addMenu("Text label thickness (now " + ("bold" if self.text_bold else "thin") + ")")
        tg = QtGui.QActionGroup(tt)
        tg.setExclusive(True)
        for label, bold in (("Thin", False), ("Bold", True)):
            a = tt.addAction(label)
            a.setCheckable(True)
            a.setChecked(self.text_bold == bold)
            tg.addAction(a)
            a.triggered.connect(lambda checked=False, b=bold: self.set_text_thickness(b))

        menu.addSection("Bond")
        bgrp = QtGui.QActionGroup(menu)
        bgrp.setExclusive(True)
        for order, name in BOND_NAMES.items():
            a = menu.addAction(name)
            a.setCheckable(True)
            a.setChecked(order == self.bond_order)
            bgrp.addAction(a)
            a.triggered.connect(lambda checked=False, o=order: self.set_bond(o))

        menu.exec(QtGui.QCursor.pos())

    def _choose_element(self, el, place_xy):
        self.set_element(el)
        if place_xy is not None and not self.atoms:
            self.place_first(*place_xy)

    def _choose_ring(self, name, place_xy, atom_id):
        self.set_ring(name)
        if place_xy is not None:
            self.place_first(*place_xy)
        elif atom_id is not None and atom_id in self.atoms:
            self.attach_ring(atom_id, name)
        else:
            self.say(f"{name}: click an atom to attach it.")

    def _group_target(self, atom_id, bond_key, xy):
        if atom_id is not None and atom_id in self.atoms:
            return atom_id
        if len(self.selected) == 1:
            return next(iter(self.selected))
        if bond_key is not None and bond_key in self.bonds:
            x, y = xy
            def dist(i):
                d = self.world_to_display(self.atoms[i].pos)
                return math.hypot(d[0] - x, d[1] - y)
            return min(bond_key, key=dist)
        return None

    def _choose_group(self, name, place_xy, atom_id, bond_key=None, xy=None):
        if not self._editable():
            return
        t = group_template(name)
        target = self._group_target(atom_id, bond_key, xy or (0, 0))
        if target is not None:
            self.attach_template(target, t)
        elif place_xy is not None:
            c = self._world_at(*place_xy)
            c[2] = self.z0
            self.place_template(t, c)
        else:
            self.say(f"{name}: select an atom to attach it.")
