import json
import math
import re

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor

from chempad_render import (
    BOND_LEN, LINE_W, GENERIC, ARROW_LEN, SceneRenderer, annot_bbox, arrow_geometry, bezier_points,
    split_el, base, el_str, display_el, is_generic, text_layout, LABEL_PT)

Qt = QtCore.Qt

_SHORT_RULES = [
    ("fail", "Failed"), ("invalid", "Invalid"), ("rejected", "Invalid"), ("valence", "Blocked"),
    ("no room", "Blocked"), ("overlap", "Blocked"), ("already", "Same"), ("nothing", "Empty"),
    ("empty", "Empty"), ("skipped", "Skipped"), ("undone", "Undone"), ("deleted", "Deleted"),
    ("moved", "Moved"), ("rotated", "Rotated"), ("resized", "Resized"), ("pasted", "Pasted"),
    ("copied", "Copied"), ("saved", "Saved"), ("loaded", "Loaded"), ("selected", "Selected"),
    ("clean", "Cleaned"), ("colour", "Coloured"), ("duplicated", "Duplicated"),
]


def short(msg):
    """Status bar text: always a single word."""
    low = str(msg).lower()
    for key, word in _SHORT_RULES:
        if key in low:
            return word
    w = re.sub(r"[^A-Za-z]", "", str(msg).split()[0]) if str(msg).split() else ""
    return w or "OK"


PRIMARY_ELEMENTS = ["C", "N", "O", "S"]
MORE_ELEMENTS = ["H", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"]
GENERIC_MENU = [("R group", ["R", "R1", "R2", "R3", "R4"]), ("X group", ["X"]), ("Y group", ["Y"]),
                ("Z group", ["Z"])]

BOND_NAMES = {1: "Single bond", 2: "Double bond", 3: "Triple bond"}
BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
DRAG_THRESHOLD = 4
OLD_UNIT = BOND_LEN / 1.5

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
        ("Indolin-2-one (oxindole)", "O=C1Cc2ccccc2N1"), ("Isatin", "O=C1Nc2ccccc2C1=O"),
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
        ("Hydrazine (-NHNH2)", "*NN"), ("Hydrazone (=N-NH2)", "*=NN"), ("Oxime (-CH=NOH)", "*C=NO"),
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

AMINO_ACIDS = [
    ("Glycine (Gly, G)", "NCC(=O)O"),
    ("Alanine (Ala, A)", "N[C@@H](C)C(=O)O"),
    ("Valine (Val, V)", "N[C@@H](C(C)C)C(=O)O"),
    ("Leucine (Leu, L)", "N[C@@H](CC(C)C)C(=O)O"),
    ("Isoleucine (Ile, I)", "N[C@@H]([C@@H](C)CC)C(=O)O"),
    ("Proline (Pro, P)", "OC(=O)[C@@H]1CCCN1"),
    ("Phenylalanine (Phe, F)", "N[C@@H](Cc1ccccc1)C(=O)O"),
    ("Tryptophan (Trp, W)", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
    ("Methionine (Met, M)", "N[C@@H](CCSC)C(=O)O"),
    ("Serine (Ser, S)", "N[C@@H](CO)C(=O)O"),
    ("Threonine (Thr, T)", "N[C@@H]([C@@H](C)O)C(=O)O"),
    ("Cysteine (Cys, C)", "N[C@@H](CS)C(=O)O"),
    ("Tyrosine (Tyr, Y)", "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
    ("Asparagine (Asn, N)", "N[C@@H](CC(N)=O)C(=O)O"),
    ("Glutamine (Gln, Q)", "N[C@@H](CCC(N)=O)C(=O)O"),
    ("Aspartic acid (Asp, D)", "N[C@@H](CC(=O)O)C(=O)O"),
    ("Glutamic acid (Glu, E)", "N[C@@H](CCC(=O)O)C(=O)O"),
    ("Lysine (Lys, K)", "N[C@@H](CCCCN)C(=O)O"),
    ("Arginine (Arg, R)", "N[C@@H](CCCNC(N)=N)C(=O)O"),
    ("Histidine (His, H)", "N[C@@H](Cc1c[nH]cn1)C(=O)O"),
]
AMINO_SMILES = dict(AMINO_ACIDS)


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

_PT = Chem.GetPeriodicTable()
LONE_PAIR_Z = {6, 7, 8, 9, 15, 16, 17, 33, 34, 35, 52, 53}


def bkey(a, b):
    return (a, b) if a < b else (b, a)


def rd_xy(x, y):
    """model (y down) -> RDKit frame (y up)."""
    return (float(x), -float(y))


def _make_atom(el):
    sym, q = split_el(el)
    if sym in GENERIC:
        a = Chem.Atom(0)
        m = re.match(r"^[A-Z](\d+)$", sym)
        if m:
            a.SetAtomMapNum(int(m.group(1)))
        a.SetProp("dummyLabel", sym)
        return a
    a = Chem.Atom(sym)
    a.SetFormalCharge(q)
    return a


def _apply_stereo(mol, idx, pos, chiral):
    """pos is in the RDKit frame (y up)."""
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
    """pos (optional) must be in the RDKit frame (y up); use rd_xy()."""
    rw = Chem.RWMol()
    idx = {}
    for i in sorted(elems):
        idx[i] = rw.AddAtom(_make_atom(elems[i]))
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


def layout_mol(m):
    """ACS layout of an RDKit molecule: regular rings, 120 deg chains, bond length 14.4 pt.
    Returns an (n, 2) array in model coordinates (y down)."""
    mm = Chem.Mol(m)
    mm.RemoveAllConformers()
    done = False
    try:
        from rdkit.Chem import rdCoordGen
        rdCoordGen.AddCoords(mm)
        done = mm.GetNumConformers() > 0
    except Exception:
        done = False
    if not done:
        rdDepictor.Compute2DCoords(mm)
    xy = np.array(mm.GetConformer().GetPositions())[:, :2].astype(float)
    ls = [float(np.hypot(*(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()]))) for b in mm.GetBonds()]
    L = float(np.mean(ls)) if ls else 1.5
    xy = xy * (BOND_LEN / max(L, 1e-6))
    xy[:, 1] *= -1.0
    return xy


def best_alignment(new, old):
    """Rotate (and mirror if that fits better) `new` onto `old` (both n x 2), keep the centroid of `old`."""
    cn, co = new.mean(axis=0), old.mean(axis=0)
    P, Q = new - cn, old - co
    best = None
    for mirror in (False, True):
        PP = P * np.array([1.0, -1.0]) if mirror else P
        ang = math.atan2((PP[:, 0] * Q[:, 1] - PP[:, 1] * Q[:, 0]).sum(), (PP * Q).sum())
        c, s = math.cos(ang), math.sin(ang)
        out = PP @ np.array([[c, -s], [s, c]]).T
        err = float(((out - Q) ** 2).sum())
        if best is None or err < best[0] - 1e-9:
            best = (err, out)
    return best[1] + co


_RING_CACHE, _GROUP_CACHE, _AMINO_CACHE = {}, {}, {}


def ring_template(name):
    if name in _RING_CACHE:
        return _RING_CACHE[name]
    m = Chem.MolFromSmiles(RING_SMILES[name])
    Chem.Kekulize(m, clearAromaticFlags=True)
    xy = layout_mol(m)
    xy = xy - xy.mean(axis=0)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(round(b.GetBondTypeAsDouble())))
             for b in m.GetBonds()]
    ri = m.GetRingInfo()
    if len(ri.AtomRings()) == 1:
        a0 = syms.index("C") if "C" in syms else 0
        ang = -math.pi / 2 - math.atan2(xy[a0][1], xy[a0][0])
    else:
        dmax, pair = -1.0, (0, 1)
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                d = float(np.hypot(*(xy[i] - xy[j])))
                if d > dmax + 1e-6:
                    dmax, pair = d, (i, j)
        v = xy[pair[1]] - xy[pair[0]]
        ang = -math.atan2(v[1], v[0])
    c, s = math.cos(ang), math.sin(ang)
    xy = xy @ np.array([[c, -s], [s, c]]).T
    a0 = syms.index("C") if "C" in syms else 0
    cv = -xy[a0]
    t = dict(syms=syms, bonds=bonds, xy=xy, a0=a0, din=cv / np.linalg.norm(cv), order=1)
    _RING_CACHE[name] = t
    return t


def group_template(name):
    if name in _GROUP_CACHE:
        return _GROUP_CACHE[name]
    m = Chem.MolFromSmiles(GROUP_SMILES[name])
    Chem.Kekulize(m, clearAromaticFlags=True)
    xy_all = layout_mol(m)
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


def _long_axis_horizontal(xy):
    if len(xy) < 2:
        return xy
    dmax, pair = -1.0, (0, 1)
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            d = float(np.hypot(*(xy[i] - xy[j])))
            if d > dmax + 1e-6:
                dmax, pair = d, (i, j)
    v = xy[pair[1]] - xy[pair[0]]
    ang = -math.atan2(v[1], v[0])
    c, s = math.cos(ang), math.sin(ang)
    return xy @ np.array([[c, -s], [s, c]]).T


def mol_to_template(m):
    """RDKit mol (no Hs) -> dict(syms, bonds, xy, chiral) for free placement."""
    m = Chem.RemoveHs(Chem.Mol(m))
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    cip = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    Chem.Kekulize(m, clearAromaticFlags=True)
    xy = layout_mol(m)
    syms = []
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            mp = a.GetAtomMapNum()
            syms.append(("R%d" % mp) if mp else "R")
        else:
            syms.append(el_str(a.GetSymbol(), a.GetFormalCharge()))
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), min(3, max(1, int(round(b.GetBondTypeAsDouble())))))
             for b in m.GetBonds()]
    xy = _long_axis_horizontal(xy)
    return dict(syms=syms, bonds=bonds, xy=xy - (xy.min(axis=0) + xy.max(axis=0)) / 2,
                chiral={i: c for i, c in cip.items() if c in ("R", "S")})


def amino_template(name):
    if name not in _AMINO_CACHE:
        _AMINO_CACHE[name] = mol_to_template(Chem.MolFromSmiles(AMINO_SMILES[name]))
    return _AMINO_CACHE[name]


def acs_cleanup(elems, bonds, pos, chiral):
    """ACS clean-up of ONE connected component. pos: id -> (x, y) model. Returns id -> (x, y) or None."""
    ids = sorted(elems)
    rdpos = {i: rd_xy(*pos[i]) for i in ids}
    mol, err, idx = build_mol(elems, bonds, rdpos, chiral)
    if mol is None:
        return None
    ez_before = {}
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() in (Chem.BondStereo.STEREOE,
                                                                          Chem.BondStereo.STEREOZ):
            ez_before[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = b.GetStereo()
    m = Chem.Mol(mol)
    if len(ids) == 1:
        return {ids[0]: pos[ids[0]]}
    try:
        xy = layout_mol(m)
    except Exception:
        return None
    inv = {v: k for k, v in idx.items()}
    old = np.array([pos[inv[k]] for k in range(len(ids))], float)
    new = best_alignment(xy, old)
    out = {inv[k]: (float(new[k][0]), float(new[k][1])) for k in range(len(ids))}
    chk, _, idx2 = build_mol(elems, bonds, {i: rd_xy(*out[i]) for i in ids}, chiral)
    if chk is not None and ez_before:
        for b in chk.GetBonds():
            key = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            if key in ez_before and b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ) \
                    and b.GetStereo() != ez_before[key]:
                out = _flip_double_bond(out, bonds, inv[key[0]], inv[key[1]])
    return out


def _rot_align(new, old):
    """Rotate/translate `new` onto `old` (no mirroring) so that relaxing never turns a structure."""
    cn, co = new.mean(axis=0), old.mean(axis=0)
    P, Q = new - cn, old - co
    ang = math.atan2((P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]).sum(), (P * Q).sum())
    cs, sn = math.cos(ang), math.sin(ang)
    return P @ np.array([[cs, -sn], [sn, cs]]).T + co


def relax_2d(elems, bonds, pos, iters=400):
    """Gentle 2D energy minimisation of ONE structure, starting from the drawn coordinates.
    Springs pull every bond to the ACS length (14.4 pt) and every angle to its ideal value
    (regular ring angles, 120 deg elsewhere, 180 deg for triple bonds / allenes) and a soft repulsion keeps
    non-bonded atoms apart. The result is rotated back onto the drawing, so nothing flips or spins."""
    ids = sorted(elems)
    n = len(ids)
    if n < 2 or not bonds:
        return None
    mol, err, idx = build_mol(elems, bonds)
    if mol is None:
        return None
    k_of = {i: k for k, i in enumerate(ids)}
    X0 = np.array([pos[i] for i in ids], float)
    X = X0 / BOND_LEN
    adj = {k: [] for k in range(n)}
    order = {}
    for (a, b), o in bonds.items():
        ka, kb = k_of[a], k_of[b]
        adj[ka].append(kb)
        adj[kb].append(ka)
        order[(ka, kb)] = order[(kb, ka)] = o
    rings = [list(r) for r in mol.GetRingInfo().AtomRings()]

    def ring_angle(i, j, k):
        best = None
        for r in rings:
            if i in r and j in r and k in r and (best is None or len(r) < len(best)):
                best = r
        return None if best is None else math.pi * (len(best) - 2) / len(best)

    A_i, A_j, A_k, A_t = [], [], [], []
    for i in range(n):
        nb = adj[i]
        if len(nb) < 2 or len(nb) > 3:
            continue
        pairs = [(nb[p], nb[q]) for p in range(len(nb)) for q in range(p + 1, len(nb))]
        tgt = {pq: ring_angle(i, *pq) for pq in pairs}
        if len(nb) == 2:
            if tgt[pairs[0]] is None:
                lin = any(order[(i, j)] == 3 for j in nb) or all(order[(i, j)] == 2 for j in nb)
                tgt[pairs[0]] = math.pi if lin else 2 * math.pi / 3
        else:
            known = [t for t in tgt.values() if t is not None]
            free = [pq for pq in pairs if tgt[pq] is None]
            if free:
                rest = (2 * math.pi - sum(known)) / len(free) if known else 2 * math.pi / 3
                for pq in free:
                    tgt[pq] = min(max(rest, math.radians(80)), math.radians(170))
        for (j, k), t in tgt.items():
            A_i.append(i); A_j.append(j); A_k.append(k); A_t.append(t)
    A_i, A_j, A_k, A_t = (np.array(v) for v in (A_i, A_j, A_k, A_t))
    B = np.array([(k_of[a], k_of[b]) for (a, b) in bonds])
    near = {(i, j) for i in range(n) for j in adj[i]}
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k != i:
                    near.add((i, k))
    R = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in near]).reshape(-1, 2)
    KB, KA, KR, DMIN = 10.0, 3.0, 2.0, 1.25

    def energy(X):
        E, G = 0.0, np.zeros_like(X)
        d = X[B[:, 0]] - X[B[:, 1]]
        L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        E += 0.5 * KB * ((L - 1.0) ** 2).sum()
        g = (KB * (L - 1.0) / L)[:, None] * d
        np.add.at(G, B[:, 0], g)
        np.add.at(G, B[:, 1], -g)
        if len(A_i):
            u, v = X[A_j] - X[A_i], X[A_k] - X[A_i]
            cr = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            phi = np.arctan2(cr, (u * v).sum(axis=1))
            s = np.where(phi >= 0, 1.0, -1.0)
            diff = np.abs(phi) - A_t
            E += 0.5 * KA * (diff ** 2).sum()
            f = (KA * diff * s)[:, None]
            uu = np.maximum((u ** 2).sum(axis=1), 1e-9)[:, None]
            vv = np.maximum((v ** 2).sum(axis=1), 1e-9)[:, None]
            gu = f * np.stack([u[:, 1], -u[:, 0]], axis=1) / uu
            gv = f * np.stack([-v[:, 1], v[:, 0]], axis=1) / vv
            np.add.at(G, A_j, gu)
            np.add.at(G, A_k, gv)
            np.add.at(G, A_i, -(gu + gv))
        if len(R):
            d = X[R[:, 0]] - X[R[:, 1]]
            L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
            m = L < DMIN
            if m.any():
                E += 0.5 * KR * ((DMIN - L[m]) ** 2).sum()
                g = (-KR * (DMIN - L[m]) / L[m])[:, None] * d[m]
                np.add.at(G, R[m, 0], g)
                np.add.at(G, R[m, 1], -g)
        return E, G

    E, G = energy(X)
    step = 0.05
    for _ in range(iters):
        if np.abs(G).max() < 1e-4:
            break
        Xn = X - step * G
        En, Gn = energy(Xn)
        if En < E:
            X, E, G = Xn, En, Gn
            step = min(step * 1.2, 0.2)
        else:
            step *= 0.5
            if step < 1e-6:
                break
    out = _rot_align(X * BOND_LEN, X0)
    if not np.all(np.isfinite(out)):
        return None
    return {i: (float(out[k][0]), float(out[k][1])) for i, k in k_of.items()}


def _flip_double_bond(out, bonds, a, b):
    adj = {}
    for (u, v) in bonds:
        if (u, v) in ((a, b), (b, a)):
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
        return out
    pa, pb = np.array(out[a]), np.array(out[b])
    d = (pb - pa) / np.linalg.norm(pb - pa)
    res = dict(out)
    for i in side:
        v = np.array(out[i]) - pa
        q = pa + 2 * np.dot(v, d) * d - v
        res[i] = (float(q[0]), float(q[1]))
    return res


def embed_3d(elems, bonds, pos, chiral=None):
    """pos in the RDKit frame. Returns (mol with Hs and one 3D conformer, idx) or None."""
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
            c.SetAtomPosition(idx[i], (float(x) / 9.6, float(y) / 9.6, float(rng.normal(0, 0.15))))
        m2 = Chem.Mol(mol)
        m2.RemoveAllConformers()
        m2.AddConformer(c, assignId=True)
        mh = Chem.AddHs(m2, addCoords=True)
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, maxIters=5000)
        elif AllChem.UFFHasAllMoleculeParams(mh):
            AllChem.UFFOptimizeMolecule(mh, maxIters=5000)
    except Exception:
        pass
    return mh, idx


class Atom:
    __slots__ = ("id", "el", "x", "y", "color", "orient", "g3", "z", "xy2")

    def __init__(self, aid, el, x, y, color=None, orient=None):
        self.id, self.el, self.x, self.y, self.color, self.orient = aid, el, float(x), float(y), color, orient
        self.g3 = None
        self.z = 0.0
        self.xy2 = None

    @property
    def pos(self):
        return np.array([self.x, self.y], float)

    def set_pos(self, p):
        self.x, self.y = float(p[0]), float(p[1])


ANNOT_NAMES = {"arrow": "reaction arrow", "eq": "equilibrium arrow", "dbl": "resonance arrow",
               "retro": "retrosynthetic arrow", "dashed": "dashed arrow",
               "curve": "curved arrow (2 e-)", "curve1": "curved half-arrow (1 e-)",
               "plus": "plus sign", "text": "text label"}
ARROW_KINDS = ("arrow", "eq", "dbl", "retro", "dashed", "curve", "curve1")
STRAIGHT_KINDS = ("arrow", "eq", "dbl", "retro", "dashed")


class Annot:
    def __init__(self, aid, kind, pts, text="", scale=1.0, color=None, anchor="c", chem=True, flip=False):
        self.id, self.kind, self.text, self.scale = aid, kind, text, float(scale)
        self.pts = [np.array(p, float) for p in pts]
        self.color, self.anchor, self.chem, self.flip = color, anchor, bool(chem), bool(flip)

    def describe(self):
        return f'text label "{self.text.splitlines()[0] if self.text else ""}"' if self.kind == "text" \
            else ANNOT_NAMES[self.kind]

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "pts": [[float(p[0]), float(p[1])] for p in self.pts],
                "text": self.text, "scale": self.scale, "color": self.color, "anchor": self.anchor,
                "chem": self.chem, "flip": self.flip}

    @staticmethod
    def from_dict(d):
        return Annot(int(d["id"]), d["kind"], d["pts"], d.get("text", ""), d.get("scale", 1.0),
                     d.get("color"), d.get("anchor", "c"), d.get("chem", True), d.get("flip", False))

    def translate(self, d):
        self.pts = [p + d for p in self.pts]


def default_arrow_pts(kind, c, length=ARROW_LEN):
    c = np.asarray(c, float)
    if kind.startswith("curve"):
        p0, p1 = c - np.array([length * 0.2, 0.0]), c + np.array([length * 0.2, 0.0])
        return [p0, c + np.array([0.0, -length * 0.2]), p1]
    return [c - np.array([length / 2, 0.0]), c + np.array([length / 2, 0.0])]


def curve_pts_from_drag(p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    n = np.array([d[1], -d[0]]) / max(L, 1e-9)
    return [p0, (p0 + p1) / 2 + n * L * 0.35, p1]


class TextLabelDialog(QtWidgets.QDialog):
    SYMBOLS = ["°", "°C", "Δ", "±", "→", "⇌", "×", "µ", "Å", "‡"]

    def __init__(self, text="", title="Text label", chem=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(""))
        self.edit = QtWidgets.QPlainTextEdit(text)
        self.edit.setMinimumSize(360, 90)
        lay.addWidget(self.edit)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Insert:"))
        for sym in self.SYMBOLS:
            b = QtWidgets.QPushButton(sym)
            b.setFocusPolicy(Qt.NoFocus)
            b.setFixedWidth(38)
            b.clicked.connect(lambda checked=False, t=sym: self._insert(t))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        for label, pre in (("subscript  x₂", "_{"), ("superscript  x²", "^{")):
            b = QtWidgets.QPushButton(label)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda checked=False, p=pre: self._wrap(p))
            row2.addWidget(b)
        row2.addStretch(1)
        lay.addLayout(row2)
        self.chem_box = QtWidgets.QCheckBox("Chemical formatting (H2O → H₂O, Na+ → Na⁺)")
        self.chem_box.setChecked(bool(chem))
        lay.addWidget(self.chem_box)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.edit.setFocus()
        self.edit.selectAll()

    def _insert(self, t):
        self.edit.insertPlainText(t)
        self.edit.setFocus()

    def _wrap(self, pre):
        cur = self.edit.textCursor()
        sel = cur.selectedText()
        cur.insertText(pre + sel + "}")
        if not sel:
            cur.movePosition(QtGui.QTextCursor.Left)
            self.edit.setTextCursor(cur)
        self.edit.setFocus()

    def value(self):
        return self.edit.toPlainText().strip("\n")

    def chem(self):
        return self.chem_box.isChecked()


class ChargeDialog(QtWidgets.QDialog):
    def __init__(self, element, charge, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Formal charge of {element}")
        lay = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size:16px;font-weight:600;")
        lay.addWidget(self.label)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
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
            t.setAlignment(Qt.AlignCenter)
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
                btn.setFocusPolicy(Qt.NoFocus)
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

    STEREO_MODES = ("No stereo marks", "Stereo labels (R) (S) (E) (Z)", "Wedge / dash bonds at R/S centres")

    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.widget = widget
        self.element, self.ring, self.group, self.amino, self.arrow_tool = "C", None, None, None, None
        self.bond_order = 1
        self.editing = True
        self.dark = False
        self.atoms, self.bonds, self.chiral, self.annots = {}, {}, {}, {}
        self.next_id, self.annot_next = 0, 0
        self.nh, self.lp, self.rings = {}, {}, []
        self.centers, self.ez, self.wedge_info = {}, {}, {}
        self.mol, self.smiles, self._mol_idx = None, "", {}
        self.history = []
        self.mono_letters = True
        self.show_lp = False
        self.circled = False
        self.stereo_mode = 1
        self.bond_scale = 1.0
        self.text_bold = False
        self.zoom, self.ox, self.oy = 3.4, 60.0, 200.0
        self.selected, self.sel_annots = set(), set()
        self.hover_id, self.hover_bond = None, None
        self.preview, self.rband, self.temp_annot = None, None, None
        self.mode = None
        self.press_pos, self.moved = (0, 0), False
        self.pan_last = (0, 0)
        self.drag_src = None
        self.click_bond = None
        self.click_annot = None
        self.right_atom, self.right_bond = None, None
        self.scaling, self.rotating = None, None
        self.last_xy = None
        self.press_shift = False
        self.three, self.next_tok, self.rot3d = {}, 0, None
        self.auto_clean = True

    def w2s(self, p):
        return self.ox + self.zoom * p[0], self.oy + self.zoom * p[1]

    def s2w(self, x, y):
        return np.array([(x - self.ox) / self.zoom, (y - self.oy) / self.zoom], float)

    def render(self):
        self.widget.update()

    def say(self, msg):
        self.statusMessage.emit(short(msg))

    notify = say

    def set_dark(self, dark):
        self.dark = bool(dark)
        self.render()

    def set_editing(self, on):
        self.editing = bool(on)
        self.mode = None
        self.preview = self.rband = self.temp_annot = None
        if not on:
            self.hover_id = self.hover_bond = None
            self.selected, self.sel_annots = set(), set()
        self.render()

    def _editable(self):
        if not self.editing:
            self.say("Editing is switched off.")
            return False
        return True

    def content_bounds(self, atom_ids=None, annot_ids=None):
        r = SceneRenderer(self, atom_ids=atom_ids, annot_ids=annot_ids)
        return r.bounds()

    def fit_view(self, w=None, h=None, margin=40.0):
        w = w or self.widget.width()
        h = h or self.widget.height()
        b = self.content_bounds()
        if b is None:
            self.zoom, self.ox, self.oy = 3.4, w * 0.25, h * 0.5
        else:
            bw, bh = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
            z = min((w - 2 * margin) / bw, (h - 2 * margin) / bh)
            self.zoom = float(np.clip(z, 0.5, 5.0))
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            self.ox, self.oy = w / 2 - self.zoom * cx, h / 2 - self.zoom * cy
        self.render()

    def zoom_at(self, x, y, f):
        z = float(np.clip(self.zoom * f, 0.4, 14.0))
        f = z / self.zoom
        self.ox = x - (x - self.ox) * f
        self.oy = y - (y - self.oy) * f
        self.zoom = z
        self.render()

    def pan_to(self, x, y):
        self.ox += x - self.pan_last[0]
        self.oy += y - self.pan_last[1]
        self.pan_last = (x, y)
        self.render()

    def paint(self, p, w, h):
        bg = QtGui.QColor(0, 0, 0) if self.dark else QtGui.QColor(255, 255, 255)
        p.fillRect(0, 0, w, h, bg)
        p.save()
        p.translate(self.ox, self.oy)
        p.scale(self.zoom, self.zoom)
        SceneRenderer(self, ui=self.editing, dark=self.dark).draw(p)
        if self.temp_annot is not None:
            r = SceneRenderer(self, ui=False, dark=self.dark, atom_ids=[], annot_ids=[])
            r._draw_annot(p, self.temp_annot)
        p.restore()

    def _clear_tools(self):
        self.ring = self.group = self.amino = self.arrow_tool = None

    def set_element(self, el):
        self._clear_tools()
        self.element = el
        self.toolChanged.emit()

    def set_ring(self, name):
        self._clear_tools()
        self.ring = name
        self.toolChanged.emit()

    def set_group(self, name):
        self._clear_tools()
        self.group = name
        self.toolChanged.emit()

    def set_amino(self, name):
        self._clear_tools()
        self.amino = name
        self.toolChanged.emit()

    def set_arrow_tool(self, kind):
        self._clear_tools()
        self.arrow_tool = kind
        self.toolChanged.emit()

    def set_bond(self, order):
        self.bond_order = order
        self.toolChanged.emit()

    def tool_name(self):
        if self.arrow_tool:
            return ANNOT_NAMES[self.arrow_tool] + " (drag on the page)"
        if self.amino:
            return self.amino
        if self.group:
            return self.group
        if self.ring:
            return self.ring
        return f"{self.element}  •  {BOND_NAMES[self.bond_order].lower()}"

    def _elems(self, ids=None):
        return {i: a.el for i, a in self.atoms.items() if ids is None or i in ids}

    def rd_inputs(self, ids=None):
        ids = set(self.atoms) if ids is None else set(ids) & set(self.atoms)
        elems = {i: self.atoms[i].el for i in ids}
        bonds = {k: o for k, o in self.bonds.items() if k[0] in ids and k[1] in ids}
        pos = {i: rd_xy(*(self.atoms[i].xy2 or (self.atoms[i].x, self.atoms[i].y))) for i in ids}
        chiral = {i: v for i, v in self.chiral.items() if i in ids}
        return elems, bonds, pos, chiral

    def _check(self, elems, bonds):
        mol, err, _ = build_mol(elems, bonds)
        if mol is None:
            self.say("Rejected: " + err)
            return False
        return True

    def _derive(self):
        elems, bonds, pos, chiral = self.rd_inputs()
        mol, err, idx = build_mol(elems, bonds, pos, chiral)
        self.mol, self._mol_idx = mol, idx
        self.nh, self.lp, self.rings = {}, {}, []
        self.centers, self.ez, self.wedge_info = {}, {}, {}
        smi = ""
        if mol is not None and mol.GetNumAtoms():
            id_of = {v: k for k, v in idx.items()}
            for a in mol.GetAtoms():
                i = id_of[a.GetIdx()]
                if a.GetAtomicNum() == 0:
                    self.nh[i], self.lp[i] = 0, 0
                    continue
                self.nh[i] = a.GetTotalNumHs()
                if a.GetAtomicNum() in LONE_PAIR_Z:
                    e = _PT.GetNOuterElecs(a.GetAtomicNum()) - a.GetFormalCharge() - int(round(a.GetTotalValence()))
                    self.lp[i] = max(0, e // 2)
                else:
                    self.lp[i] = 0
            self.rings = [[id_of[k] for k in ring] for ring in mol.GetRingInfo().AtomRings()]
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                smi = Chem.MolToSmiles(mol)
            try:
                for i, c in Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True):
                    self.centers[id_of[i]] = c
            except Exception:
                pass
            for b in mol.GetBonds():
                if b.GetBondType() != Chem.BondType.DOUBLE:
                    continue
                st = b.GetStereo()
                code = "E" if st == Chem.BondStereo.STEREOE else "Z" if st == Chem.BondStereo.STEREOZ else None
                if code:
                    self.ez[bkey(id_of[b.GetBeginAtomIdx()], id_of[b.GetEndAtomIdx()])] = code
            self.wedge_info = self._compute_wedges(mol, id_of)
        self.smiles = smi

    def _compute_wedges(self, mol, id_of):
        if self.stereo_mode != 2 or not mol.GetNumConformers():
            return {}
        try:
            m = Chem.Mol(mol)
            Chem.WedgeMolBonds(m, m.GetConformer())
        except Exception:
            return {}
        out = {}
        for b in m.GetBonds():
            d = b.GetBondDir()
            if d in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
                a0, a1 = id_of[b.GetBeginAtomIdx()], id_of[b.GetEndAtomIdx()]
                out[bkey(a0, a1)] = ("wedge" if d == Chem.BondDir.BEGINWEDGE else "dash", a0)
        return out

    def _sync(self, push=True):
        self._derive()
        if push:
            self._push_history()
        self.smilesChanged.emit(self.smiles)
        self.render()

    def _commit(self, fit=False, clean=True):
        """Every drawing change: minimise the layout of each structure, restore ACS format, then sync."""
        if clean and self.auto_clean and self.editing and self.atoms:
            self._relax(self.components())
        self._sync()

    def _relax(self, comps):
        """Minimise while drawing: gentle 2D relaxation of each structure, same place, same orientation."""
        for comp in comps:
            if self._is3d(comp):
                continue
            elems, bonds, _, _ = self.rd_inputs(comp)
            pos = {i: (self.atoms[i].x, self.atoms[i].y) for i in comp}
            try:
                out = relax_2d(elems, bonds, pos)
            except Exception:
                out = None
            if out:
                for i, (x, y) in out.items():
                    self.atoms[i].set_pos((x, y))

    def _relayout(self, comps):
        """Energy-minimised (CoordGen) 2D layout -> ACS 1996 geometry for each connected structure."""
        done = 0
        for comp in comps:
            if self._is3d(comp):
                continue
            elems, bonds, _, chiral = self.rd_inputs(comp)
            pos = {i: (self.atoms[i].x, self.atoms[i].y) for i in comp}
            try:
                out = acs_cleanup(elems, bonds, pos, chiral)
            except Exception:
                out = None
            if out is None:
                continue
            for i, (x, y) in out.items():
                self.atoms[i].set_pos((x, y))
            done += 1
        return done

    def state(self):
        return {
            "atoms": [{"id": i, "element": a.el, "x": (a.xy2 or (a.x, a.y))[0], "y": (a.xy2 or (a.x, a.y))[1],
                       "color": a.color, "orient": a.orient}
                      for i, a in sorted(self.atoms.items())],
            "bonds": [[a, b, o] for (a, b), o in sorted(self.bonds.items())],
            "chiral": {str(i): c for i, c in self.chiral.items()},
            "annots": [r.to_dict() for _, r in sorted(self.annots.items())],
            "next_id": self.next_id, "annot_next": self.annot_next,
        }

    def _load_state(self, d):
        self.three = {}
        self.atoms = {int(a["id"]): Atom(int(a["id"]), a["element"], a["x"], a["y"], a.get("color"),
                                         a.get("orient")) for a in d.get("atoms", [])}
        self.bonds = {bkey(int(a), int(b)): int(o) for a, b, o in d.get("bonds", [])}
        self.chiral = {int(i): c for i, c in d.get("chiral", {}).items()}
        self.annots = {}
        for n in d.get("annots", []):
            r = Annot.from_dict(n)
            self.annots[r.id] = r
        self.next_id = int(d.get("next_id", max(list(self.atoms) + [-1]) + 1))
        self.annot_next = int(d.get("annot_next", max(list(self.annots) + [-1]) + 1))
        self.selected &= set(self.atoms)
        self.sel_annots &= set(self.annots)

    def _push_history(self):
        self.history.append(json.loads(json.dumps(self.state())))
        del self.history[:-100]

    def undo(self):
        if not self.editing:
            return
        if not self.history:
            self.say("Nothing to undo.")
            return
        self.history.pop()
        prev = self.history[-1] if self.history else {}
        self._load_state(prev)
        self.hover_id = self.hover_bond = None
        self._sync(push=False)
        self.say("Undone.")

    def clear(self):
        self.three = {}
        self.atoms, self.bonds, self.chiral, self.annots = {}, {}, {}, {}
        self.next_id = self.annot_next = 0
        self.selected, self.sel_annots, self.history = set(), set(), []
        self.hover_id = self.hover_bond = None
        self.preview = self.rband = self.temp_annot = None
        self.mode = None
        self._derive()
        self.render()

    def scene_dict(self):
        d = self.state()
        d.update({
            "format": "pad", "version": 2, "units": "pt",
            "bond_scale": self.bond_scale, "text_bold": self.text_bold, "dark": self.dark,
            "mono_letters": self.mono_letters, "stereo_mode": self.stereo_mode,
            "show_lone_pairs": self.show_lp, "circled_charges": self.circled,
            "smiles": self.smiles,
            "view": {"zoom": self.zoom, "ox": self.ox, "oy": self.oy},
        })
        return d

    def load_scene_dict(self, d):
        if d.get("format") not in ("pad", "ca"):
            raise ValueError("not a .pad scene file")
        self.clear()
        self.dark = bool(d.get("dark", self.dark))
        old = int(d.get("version", 1)) < 2
        if old:
            self.bond_scale = 1.0
            self.text_bold = False
            self.stereo_mode = int(d.get("stereo_mode", 1))
            self.mono_letters = bool(d.get("mono_letters", True))
            for a in d.get("atoms", []):
                i = int(a["id"])
                self.atoms[i] = Atom(i, a["element"], a["x"] * OLD_UNIT, -a["y"] * OLD_UNIT)
            self.next_id = int(d.get("next_id", max(list(self.atoms) + [-1]) + 1))
            self.bonds = {bkey(int(a), int(b)): int(o) for a, b, o in d.get("bonds", [])}
            self.chiral = {int(i): c for i, c in d.get("chiral", {}).items()}
            for n in d.get("annots", []):
                kind = n["kind"]
                c = np.array([n["x"] * OLD_UNIT, -n["y"] * OLD_UNIT])
                if kind in ("text", "plus"):
                    r = Annot(int(n["id"]), kind, [c], n.get("text", ""), 1.0)
                else:
                    ang = -float(n.get("angle", 0.0))
                    L = ARROW_LEN * 0.4 * float(n.get("scale", 1.0)) * 2.4
                    u = np.array([math.cos(ang), math.sin(ang)])
                    r = Annot(int(n["id"]), kind if kind in ARROW_KINDS else "arrow", [c - u * L / 2, c + u * L / 2])
                self.annots[r.id] = r
            self.annot_next = int(d.get("annot_next", max(list(self.annots) + [-1]) + 1))
        else:
            self.bond_scale = float(d.get("bond_scale", 1.0))
            self.text_bold = bool(d.get("text_bold", False))
            self.stereo_mode = int(d.get("stereo_mode", 1))
            self.mono_letters = bool(d.get("mono_letters", True))
            self.show_lp = bool(d.get("show_lone_pairs", False))
            self.circled = bool(d.get("circled_charges", False))
            self._load_state(d)
        self._commit(clean=False)
        v = d.get("view")
        if v:
            self.zoom, self.ox, self.oy = float(v["zoom"]), float(v["ox"]), float(v["oy"])
        self.render()
        return bool(v) and not old

    def _pick_r(self):
        return max(9.0, 4.6 * self.zoom)

    def pick_atom(self, x, y):
        best, bd = None, self._pick_r()
        for i, a in self.atoms.items():
            sx, sy = self.w2s((a.x, a.y))
            d = math.hypot(sx - x, sy - y)
            if d <= bd:
                best, bd = i, d
        return best

    def _seg_dist(self, p, a, b):
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
        return float(np.linalg.norm(p - (a + t * ab)))

    def pick_bond(self, x, y):
        best, bd = None, 7.0
        p = np.array([x, y], float)
        for k in self.bonds:
            a = np.array(self.w2s(self.atoms[k[0]].pos))
            b = np.array(self.w2s(self.atoms[k[1]].pos))
            d = self._seg_dist(p, a, b)
            if d < bd:
                best, bd = k, d
        return best

    def pick_annot(self, x, y):
        best, bd = None, 1e9
        p = np.array([x, y], float)
        w = self.s2w(x, y)
        for i, r in self.annots.items():
            if r.kind in ("text", "plus"):
                bb = annot_bbox(r, self.text_bold)
                pad = 2.0
                if bb[0] - pad <= w[0] <= bb[2] + pad and bb[1] - pad <= w[1] <= bb[3] + pad:
                    d = math.hypot(w[0] - r.pts[0][0], w[1] - r.pts[0][1])
                    if d < bd:
                        best, bd = i, d
            else:
                lines, polys, _ = arrow_geometry(r.kind, r.pts[0], r.pts[-1], r.pts, r.flip)
                for pl in lines + [q for q in polys]:
                    for k in range(len(pl) - 1):
                        d = self._seg_dist(p, np.array(self.w2s(pl[k])), np.array(self.w2s(pl[k + 1])))
                        if d < 8.0 and d < bd:
                            best, bd = i, d
        return best

    def pick_handle(self, x, y):
        if len(self.sel_annots) != 1:
            return None
        rec = self.annots.get(next(iter(self.sel_annots)))
        if rec is None or rec.kind in ("text", "plus"):
            return None
        for k, q in enumerate(rec.pts):
            sx, sy = self.w2s(q)
            if math.hypot(sx - x, sy - y) <= 8.0:
                return rec.id, k
        return None

    def snap_point(self, x, y):
        a = self.pick_atom(x, y)
        if a is not None:
            return self.atoms[a].pos
        k = self.pick_bond(x, y)
        if k is not None:
            return (self.atoms[k[0]].pos + self.atoms[k[1]].pos) / 2
        return self.s2w(x, y)

    def press(self, button, x, y, shift=False):
        self.press_pos, self.moved = (x, y), False
        self.last_xy = (x, y)
        if button == "left":
            if self.arrow_tool and self.editing:
                self.mode = "arrow"
                self.arrow_p0 = self.snap_point(x, y) if self.arrow_tool.startswith("curve") else self.s2w(x, y)
                return
            h = self.pick_handle(x, y)
            if h is not None:
                self.mode, self.handle = "handle", h
                return
            if shift and (self.selected or self.sel_annots):
                self.mode = "move"
                self.move_start = self.s2w(x, y)
                self.move_atoms = {i: self.atoms[i].pos.copy() for i in self.selected if i in self.atoms}
                self.move_annots = {i: [p.copy() for p in self.annots[i].pts] for i in self.sel_annots
                                    if i in self.annots}
                return
            aid = self.pick_atom(x, y)
            if aid is not None and self.atoms[aid].g3 is not None:
                self.mode = "rot3d"
                self.rot3d = (self.atoms[aid].g3, x, y)
                return
            if aid is not None:
                self.mode, self.drag_src = "bond", aid
                self.hover_id = aid
                self.render()
                return
            bk = self.pick_bond(x, y)
            if bk is not None and (self.atoms[bk[0]].g3 is not None or self.atoms[bk[1]].g3 is not None):
                self.mode = "none"
                return
            if bk is not None:
                self.mode, self.click_bond = "bondclick", bk
                return
            an = self.pick_annot(x, y)
            if an is not None:
                self.mode, self.click_annot = "annotclick", an
                self.was_selected = an in self.sel_annots
                if not self.was_selected:
                    self.set_selection([], [an])
                self.move_start = self.s2w(x, y)
                self.move_annots = {i: [p.copy() for p in self.annots[i].pts] for i in self.sel_annots
                                    if i in self.annots}
                return
            self.mode = "empty"
        else:
            self.press_shift = bool(shift)
            self.right_atom = self.pick_atom(x, y)
            self.right_bond = self.pick_bond(x, y) if self.right_atom is None else None
            self.mode = "right"

    def _check_moved(self, x, y):
        px, py = self.press_pos
        if math.hypot(x - px, y - py) > DRAG_THRESHOLD:
            self.moved = True

    def move(self, x, y, left_down=False, right_down=False):
        self.last_xy = (x, y)
        if self.mode == "rot3d":
            tok, lx, ly = self.rot3d
            self.rotate_3d(tok, (x - lx) * 0.012, (y - ly) * 0.012)
            self.rot3d = (tok, x, y)
            return
        if self.mode == "empty":
            self._check_moved(x, y)
            return
        if self.mode == "arrow":
            self._check_moved(x, y)
            if self.moved:
                p1 = self.snap_point(x, y) if self.arrow_tool.startswith("curve") else self._snap_dir(self.arrow_p0, self.s2w(x, y))
                pts = curve_pts_from_drag(self.arrow_p0, p1) if self.arrow_tool.startswith("curve") \
                    else [self.arrow_p0, p1]
                self.temp_annot = Annot(-1, self.arrow_tool, pts)
                self.render()
            return
        if self.mode == "handle":
            self._check_moved(x, y)
            rec = self.annots.get(self.handle[0])
            if rec is not None and self.moved:
                k = self.handle[1]
                end = k in (0, len(rec.pts) - 1)
                if rec.kind in STRAIGHT_KINDS:
                    other = rec.pts[1 - k]
                    rec.pts[k] = self._snap_dir(other, self.s2w(x, y))
                elif end:
                    rec.pts[k] = self.snap_point(x, y)
                else:
                    rec.pts[k] = self.s2w(x, y)
                self.render()
            return
        if self.mode in ("move", "annotclick"):
            self._check_moved(x, y)
            if self.moved and self.mode == "annotclick":
                self.mode = "move"
                self.move_atoms = {}
            if self.mode == "move" and self.moved:
                d = self.s2w(x, y) - self.move_start
                for i, p0 in self.move_atoms.items():
                    self.atoms[i].set_pos(p0 + d)
                for i, pts in self.move_annots.items():
                    self.annots[i].pts = [q + d for q in pts]
                self.render()
            return
        if self.mode in ("right", "rband"):
            self._check_moved(x, y)
            if self.moved:
                self.mode = "rband"
                a, b = self.s2w(*self.press_pos), self.s2w(x, y)
                self.rband = (a[0], a[1], b[0], b[1])
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
            self.render()

    def _snap_dir(self, p0, p1, step=math.pi / 12):
        d = np.asarray(p1, float) - np.asarray(p0, float)
        L = float(np.hypot(*d))
        if L < 1e-6:
            return np.asarray(p1, float)
        a = round(math.atan2(d[1], d[0]) / step) * step
        return np.asarray(p0, float) + L * np.array([math.cos(a), math.sin(a)])

    def release(self, button, x, y, shift=False):
        mode, self.mode = self.mode, None
        if button == "left":
            if mode == "arrow":
                kind = self.arrow_tool
                self.temp_annot = None
                if self.moved:
                    p1 = self.snap_point(x, y) if kind.startswith("curve") else self._snap_dir(self.arrow_p0, self.s2w(x, y))
                    pts = curve_pts_from_drag(self.arrow_p0, p1) if kind.startswith("curve") else [self.arrow_p0, p1]
                    if float(np.hypot(*(pts[-1] - pts[0]))) > 4.0:
                        self._add_arrow(kind, pts)
                else:
                    self._add_arrow(kind, default_arrow_pts(kind, self.s2w(x, y)))
                self.arrow_tool = None
                self.toolChanged.emit()
            elif mode == "handle":
                if self.moved:
                    self._push_history()
                    self.notify("Arrow edited.")
            elif mode == "empty":
                if not self.moved:
                    if self.selected or self.sel_annots:
                        self.clear_selection()
                    else:
                        self.place_first(*self.press_pos)
            elif mode == "move":
                if self.moved:
                    self._commit(clean=False)
                    self.notify("Moved")
            elif mode == "annotclick":
                if not self.moved and self.click_annot in self.annots and self.was_selected:
                    self.set_selection([], [])
                self.click_annot = None
            elif mode == "bondclick":
                if not self.moved and self.click_bond in self.bonds:
                    self.cycle_bond(self.click_bond)
                self.click_bond = None
            elif mode == "bond":
                self.preview = None
                src = self.drag_src
                if src in self.atoms:
                    if not self.moved:
                        self._click_atom(src)
                    else:
                        tgt = self.pick_atom(x, y)
                        if tgt is not None and tgt != src:
                            self.connect(src, tgt, self.bond_order)
                        elif tgt is None:
                            pos, _ = self.ghost_position(src, x, y)
                            self.add_atom_bonded(src, self.element, pos, self.bond_order)
                self.drag_src = None
                self.hover_id = self.pick_atom(x, y)
                self.render()
        else:
            if mode == "rband":
                self.rband = None
                x0, y0 = self.press_pos
                xa, xb = sorted((x0, x))
                ya, yb = sorted((y0, y))
                ids = [i for i, a in self.atoms.items()
                       if xa <= self.w2s(a.pos)[0] <= xb and ya <= self.w2s(a.pos)[1] <= yb]
                ann = []
                for i, rec in self.annots.items():
                    bb = annot_bbox(rec, self.text_bold)
                    c = self.w2s(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))
                    if xa <= c[0] <= xb and ya <= c[1] <= yb:
                        ann.append(i)
                if shift or self.press_shift:
                    ids = set(ids) | self.selected
                    ann = set(ann) | self.sel_annots
                self.set_selection(ids, ann)
                if ids or ann:
                    self.say("Selected")
            elif mode == "right":
                aid, bk, xy = self.right_atom, self.right_bond, self.press_pos
                QtCore.QTimer.singleShot(0, lambda: self.show_menu(aid, bk, xy))

    def _click_atom(self, src):
        if not self._editable():
            return
        deg = sum(1 for k in self.bonds if src in k)
        if self.group:
            self.attach_group(src, self.group)
        elif self.ring:
            self.attach_ring(src, self.ring)
        elif self.amino:
            self.say("Click on empty space to place an amino acid.")
        elif is_generic(self.element) and deg >= 2:
            self.add_substituent(src, self.element)
        elif self.atoms[src].el != self.element:
            self._set_element(src, self.element)

    def _update_preview(self, x, y):
        src = self.drag_src
        if src not in self.atoms or not self.moved:
            return
        sp = self.atoms[src].pos
        tgt = self.pick_atom(x, y)
        if tgt is not None and tgt != src:
            self.preview = (sp, self.atoms[tgt].pos, False, None)
            self.hover_id = tgt
        else:
            self.hover_id = src
            if tgt == src:
                self.preview = None
                self.render()
                return
            end, L = self.ghost_position(src, x, y)
            self.preview = (sp, end, self.clashes(end, src, L), self.element)
        self.render()

    def ghost_position(self, src, x, y):
        rec = self.atoms[src]
        sx, sy = self.w2s(rec.pos)
        step = math.pi / 6
        ang = round(math.atan2(y - sy, x - sx) / step) * step
        return rec.pos + BOND_LEN * np.array([math.cos(ang), math.sin(ang)]), BOND_LEN

    def clashes(self, pos, ignore, L):
        return any(np.linalg.norm(a.pos - pos) < 0.6 * L for i, a in self.atoms.items() if i != ignore)

    def _create_atom(self, el, x, y, color=None):
        aid = self.next_id
        self.next_id += 1
        self.atoms[aid] = Atom(aid, el, x, y, color)
        return aid

    def place_first(self, x, y):
        if not self._editable():
            return
        c = self.s2w(x, y)
        if self.arrow_tool:
            return
        if self.amino:
            self.place_template(amino_template(self.amino), c)
        elif self.ring:
            self.place_template(ring_template(self.ring), c)
        elif self.group:
            self.place_template(group_template(self.group), c)
        else:
            self._create_atom(self.element, c[0], c[1])
            self._commit()

    def place_template(self, t, center, select=False):
        xy = t["xy"] - t["xy"].mean(axis=0)
        ids = [self._create_atom(s, center[0] + p[0], center[1] + p[1]) for s, p in zip(t["syms"], xy)]
        for i, j, o in t["bonds"]:
            self.bonds[bkey(ids[i], ids[j])] = o
        for i, c in (t.get("chiral") or {}).items():
            self.chiral[ids[i]] = c
        if select:
            self.selected = set(ids)
        self._commit()
        return ids

    def place_ring(self, name, center):
        self.place_template(ring_template(name), center)

    def attach_ring(self, aid, name):
        self.attach_template(aid, ring_template(name))

    def attach_group(self, aid, name):
        self.attach_template(aid, group_template(name))

    def _free_direction(self, aid):
        A = self.atoms[aid]
        vecs = []
        for k in self.bonds:
            if aid in k:
                o = self.atoms[k[0] if k[1] == aid else k[1]]
                v = o.pos - A.pos
                n = np.linalg.norm(v)
                if n > 1e-9:
                    vecs.append(v / n)
        if vecs:
            m = -np.sum(vecs, axis=0)
            if np.linalg.norm(m) < 0.3:
                m = np.array([-vecs[0][1], vecs[0][0]])
            return m / np.linalg.norm(m)
        return np.array([math.cos(math.pi / 6), -math.sin(math.pi / 6)])

    def add_substituent(self, aid, el):
        d0 = self._free_direction(aid)
        ang0 = math.atan2(d0[1], d0[0])
        others = np.array([a.pos for a in self.atoms.values()])
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6):
            ang = ang0 + k * math.pi / 6
            p = self.atoms[aid].pos + BOND_LEN * np.array([math.cos(ang), math.sin(ang)])
            if np.min(np.linalg.norm(others - p, axis=1)) >= 0.6 * BOND_LEN:
                self.add_atom_bonded(aid, el, p, 1)
                return
        self.say("No room to attach it there.")

    def attach_template(self, aid, t):
        if not self._editable():
            return
        A = self.atoms[aid]
        d0 = self._free_direction(aid)
        ang0 = math.atan2(d0[1], d0[0])
        a0, xy, order = t["a0"], t["xy"], t.get("order", 1)
        rel = xy - xy[a0]
        cv = t["din"]
        th = math.atan2(cv[1], cv[0])
        others = np.array([a.pos for a in self.atoms.values()])
        pts = None
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6):
            ang = ang0 + k * math.pi / 6
            phi = ang - th
            R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
            cand = A.pos + BOND_LEN * np.array([math.cos(ang), math.sin(ang)]) + rel @ R.T
            if np.min(np.linalg.norm(cand[:, None, :] - others[None, :, :], axis=2)) >= 0.6 * BOND_LEN:
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
            self._create_atom(s, p[0], p[1])
        self.bonds = bonds
        self._commit()

    def add_atom_bonded(self, src, element, pos, order):
        if self.clashes(pos, src, BOND_LEN):
            self.say("Position overlaps another atom - drag in another direction.")
            return
        new_id = self.next_id
        elems = self._elems()
        elems[new_id] = element
        bonds = dict(self.bonds)
        bonds[bkey(src, new_id)] = order
        if not self._check(elems, bonds):
            return
        self._create_atom(element, pos[0], pos[1])
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
        self.atoms[aid].el = element
        self._commit()
        return True

    def set_charge(self, aid, q):
        if not self._editable() or aid not in self.atoms:
            return
        sym = base(self.atoms[aid].el)
        new = el_str(sym, q)
        if new == self.atoms[aid].el:
            return
        if self._set_element(aid, new):
            self.say(f"{sym} formal charge set to {q:+d}." if q else f"{sym} made neutral.")

    def charge_dialog(self, aid):
        if not self._editable() or aid not in self.atoms:
            return
        sym, q = split_el(self.atoms[aid].el)
        dlg = ChargeDialog(sym, q, self.widget)
        if dlg.exec():
            self.set_charge(aid, dlg.value())

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
                self.say("Bond")
                return
        self.say("Bond order can't change: it would exceed an atom's valence.")

    def delete_atoms(self, ids):
        if not self._editable():
            return
        ids = [i for i in ids if i in self.atoms]
        if not ids:
            return
        idset = set(ids)
        self.bonds = {k: o for k, o in self.bonds.items() if k[0] not in idset and k[1] not in idset}
        for i in ids:
            self.atoms.pop(i)
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
        for i in ann:
            self.annots.pop(i, None)
        self.sel_annots = set()
        if ids:
            self.delete_atoms(ids)
        else:
            self._commit()
            self.notify(f"Deleted {len(ann)} object(s).")

    def delete_annots(self, ids):
        if not self._editable():
            return
        ids = [i for i in ids if i in self.annots]
        for i in ids:
            self.annots.pop(i)
            self.sel_annots.discard(i)
        if ids:
            self._push_history()
            self.render()
            self.notify(f"Deleted {len(ids)} object(s).")

    def set_chirality(self, aid, want):
        if not self._editable() or aid not in self.atoms:
            return
        if want is None:
            self.chiral.pop(aid, None)
        else:
            self.chiral[aid] = want
        self._commit()
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
        pos = {i: (r.x, r.y) for i, r in self.atoms.items()}
        res = _flip_double_bond(pos, self.bonds, a, b)
        if res is pos or res == pos:
            self.say("This double bond is in a ring - E/Z cannot be flipped here.")
            return
        for i, (x, y) in res.items():
            self.atoms[i].set_pos((x, y))
        self._commit()
        self.say(f"Double bond set to {self.ez.get(k, want)}.")

    def cycle_stereo_display(self):
        self.stereo_mode = (self.stereo_mode + 1) % 3
        self._derive()
        self.render()
        self.notify(f"Stereo display: {self.STEREO_MODES[self.stereo_mode]}  (Ctrl+H cycles)")

    toggle_stereo_labels = cycle_stereo_display

    def toggle_auto_clean(self):
        self.auto_clean = not self.auto_clean
        self.notify("Auto on" if self.auto_clean else "Auto off")

    def toggle_mono_letters(self):
        self.mono_letters = not self.mono_letters
        self.render()
        self.notify("Atom letters: " + ("black (ACS)" if self.mono_letters else "coloured by element")
                    + "  (Ctrl+G toggles)")

    def toggle_lone_pairs(self):
        self.show_lp = not self.show_lp
        self.render()
        self.notify("Lone pairs " + ("shown" if self.show_lp else "hidden"))

    def toggle_circled_charges(self):
        self.circled = not self.circled
        self.render()
        self.notify("Formal charges drawn " + ("in circles (⊕ ⊖)" if self.circled else "as plain + / −"))

    def set_text_thickness(self, bold):
        self.text_bold = bool(bold)
        self.render()
        self.notify("Text labels: " + ("bold" if self.text_bold else "regular"))

    def set_bond_thickness(self, factor=None, scale_by=None):
        f = self.bond_scale if factor is None else factor
        if scale_by is not None:
            f *= scale_by
        self.bond_scale = float(np.clip(f, 0.3, 4.0))
        self.render()
        self.notify(f"Bond line width x{self.bond_scale:.2f}  (ACS = 0.6 pt at x1.00)")

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

    def selected_components(self):
        return [c for c in self.components() if c & self.selected]

    def selected_compounds(self):
        out = []
        for comp in self.selected_components():
            elems, bonds, pos, chiral = self.rd_inputs(comp)
            mol, err, _ = build_mol(elems, bonds, pos, chiral)
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                smi = Chem.MolToSmiles(mol)
            out.append({"smiles": smi, "ids": set(comp)})
        return out

    def set_selection(self, ids, annots=()):
        self.selected = set(ids) & set(self.atoms)
        self.sel_annots = set(annots) & set(self.annots)
        self.render()

    def clear_selection(self):
        if self.selected or self.sel_annots:
            self.set_selection([])

    def _add_arrow(self, kind, pts):
        if not self._editable():
            return None
        aid = self.annot_next
        self.annot_next += 1
        self.annots[aid] = Annot(aid, kind, pts)
        self.sel_annots = {aid}
        self.selected = set()
        self._push_history()
        self.render()
        return aid

    def add_annot(self, kind, xy, text="", scale=1.0, anchor="c", chem=True):
        if not self._editable():
            return None
        c = self.s2w(*xy)
        if kind in ARROW_KINDS:
            return self._add_arrow(kind, default_arrow_pts(kind, c))
        aid = self.annot_next
        self.annot_next += 1
        self.annots[aid] = Annot(aid, kind, [c], text, scale, None, anchor, chem)
        self._push_history()
        self.render()
        return aid

    def choose_arrow(self, kind, place_xy=None):
        if place_xy is not None:
            self.add_annot(kind, place_xy)
        else:
            self.set_arrow_tool(kind)
            self.say(f"{ANNOT_NAMES[kind].capitalize()}: drag on the page (click = default size). "
                     "Select an arrow to edit its handles.")

    def add_text_label(self, xy, anchor="c", at=None):
        dlg = TextLabelDialog("", "Text label", True, self.widget)
        if dlg.exec() and dlg.value():
            aid = self.annot_next
            self.annot_next += 1
            c = np.array(at, float) if at is not None else self.s2w(*xy)
            self.annots[aid] = Annot(aid, "text", [c], dlg.value(), 1.0, None, anchor, dlg.chem())
            self._push_history()
            self.render()

    def add_arrow_text(self, arrow_id, above=True):
        rec = self.annots.get(arrow_id)
        if rec is None:
            return
        p0, p1 = rec.pts[0], rec.pts[-1]
        mid = (p0 + p1) / 2
        dlg = TextLabelDialog("", "Text above the arrow" if above else "Text below the arrow", True, self.widget)
        if dlg.exec() and dlg.value():
            aid = self.annot_next
            self.annot_next += 1
            pos = mid + np.array([0.0, -4.5 if above else 5.5]) if rec.kind in STRAIGHT_KINDS else mid
            self.annots[aid] = Annot(aid, "text", [pos], dlg.value(), 1.0, None, "b" if above else "t", dlg.chem())
            self._push_history()
            self.render()

    def edit_text_label(self, aid):
        rec = self.annots.get(aid)
        if rec is None or rec.kind != "text" or not self._editable():
            return
        dlg = TextLabelDialog(rec.text, "Edit text label", rec.chem, self.widget)
        if dlg.exec() and dlg.value():
            rec.text, rec.chem = dlg.value(), dlg.chem()
            self._push_history()
            self.render()

    def flip_curve(self, aid):
        rec = self.annots.get(aid)
        if rec is None or not rec.kind.startswith("curve"):
            return
        rec.pts[1] = rec.pts[0] + rec.pts[2] - rec.pts[1]
        rec.flip = not rec.flip
        self._push_history()
        self.render()

    def _sel_centre(self):
        pts = [self.atoms[i].pos for i in self.selected if i in self.atoms]
        for i in self.sel_annots:
            if i in self.annots:
                pts += [np.mean(self.annots[i].pts, axis=0)]
        return np.mean(pts, axis=0) if pts else None

    def begin_scale(self, x, y):
        if self.scaling is not None:
            return True
        if not (self.selected or self.sel_annots) or not self._editable():
            return False
        self.scaling = dict(x0=x, y0=y, f=1.0, centre=self._sel_centre(),
                            atoms={i: self.atoms[i].pos.copy() for i in self.selected if i in self.atoms},
                            annots={i: ([p.copy() for p in self.annots[i].pts], self.annots[i].scale)
                                    for i in self.sel_annots if i in self.annots})
        return True

    def scale_to(self, x, y):
        sc = self.scaling
        if sc is None:
            return
        f = float(np.clip(math.exp(0.004 * ((x - sc["x0"]) - (y - sc["y0"]))), 0.2, 6.0))
        sc["f"] = f
        c = sc["centre"]
        for i, p0 in sc["atoms"].items():
            self.atoms[i].set_pos(c + (p0 - c) * f)
        for i, (pts, s0) in sc["annots"].items():
            r = self.annots[i]
            if r.kind in ("text", "plus"):
                r.pts = [c + (pts[0] - c) * f]
                r.scale = float(np.clip(s0 * f, 0.3, 8.0))
            else:
                r.pts = [c + (q - c) * f for q in pts]
        self.render()

    def end_scale(self):
        sc, self.scaling = self.scaling, None
        if sc is not None and abs(sc["f"] - 1.0) > 1e-3:
            self._commit()
            self.notify("Resized")

    def _bond_fragment(self, bk, x, y):
        """smaller side of a non-ring bond: (pivot atom id, moving atom ids)."""
        a, b = bk
        adj = {}
        for (u, v) in self.bonds:
            if (u, v) == bk:
                continue
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        def side(start):
            seen, st = {start}, [start]
            while st:
                n = st.pop()
                for m in adj.get(n, []):
                    if m not in seen:
                        seen.add(m)
                        st.append(m)
            return seen
        sa, sb = side(a), side(b)
        if a in sb:
            return None
        return (b, sa) if len(sa) <= len(sb) else (a, sb)

    def begin_rotate(self, x, y):
        if self.rotating is not None:
            return True
        if not self._editable():
            return False
        if self.selected or self.sel_annots:
            c = self._sel_centre()
            self.rotating = dict(
                a=0.0, centre=c, a0=math.atan2(y - self.w2s(c)[1], x - self.w2s(c)[0]),
                atoms={i: self.atoms[i].pos.copy() for i in self.selected if i in self.atoms},
                annots={i: [p.copy() for p in self.annots[i].pts] for i in self.sel_annots if i in self.annots})
            return True
        if self.hover_bond is not None and self.hover_bond in self.bonds:
            fr = self._bond_fragment(self.hover_bond, x, y)
            if fr is None:
                self.say("That bond is part of a ring - it cannot be rotated.")
                return False
            pivot, moving = fr
            c = self.atoms[pivot].pos.copy()
            self.rotating = dict(a=0.0, centre=c, a0=math.atan2(y - self.w2s(c)[1], x - self.w2s(c)[0]),
                                 atoms={i: self.atoms[i].pos.copy() for i in moving}, annots={})
            return True
        return False

    def rotate_to(self, x, y, free=False):
        rt = self.rotating
        if rt is None:
            return
        cx, cy = self.w2s(rt["centre"])
        a = math.atan2(y - cy, x - cx) - rt["a0"]
        a = (a + math.pi) % (2 * math.pi) - math.pi
        if not free:
            step = math.pi / 12
            a = round(a / step) * step
        rt["a"] = a
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        ctr = rt["centre"]
        for i, p0 in rt["atoms"].items():
            self.atoms[i].set_pos(ctr + R @ (p0 - ctr))
        for i, pts in rt["annots"].items():
            self.annots[i].pts = [ctr + R @ (q - ctr) for q in pts]
        self.render()

    def end_rotate(self):
        rt, self.rotating = self.rotating, None
        if rt is not None and abs(rt["a"]) > 1e-3:
            self._commit(clean=False)
            self.notify("Rotated")

    def _rotate_ids(self, ids, angle):
        pts = np.array([self.atoms[i].pos for i in ids])
        c = pts.mean(axis=0)
        cs, sn = math.cos(angle), math.sin(angle)
        R = np.array([[cs, -sn], [sn, cs]])
        for i in ids:
            self.atoms[i].set_pos(c + R @ (self.atoms[i].pos - c))

    def toggle_orient(self):
        """Ctrl+Tab: every SELECTED compound is turned horizontal, then vertical (long axis). Needs a selection."""
        comps = self.selected_components()
        if not comps or not self._editable():
            return
        went = None
        for ids in comps:
            if len(ids) < 2:
                continue
            pts = {i: self.atoms[i].pos for i in ids}
            keys = list(pts)
            dmax, pair = -1.0, (keys[0], keys[1])
            for x in range(len(keys)):
                for y in range(x + 1, len(keys)):
                    d = float(np.linalg.norm(pts[keys[x]] - pts[keys[y]]))
                    if d > dmax + 1e-6:
                        dmax, pair = d, (keys[x], keys[y])
            v = pts[pair[1]] - pts[pair[0]]
            th = math.atan2(v[1], v[0]) % math.pi
            if min(th, math.pi - th) < math.radians(3.0):
                self._rotate_ids(ids, math.pi / 2)
                went = "Vertical"
            else:
                self._rotate_ids(ids, -th if th < math.pi / 2 else math.pi - th)
                went = "Horizontal"
        if went:
            self._commit(clean=False)
            self.say(went)

    def color_selected(self):
        if not (self.selected or self.sel_annots):
            self.say("Select atoms first (right-drag a box), then Ctrl+K.")
            return
        if not self._editable():
            return
        first = next(iter(self.selected), None)
        cur = QtGui.QColor(self.atoms[first].color) if first is not None and self.atoms[first].color \
            else QtGui.QColor(220, 30, 30)
        col = QtWidgets.QColorDialog.getColor(cur, self.widget, "Colour of the selected atoms")
        if not col.isValid():
            return
        self.apply_color(col.name())

    def apply_color(self, hexcol):
        for i in self.selected:
            if i in self.atoms:
                self.atoms[i].color = hexcol
        for i in self.sel_annots:
            if i in self.annots:
                self.annots[i].color = hexcol
        self._push_history()
        self.render()
        self.say("Colour reset." if hexcol is None else f"Colour {hexcol} applied.")

    def clean_structure(self):
        """Ctrl+M - ACS clean-up: regular rings, 120 deg chains, 14.4 pt bonds, stereo kept."""
        if not self._editable():
            return
        comps = self.selected_components() if self.selected else self.components()
        if not comps:
            self.say("Nothing to clean up.")
            return
        self._relayout(comps)
        self._commit()
        self.say("Cleaned")

    def minimize_selection(self):
        self.clean_structure()

    def minimize_all(self):
        self.selected = set()
        self.clean_structure()

    def duplicate_selected(self):
        if not (self.selected or self.sel_annots) or not self._editable():
            return
        d = np.array([3.0 * BOND_LEN, 2.0 * BOND_LEN])
        old = sorted(i for i in self.selected if i in self.atoms)
        new = {}
        for i in old:
            r = self.atoms[i]
            new[i] = self._create_atom(r.el, r.x + d[0], r.y + d[1], r.color)
            self.atoms[new[i]].orient = r.orient
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
            r2 = Annot.from_dict(rec.to_dict())
            r2.id = aid
            r2.translate(d)
            self.annots[aid] = r2
            new_ann.append(aid)
        self.selected, self.sel_annots = set(new.values()), set(new_ann)
        self._commit()
        self.notify(f"Duplicated {len(new)} atom(s) and {len(new_ann)} object(s).")

    def _free_spot(self, width, height):
        b = self.content_bounds()
        if b is None:
            c = self.s2w(self.widget.width() / 2, self.widget.height() / 2)
            return c
        return np.array([b[2] + 2.0 * BOND_LEN + width / 2, (b[1] + b[3]) / 2])

    def add_mol(self, mol, center=None, select=True):
        """Add an RDKit molecule to the drawing (never clears what is already there)."""
        t = mol_to_template(mol)
        w = float(t["xy"][:, 0].max() - t["xy"][:, 0].min())
        h = float(t["xy"][:, 1].max() - t["xy"][:, 1].min())
        c = np.asarray(center, float) if center is not None else self._free_spot(w, h)
        return self.place_template(t, c, select=select)

    def paste_text(self, text, at_screen=None):
        text = (text or "").strip()
        if not text:
            return False
        m = None
        if "V2000" in text or "V3000" in text:
            m = Chem.MolFromMolBlock(text)
        if m is None:
            for tok in text.replace(",", " ").split():
                m = Chem.MolFromSmiles(tok)
                if m is not None:
                    break
        if m is None:
            self.say("Clipboard does not contain a valid SMILES / MOL block.")
            return False
        if m.GetNumHeavyAtoms() == 0:
            return False
        c = self.s2w(*at_screen) if at_screen is not None else None
        self.add_mol(m, c, select=True)
        self.say("Pasted structure (existing drawing kept). Ctrl+M cleans up, Ctrl+Z undoes.")
        return True

    def _is3d(self, ids):
        return any(self.atoms[i].g3 is not None for i in ids if i in self.atoms)

    def to_3d(self, comps):
        """Selected compounds -> MMFF94 3D, shown in the same ACS style; drag an atom to rotate."""
        done = 0
        for ids in comps:
            ids = {i for i in ids if i in self.atoms}
            if not ids or self._is3d(ids):
                continue
            elems, bonds, pos, chiral = self.rd_inputs(ids)
            if any(is_generic(e) for e in elems.values()):
                self.say("Invalid")
                continue
            res = embed_3d(elems, bonds, pos, chiral)
            if res is None:
                self.say("Failed")
                continue
            mh, idx = res
            order = sorted(ids)
            P = np.array(mh.GetConformer().GetPositions())[[idx[i] for i in order]] * np.array([1.0, -1.0, -1.0])
            P = (P - P.mean(axis=0)) * (BOND_LEN / 1.5)
            tok = self.next_tok
            self.next_tok += 1
            self.three[tok] = dict(ids=order, P=P, R=np.eye(3))
            for i in order:
                a = self.atoms[i]
                a.xy2, a.g3 = (a.x, a.y), tok
            self._apply_3d(tok)
            done += 1
        if done:
            self.render()
            self.say("3D")

    def toggle_3d(self):
        """Ctrl+3: selected compounds -> 3D; if they are all already 3D -> back to 2D."""
        comps = self.selected_components()
        if not comps:
            self.say("Select")
            return
        if any(not self._is3d(cp) for cp in comps):
            self.to_3d(comps)
        else:
            self.to_2d(comps)

    def to_2d(self, comps):
        """Back to the flat drawing exactly as it was."""
        toks = {self.atoms[i].g3 for ids in comps for i in ids if i in self.atoms and self.atoms[i].g3 is not None}
        for tok in toks:
            for i in self.three.get(tok, {}).get("ids", []):
                a = self.atoms.get(i)
                if a is not None and a.g3 == tok:
                    if a.xy2 is not None:
                        a.x, a.y = a.xy2
                    a.xy2, a.g3, a.z = None, None, 0.0
            self.three.pop(tok, None)
        if toks:
            self.render()
            self.say("2D")

    def _apply_3d(self, tok, centre=None):
        rec = self.three[tok]
        ids = [i for i in rec["ids"] if i in self.atoms]
        if not ids:
            return
        if centre is None:
            centre = np.mean([self.atoms[i].pos for i in ids], axis=0)
        Q = (rec["R"] @ rec["P"].T).T
        for k, i in enumerate(rec["ids"]):
            if i in self.atoms:
                a = self.atoms[i]
                a.x, a.y, a.z = float(centre[0] + Q[k, 0]), float(centre[1] + Q[k, 1]), float(Q[k, 2])

    def rotate_3d(self, tok, dx, dy):
        """Trackball rotation about the compound's centre (dx, dy in radians; screen x right, y down)."""
        rec = self.three.get(tok)
        if rec is None:
            return
        ay, ax = -dx, dy
        cy, sy, cx, sx = math.cos(ay), math.sin(ay), math.cos(ax), math.sin(ax)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rec["R"] = Rx @ Ry @ rec["R"]
        self._apply_3d(tok)
        self.render()

    def _act(self, menu, text, fn, checked=None, enabled=True):
        a = menu.addAction(text)
        if checked is not None:
            a.setCheckable(True)
            a.setChecked(bool(checked))
        a.setEnabled(enabled)
        a.triggered.connect(lambda checked_=False, f=fn: f())
        return a

    def fill_atoms(self, menu, ctx=None):
        ctx = ctx or {}
        place_xy = ctx.get("place_xy")
        grp = QtGui.QActionGroup(menu)
        grp.setExclusive(True)

        def add_el(m, el):
            a = self._act(m, el, lambda e=el: self._choose_element(e, place_xy),
                          checked=(not (self.ring or self.group or self.amino or self.arrow_tool)
                                   and el == self.element))
            grp.addAction(a)

        for el in PRIMARY_ELEMENTS:
            add_el(menu, el)
        more = menu.addMenu("More atoms")
        for el in MORE_ELEMENTS:
            add_el(more, el)
        self._act(menu, "Periodic table...", lambda: self.open_periodic_table(place_xy))

    def fill_generic(self, menu, ctx=None):
        ctx = ctx or {}
        place_xy = ctx.get("place_xy")
        for title, items in GENERIC_MENU:
            if len(items) == 1:
                self._act(menu, items[0], lambda e=items[0]: self._choose_element(e, place_xy),
                          checked=self.element == items[0] and not (self.ring or self.group or self.amino))
            else:
                sub = menu.addMenu(title)
                for e in items:
                    self._act(sub, e, lambda e=e: self._choose_element(e, place_xy),
                              checked=self.element == e and not (self.ring or self.group or self.amino))
        menu.addSeparator()
        self._act(menu, "Click an atom to attach the label as a substituent, drag for a new bond",
                  lambda: None, enabled=False)

    def fill_rings(self, menu, ctx=None):
        ctx = ctx or {}
        for gname, items in RING_GROUPS.items():
            sub = menu.addMenu(gname)
            for name, _ in items:
                self._act(sub, name, lambda n=name: self._choose_ring(n, ctx.get("place_xy"), ctx.get("atom_id")),
                          checked=self.ring == name)

    def fill_groups(self, menu, ctx=None):
        ctx = ctx or {}
        gen = menu.addMenu("R / X / Y / Z groups")
        self.fill_generic(gen, ctx)
        for gname, items in GROUP_CATEGORIES.items():
            sub = menu.addMenu(gname)
            for name, _ in items:
                self._act(sub, name, lambda n=name: self._choose_group(
                    n, ctx.get("place_xy"), ctx.get("atom_id"), ctx.get("bond_key"), ctx.get("xy")),
                    checked=self.group == name)

    def fill_amino(self, menu, ctx=None):
        ctx = ctx or {}
        for name, _ in AMINO_ACIDS:
            self._act(menu, name, lambda n=name: self._choose_amino(n, ctx.get("place_xy")),
                      checked=self.amino == name)

    def fill_arrows(self, menu, ctx=None):
        ctx = ctx or {}
        pxy = ctx.get("place_xy") or ctx.get("xy")
        pxy = pxy if ctx.get("place_xy") is not None else None
        for label, kind in (("Reaction arrow  →", "arrow"), ("Equilibrium arrow  ⇌", "eq"),
                            ("Resonance arrow  ↔", "dbl"), ("Retrosynthetic arrow  ⇒", "retro"),
                            ("Dashed arrow  ⇢", "dashed")):
            self._act(menu, label, lambda k=kind: self.choose_arrow(k, pxy), checked=self.arrow_tool == kind)
        menu.addSeparator()
        cm = menu.addMenu("Curved arrows (mechanism)")
        self._act(cm, "Full-headed curved arrow (2 e⁻ pair)", lambda: self.choose_arrow("curve", pxy),
                  checked=self.arrow_tool == "curve")
        self._act(cm, "Half-headed curved arrow (1 e⁻, fish-hook)", lambda: self.choose_arrow("curve1", pxy),
                  checked=self.arrow_tool == "curve1")
        menu.addSeparator()
        self._act(menu, "Plus sign  +", lambda: self._place_annot("plus", ctx))
        self._act(menu, "Heat symbol  Δ", lambda: self._place_annot("text", ctx, "Δ"))
        self._act(menu, "Text label...", lambda: self._place_annot("textdlg", ctx))

    def _place_annot(self, kind, ctx, text=""):
        xy = ctx.get("place_xy") or ctx.get("xy")
        if xy is None:
            c = self.widget.width() / 2, self.widget.height() / 2
            xy = self.last_xy or c
        if kind == "textdlg":
            self.add_text_label(xy)
        else:
            self.add_annot(kind, xy, text, 1.7 if text == "Δ" else 1.0)

    def fill_bonds(self, menu, ctx=None):
        bgrp = QtGui.QActionGroup(menu)
        bgrp.setExclusive(True)
        for order, name in BOND_NAMES.items():
            a = self._act(menu, name, lambda o=order: self.set_bond(o), checked=order == self.bond_order)
            bgrp.addAction(a)
        menu.addSeparator()
        tm = menu.addMenu(f"Bond line width (now x{self.bond_scale:.2f})")
        for label, kw in (("Thinner", dict(scale_by=0.8)), ("Thicker", dict(scale_by=1.25)),
                          ("ACS 0.6 pt  x1.0", dict(factor=1.0)), ("Bold  x2.0", dict(factor=2.0))):
            self._act(tm, label, lambda k=kw: self.set_bond_thickness(**k))

    def fill_charges(self, menu, ctx=None):
        ctx = ctx or {}
        aid = ctx.get("atom_id")
        if aid is not None and aid in self.atoms:
            q = split_el(self.atoms[aid].el)[1]
            self._act(menu, f"Set formal charge of this atom... (now {q:+d})", lambda: self.charge_dialog(aid))
            menu.addSeparator()
        self._act(menu, "Show lone pairs (dots)", self.toggle_lone_pairs, checked=self.show_lp)
        self._act(menu, "Draw formal charges in circles  ⊕ ⊖", self.toggle_circled_charges,
                  checked=self.circled)
        if aid is None:
            self._act(menu, "(right-click an atom to set its charge)", lambda: None, enabled=False)

    def fill_edit(self, menu, ctx=None):
        self._act(menu, "Undo   Ctrl+Z", self.undo)
        self._act(menu, "Select all   Ctrl+A", self.select_all)
        n = bool(self.selected or self.sel_annots)
        self._act(menu, "Delete selection   Ctrl+X / Del", self.delete_selected, enabled=n)
        self._act(menu, "Duplicate selection   Ctrl+D", self.duplicate_selected, enabled=n)
        menu.addSeparator()
        self._act(menu, "Colour selected atoms...   Ctrl+K", self.color_selected, enabled=n)
        self._act(menu, "Reset colour of selection", lambda: self.apply_color(None), enabled=n)
        self._act(menu, "Flip label horizontal ⇄ vertical   Ctrl+Tab", self.toggle_orient, enabled=n)
        self._act(menu, "ACS clean-up of selection   Ctrl+M", self.clean_structure)

    def fill_view(self, menu, ctx=None):
        self._act(menu, "Cycle stereo display   Ctrl+H", self.cycle_stereo_display)
        self._act(menu, "Atom letters: black ⇄ element colours   Ctrl+G", self.toggle_mono_letters,
                  checked=not self.mono_letters)
        self._act(menu, "Show lone pairs", self.toggle_lone_pairs, checked=self.show_lp)
        self._act(menu, "Circled formal charges", self.toggle_circled_charges, checked=self.circled)
        tt = menu.addMenu("Text label weight")
        self._act(tt, "Regular (ACS)", lambda: self.set_text_thickness(False), checked=not self.text_bold)
        self._act(tt, "Bold", lambda: self.set_text_thickness(True), checked=self.text_bold)
        self._act(menu, "Minimise while drawing", self.toggle_auto_clean, checked=self.auto_clean)
        self._act(menu, "Zoom to fit   Home", self.fit_view)

    def _flat_top(self, menu, ctx):
        """Common atoms, more atoms, periodic table, text label and reaction arrow - all on one level."""
        place_xy = ctx.get("place_xy")
        grp = QtGui.QActionGroup(menu)
        grp.setExclusive(True)
        def add_el(m, el):
            a = self._act(m, el, lambda e=el: self._choose_element(e, place_xy),
                          checked=(not (self.ring or self.group or self.amino or self.arrow_tool)
                                   and el == self.element))
            grp.addAction(a)

        for el in PRIMARY_ELEMENTS:
            add_el(menu, el)
        more = menu.addMenu("More atoms")
        for el in MORE_ELEMENTS:
            add_el(more, el)
        self._act(menu, "Periodic table...", lambda: self.open_periodic_table(place_xy))
        self._act(menu, "Text label...", lambda: self._place_annot("textdlg", ctx))
        pxy = place_xy
        self._act(menu, "Reaction arrow  \u2192", lambda: self.choose_arrow("arrow", pxy),
                  checked=self.arrow_tool == "arrow")
        self._act(menu, "Plus sign  +", lambda: self._place_annot("plus", ctx))

    def show_menu(self, atom_id, bond_key, xy):
        menu = QtWidgets.QMenu(self.widget)
        ctx = dict(place_xy=xy if (atom_id is None and bond_key is None) else None,
                   atom_id=atom_id, bond_key=bond_key, xy=xy)
        n_sel = len(self.selected)
        sel_comps = self.selected_components()
        here = None
        if atom_id is not None and atom_id in self.atoms and self.atoms[atom_id].g3 is not None:
            here = [c for c in self.components() if atom_id in c]
        show3d = bool(sel_comps) and any(not self._is3d(c) for c in sel_comps)
        show2d = (bool(sel_comps) and any(self._is3d(c) for c in sel_comps)) or bool(here)
        if show3d:
            self._act(menu, "Show selected in 3D   Ctrl+3", lambda cs=sel_comps: self.to_3d(cs))
        if show2d:
            self._act(menu, "Back to 2D   Ctrl+3", lambda cs=(sel_comps if any(self._is3d(c) for c in sel_comps) else here):
                      self.to_2d(cs))
        if show3d or show2d:
            menu.addSeparator()
        if atom_id is not None and atom_id in self.atoms:
            q = split_el(self.atoms[atom_id].el)[1]
            self._act(menu, f"Set formal charge...  (now {q:+d})", lambda i=atom_id: self.charge_dialog(i))
            menu.addSeparator()
        if atom_id is None:
            self._flat_top(menu, ctx)
            menu.addSeparator()
        if atom_id is not None and atom_id in self.atoms:
            r = self.atoms[atom_id]
            self._act(menu, f"Delete atom ({r.el}{atom_id + 1})", lambda i=atom_id: self.delete_atoms([i]))
            if atom_id in self.centers:
                cur = self.centers[atom_id]
                sub = menu.addMenu(f"R/S stereocentre (now {cur if cur in ('R', 'S') else 'unspecified'})")
                for lab in ("R", "S"):
                    self._act(sub, f"Set {lab}", lambda i=atom_id, l=lab: self.set_chirality(i, l), checked=cur == lab)
                if atom_id in self.chiral:
                    self._act(sub, "Clear chirality", lambda i=atom_id: self.set_chirality(i, None))
            if r.orient or self.nh.get(atom_id, 0):
                self._act(menu, "Flip label horizontal ⇄ vertical", lambda i=atom_id: self._flip_label(i))
        if bond_key is not None and bond_key in self.bonds and self.bonds[bond_key] == 2:
            if bond_key in self.ez:
                cur = self.ez[bond_key]
                sub = menu.addMenu(f"E/Z double bond (now {cur})")
                for lab in ("E", "Z"):
                    self._act(sub, f"Set {lab}", lambda k=bond_key, l=lab: self.set_ez(k, l), checked=cur == lab)
        if self.atoms:
            self._act(menu, "Select all drawn atoms", self.select_all)
        if n_sel or self.sel_annots:
            self._act(menu, f"Delete selection   Ctrl+X", self.delete_selected)
            self._act(menu, "Duplicate selection   Ctrl+D", self.duplicate_selected)
            self._act(menu, "Colour...   Ctrl+K", self.color_selected)
        if n_sel:
            self._act(menu, "ACS clean-up   Ctrl+M", self.clean_structure)
        an = self.pick_annot(*xy)
        if an is not None:
            rec = self.annots[an]
            menu.addSeparator()
            if rec.kind == "text":
                self._act(menu, "Edit text...", lambda i=an: self.edit_text_label(i))
            if rec.kind in STRAIGHT_KINDS:
                self._act(menu, "Add text above the arrow...", lambda i=an: self.add_arrow_text(i, True))
                self._act(menu, "Add text below the arrow...", lambda i=an: self.add_arrow_text(i, False))
            if rec.kind.startswith("curve"):
                self._act(menu, "Flip curvature / head side", lambda i=an: self.flip_curve(i))
            self._act(menu, f"Delete {rec.describe()}", lambda i=an: self.delete_annots([i]))
        menu.addSeparator()
        owner = self.parent()
        file_fill = getattr(owner, "_fill_file", None)
        for title, fn in ((("File", file_fill),) if file_fill else ()) + (("R / X / Y groups", self.fill_generic),
                          ("Rings", self.fill_rings), ("Functional groups", self.fill_groups),
                          ("Amino acids", self.fill_amino), ("Arrows", self.fill_arrows),
                          ("Bonds", self.fill_bonds), ("Charges and Lone pairs", self.fill_charges),
                          ("Edit", self.fill_edit), ("View", self.fill_view)):
            fn(menu.addMenu(title.replace("&&", "&")), ctx)
        pos = QtGui.QCursor.pos()
        scr = QtGui.QGuiApplication.screenAt(pos) or QtGui.QGuiApplication.primaryScreen()
        geo = scr.availableGeometry()
        sz = menu.sizeHint()
        x = min(pos.x(), geo.right() - sz.width())
        y = pos.y() if pos.y() + sz.height() <= geo.bottom() else geo.bottom() - sz.height()
        menu.exec(QtCore.QPoint(max(geo.left(), x), max(geo.top(), y)))

    def _flip_label(self, aid):
        a = self.atoms[aid]
        a.orient = "h" if a.orient == "v" else "v"
        self._commit()

    def open_periodic_table(self, place_xy=None):
        dlg = PeriodicTableDialog(self.element if not (self.ring or self.group) else "", self.widget)
        if dlg.exec() and dlg.chosen:
            self._choose_element(dlg.chosen, place_xy)

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
            self.say(f"{name}: click an atom to attach it, or empty space to place it.")

    def _choose_amino(self, name, place_xy):
        self.set_amino(name)
        if place_xy is not None:
            self.place_first(*place_xy)
        else:
            self.say(f"{name}: click on empty space to place it (L-configuration).")

    def _group_target(self, atom_id, bond_key, xy):
        if atom_id is not None and atom_id in self.atoms:
            return atom_id
        if len(self.selected) == 1:
            return next(iter(self.selected))
        if bond_key is not None and bond_key in self.bonds:
            x, y = xy

            def dist(i):
                d = self.w2s(self.atoms[i].pos)
                return math.hypot(d[0] - x, d[1] - y)
            return min(bond_key, key=dist)
        return None

    def _choose_group(self, name, place_xy, atom_id, bond_key=None, xy=None):
        if not self._editable():
            return
        self.set_group(name)
        t = group_template(name)
        target = self._group_target(atom_id, bond_key, xy or (0, 0))
        if target is not None:
            self.attach_template(target, t)
        elif place_xy is not None:
            self.place_template(t, self.s2w(*place_xy))
        else:
            self.say(f"{name}: click an atom to attach it, or empty space to place it.")
