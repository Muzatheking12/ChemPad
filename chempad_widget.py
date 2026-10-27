# MolSketchWidget - an embeddable molecule sketcher for PySide6 (VTK + RDKit)
#
# FILES (keep them in the same folder)
#   chempad_core.py       sketch engine (drawing, bonds, rings, groups, 3D embedding)
#   chempad_widget.py  this file, the QWidget you embed
#
# INSTALL
#   pip install PySide6 vtk rdkit numpy
#
# EMBED IN A PySide6 APP
#   import sys
#   from PySide6 import QtWidgets
#   from chempad_widget import MolSketchWidget
#
#   app = QtWidgets.QApplication(sys.argv)
#   win = QtWidgets.QMainWindow()
#   sketcher = MolSketchWidget(dark=False, editable=True)
#   win.setCentralWidget(sketcher)
#   sketcher.message.connect(lambda m: win.statusBar().showMessage(m, 5000))
#   win.resize(900, 650)
#   win.show()
#   sys.exit(app.exec())
#
#   It also works inside any layout:  layout.addWidget(MolSketchWidget())
#   Ask before closing the window:
#       def closeEvent(self, ev):
#           ev.accept() if self.sketcher.confirm_close() else ev.ignore()
#
# SIGNALS
#   smilesChanged(str)     new canonical SMILES after every edit
#   moleculeChanged()      the drawing changed
#   message(str)           status text for a status bar
#   dockRequested(list)    list of {"smiles": ...} dicts (Ctrl+R)
#   tableRequested(list)   list of {"smiles": ...} dicts (Ctrl+Shift+A)
#
# SMILES
#   smi = sketcher.smiles()                       # canonical SMILES of the drawing, "" if empty
#   smi = sketcher.smiles(selected_only=True)     # only the selected compounds
#   sketcher.set_smiles("CC(=O)Oc1ccccc1C(=O)O")  # draw it (raises ValueError if invalid)
#   sketcher.smilesChanged.connect(print)         # live updates while drawing
#   sketcher.copy_smiles(); sketcher.paste_smiles()
#
# MOL (RDKit Mol, 2D, as drawn)
#   mol = sketcher.mol()                          # rdkit.Chem.Mol or None
#   mol = sketcher.mol(selected_only=True)
#   block = sketcher.molblock()                   # MDL V2000 text
#   sketcher.set_mol(mol)                         # draw an RDKit Mol
#   sketcher.set_molblock(block)                  # draw a mol block
#   from rdkit.Chem import Descriptors
#   Descriptors.MolWt(sketcher.mol())
#
# MOL3D (RDKit Mol with explicit hydrogens and 3D coordinates)
#   m3 = sketcher.mol3d()                         # ETKDG embed + MMFF94/UFF minimisation, or None
#   m3 = sketcher.mol3d(selected_only=True)
#   conf = m3.GetConformer(); conf.GetPositions()
#
# EXPORT AND SAVE
#   text = sketcher.export("sdf")                 # returns the text; formats:
#                                                 #   mol, mdl, sdf, smi, smiles, txt,
#                                                 #   inchi, inchikey, pdb, xyz   (pdb/xyz are 3D)
#   sketcher.export("pdb", "out.pdb")             # also writes the file
#   sketcher.save("out.xyz")                      # format taken from the extension; .png gives a 2D image
#   path, ff = sketcher.save_sdf_3d("C:/folder")  # 3D SDF, ff is "MMFF94", "UFF" or "none"
#   png = sketcher.save_image_2d("C:/folder")     # 2D PNG drawn by RDKit
#   sketcher.export_sdf_dialog()                  # folder picker, then 3D SDF (Ctrl+E)
#   sketcher.save_image_dialog()                  # folder picker, then PNG (Ctrl+Q)
#
# SCENE FILES AND VIEW
#   sketcher.save_scene("work.pad"); sketcher.load_scene("work.pad")
#   sketcher.screenshot("view.png")
#   sketcher.set_dark(True)                       # black background
#   sketcher.set_editable(False)                  # read-only viewer
#   sketcher.reset_view(); sketcher.clear()
#   sketcher.is_modified()                        # unsaved changes?
#   sketcher.show_shortcuts()                     # shortcut table (Ctrl+W)
#
# PYINSTALLER
#   pyinstaller --name ChemPad --icon iconx.ico --noconsole --onedir ChemPad.py
#
import json
import math
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from rdkit import Chem
from rdkit.Chem import rdDepictor

from chempad_core import Sketch, build_mol, embed_3d, el_str

Qt = QtCore.Qt


class _VtkView(QVTKRenderWindowInteractor):

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def _xy(self, ev):
        r = self.devicePixelRatioF()
        p = ev.position()
        return p.x() * r, (self.height() - 1 - p.y()) * r

    def mousePressEvent(self, ev):
        self.setFocus()
        self.owner._mouse_press(ev, *self._xy(ev))

    def mouseMoveEvent(self, ev):
        self.owner._mouse_move(ev, *self._xy(ev))

    def mouseReleaseEvent(self, ev):
        self.owner._mouse_release(ev, *self._xy(ev))

    def wheelEvent(self, ev):
        self.owner._wheel(ev, *self._xy(ev))

    def keyPressEvent(self, ev):
        if not self.owner._key(ev):
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        sk = self.owner.sketch
        if ev.key() in (Qt.Key_Control, Qt.Key_Shift) and sk.scaling is not None:
            sk.end_scale()
        if ev.key() in (Qt.Key_Control, Qt.Key_Alt) and sk.rotating is not None:
            sk.end_rotate()

    def focusOutEvent(self, ev):
        if self.owner.sketch.scaling is not None:
            self.owner.sketch.end_scale()
        if self.owner.sketch.rotating is not None:
            self.owner.sketch.end_rotate()
        super().focusOutEvent(ev)


class MolSketchWidget(QtWidgets.QWidget):
    smilesChanged = QtCore.Signal(str)
    moleculeChanged = QtCore.Signal()
    message = QtCore.Signal(str)
    dockRequested = QtCore.Signal(list)
    tableRequested = QtCore.Signal(list)

    def __init__(self, parent=None, dark=False, editable=True):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._view = _VtkView(self)
        lay.addWidget(self._view)

        self.ren = vtk.vtkRenderer()
        self._view.GetRenderWindow().AddRenderer(self.ren)
        self._view.GetRenderWindow().GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleUser())

        self.sketch = Sketch(self.ren, self._view, self)
        self.sketch.smilesChanged.connect(self._on_smiles)
        self.sketch.statusMessage.connect(self.message)
        self.sketch.dockRequested.connect(self.dockRequested)
        self.sketch.tableRequested.connect(self.tableRequested)

        self._editable = bool(editable)
        self._dark = bool(dark)
        self._last_dir = ""
        self._inited = False
        self._panning = False
        self.set_dark(dark)
        self.sketch.reattach(0.0)
        self.sketch.set_editing(self._editable)
        self.reset_view()
        self._saved_key = self._state_key()

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    def _collect(self, ids=None):
        s = self.sketch
        ids = set(s.atoms) if ids is None else set(ids) & set(s.atoms)
        if not ids:
            return None
        elems = {i: s.atoms[i].element for i in ids}
        bonds = {k: o for k, o in s.bonds.items() if k[0] in ids and k[1] in ids}
        pos = {i: (float(s.atoms[i].pos2[0]), float(s.atoms[i].pos2[1])) for i in ids}
        chiral = {i: v for i, v in s.chiral.items() if i in ids}
        return elems, bonds, pos, chiral

    def _ids(self, selected_only):
        return set(self.sketch.selected) if (selected_only and self.sketch.selected) else None

    def mol(self, selected_only=False):
        c = self._collect(self._ids(selected_only))
        if c is None:
            return None
        m, err, _ = build_mol(*c)
        return Chem.Mol(m) if m is not None else None

    def smiles(self, selected_only=False):
        if selected_only:
            return ".".join(c["smiles"] for c in self.sketch.selected_compounds())
        return self.sketch.smiles

    def molblock(self, selected_only=False):
        m = self.mol(selected_only)
        return Chem.MolToMolBlock(m) if m is not None else ""

    def mol3d(self, selected_only=False):
        c = self._collect(self._ids(selected_only))
        if c is None:
            return None
        res = embed_3d(*c)
        return res[0] if res else None

    def export(self, fmt="mol", path=None, selected_only=False):
        f = fmt.lower().lstrip(".")
        if f in ("smi", "smiles", "txt"):
            text = self.smiles(selected_only) + "\n"
        else:
            if f in ("pdb", "xyz"):
                m = self.mol3d(selected_only)
            else:
                m = self.mol(selected_only)
            if m is None:
                raise ValueError("nothing drawn, or the drawn structure is not valid")
            if f in ("mol", "mdl"):
                text = Chem.MolToMolBlock(m)
            elif f == "sdf":
                text = Chem.MolToMolBlock(m) + "$$$$\n"
            elif f == "pdb":
                text = Chem.MolToPDBBlock(m)
            elif f == "xyz":
                text = Chem.MolToXYZBlock(m)
            elif f == "inchi":
                text = Chem.MolToInchi(m) + "\n"
            elif f == "inchikey":
                text = Chem.MolToInchiKey(m) + "\n"
            else:
                raise ValueError(f"unknown format '{fmt}'")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def save(self, path, selected_only=False):
        ext = str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else "mol"
        if ext == "png":
            from rdkit.Chem import Draw
            m = self.mol(selected_only)
            if m is None:
                raise ValueError("nothing drawn")
            Draw.MolToFile(m, str(path), size=(600, 450))
            return
        self.export(ext, path, selected_only)

    @staticmethod
    def _unique_path(directory, stem, ext):
        import os
        p = os.path.join(directory, f"{stem}.{ext}")
        n = 1
        while os.path.exists(p):
            n += 1
            p = os.path.join(directory, f"{stem}_{n}.{ext}")
        return p

    def save_sdf_3d(self, directory, name="molecule", selected_only=False):
        from rdkit.Chem import AllChem
        m = self.mol3d(selected_only)
        if m is None:
            raise ValueError("nothing drawn, or the drawn structure is not valid")
        ff = "none"
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=5000)
                ff = "MMFF94"
            elif AllChem.UFFHasAllMoleculeParams(m):
                AllChem.UFFOptimizeMolecule(m, maxIters=5000)
                ff = "UFF"
        except Exception:
            pass
        m.SetProp("_Name", name)
        m.SetProp("SMILES", self.smiles(selected_only))
        m.SetProp("ForceField", ff)
        path = self._unique_path(directory, name, "sdf")
        w = Chem.SDWriter(path)
        w.write(m)
        w.close()
        return path, ff

    def save_image_2d(self, directory, name="molecule", selected_only=False, size=(800, 600)):
        from rdkit.Chem import Draw
        m = self.mol(selected_only)
        if m is None:
            raise ValueError("nothing drawn, or the drawn structure is not valid")
        path = self._unique_path(directory, name, "png")
        Draw.MolToFile(m, path, size=size)
        return path

    def clear(self):
        self.sketch.clear()
        self.sketch.smilesChanged.emit("")
        self.reset_view()

    def set_smiles(self, smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise ValueError(f"invalid SMILES: {smi}")
        self.set_mol(m)

    def set_molblock(self, block):
        m = Chem.MolFromMolBlock(block)
        if m is None:
            raise ValueError("invalid mol block")
        self.set_mol(m)

    def set_mol(self, mol):
        m = Chem.RemoveHs(Chem.Mol(mol))
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        cip = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
        if m.GetNumConformers() == 0 or m.GetConformer().Is3D():
            m.RemoveAllConformers()
            rdDepictor.Compute2DCoords(m)
        Chem.Kekulize(m, clearAromaticFlags=True)
        xy = np.array(m.GetConformer().GetPositions())[:, :2]
        xy = xy - (xy.min(axis=0) + xy.max(axis=0)) / 2
        s = self.sketch
        s.clear()
        ids = {}
        for a in m.GetAtoms():
            i = a.GetIdx()
            p = xy[i]
            ids[i] = s._create_atom(el_str(a.GetSymbol(), a.GetFormalCharge()), (p[0], p[1], s.z0))
        for b in m.GetBonds():
            o = min(3, max(1, int(round(b.GetBondTypeAsDouble()))))
            k = (ids[b.GetBeginAtomIdx()], ids[b.GetEndAtomIdx()])
            s.bonds[(min(k), max(k))] = o
        s.chiral = {ids[i]: c for i, c in cip.items() if c in ("R", "S")}
        s._commit(fit=False)
        self.reset_view()

    def set_dark(self, dark):
        self._dark = bool(dark)
        self.ren.SetBackground(*((0.0, 0.0, 0.0) if dark else (1.0, 1.0, 1.0)))
        self.sketch.set_dark(dark)
        self.sketch.render()

    def set_editable(self, on):
        self._editable = bool(on)
        self.sketch.set_editing(self._editable)

    def _state_key(self):
        d = self.sketch.scene_dict()
        d.pop("camera", None)
        return json.dumps(d, sort_keys=True)

    def is_modified(self):
        return self._state_key() != self._saved_key

    def confirm_close(self):
        Box = QtWidgets.QMessageBox
        box = Box(self.window())
        box.setWindowTitle("ChemPad")
        box.setIcon(Box.Question)
        if self.is_modified():
            box.setText("Save the drawing before exiting?")
            box.setInformativeText("Your changes will be lost if you do not save them.")
            box.setStandardButtons(Box.Save | Box.Discard | Box.Cancel)
            box.setDefaultButton(Box.Save)
            r = box.exec()
            if r == Box.Cancel:
                return False
            if r == Box.Save:
                return self.save_scene() is not None
            return True
        box.setText("Exit ChemPad?")
        box.setStandardButtons(Box.Yes | Box.No)
        box.setDefaultButton(Box.No)
        return box.exec() == Box.Yes

    def export_sdf_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder to save the 3D SDF", self._last_dir)
        if not d:
            return None
        self._last_dir = d
        try:
            path, ff = self.save_sdf_3d(d, selected_only=bool(self.sketch.selected))
            self.message.emit(f"Saved {path}  (force field: {ff})")
            return path
        except Exception as exc:
            self.message.emit(f"SDF export failed: {exc}")
            return None

    def save_image_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder to save the 2D image", self._last_dir)
        if not d:
            return None
        self._last_dir = d
        try:
            path = self.save_image_2d(d, selected_only=bool(self.sketch.selected))
            self.message.emit(f"Saved {path}")
            return path
        except Exception as exc:
            self.message.emit(f"Image save failed: {exc}")
            return None

    def show_shortcuts(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Shortcuts and actions")
        lay = QtWidgets.QVBoxLayout(dlg)
        tbl = QtWidgets.QTableWidget(len(SHORTCUTS), 3)
        tbl.setHorizontalHeaderLabels(["Group", "Action", "Shortcut / mouse"])
        for r, (grp, action, keys) in enumerate(SHORTCUTS):
            for c, text in enumerate((grp, action, keys)):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(Qt.ItemIsEnabled)
                tbl.setItem(r, c, it)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.resizeColumnsToContents()
        lay.addWidget(tbl)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+W"), dlg, activated=dlg.accept)
        dlg.resize(760, 640)
        dlg.exec()

    def toggle_background(self):
        self.set_dark(not self._dark)

    def save_scene(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save ChemPad", "scene.pad", "ChemPad (*.pad)")
            if not path:
                return None
        if not str(path).lower().endswith(".pad"):
            path = str(path) + ".pad"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.sketch.scene_dict(), fh, indent=1)
        self._saved_key = self._state_key()
        self.message.emit(f"Saved scene {path}")
        return path

    def load_scene(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open ChemPad File", "", "ChemPad (*.pad *.ca)")
            if not path:
                return None
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        had_camera = self.sketch.load_scene_dict(d)
        self.set_dark(self._dark if "dark" not in d else bool(d["dark"]))
        if not had_camera:
            self.reset_view()
        self._saved_key = self._state_key()
        self.message.emit(f"Loaded scene {path}")
        return path

    def screenshot(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save screenshot", "scene.png", "PNG image (*.png)")
            if not path:
                return None
        if not str(path).lower().endswith(".png"):
            path = str(path) + ".png"
        rw = self._view.GetRenderWindow()
        rw.Render()
        f = vtk.vtkWindowToImageFilter()
        f.SetInput(rw)
        f.SetInputBufferTypeToRGB()
        f.ReadFrontBufferOff()
        f.Update()
        w = vtk.vtkPNGWriter()
        w.SetFileName(str(path))
        w.SetInputConnection(f.GetOutputPort())
        w.Write()
        self.message.emit(f"Saved screenshot {path}")
        return path

    def cycle_stereo_display(self):
        self.sketch.cycle_stereo_display()

    toggle_stereo_labels = cycle_stereo_display

    def reset_view(self):
        s, cam = self.sketch, self.ren.GetActiveCamera()
        if s.atoms:
            P = np.array([r.pos[:2] for r in s.atoms.values()])
            lo, hi = P.min(axis=0), P.max(axis=0)
        else:
            lo = hi = np.zeros(2)
        c, ext = (lo + hi) / 2, hi - lo
        w, h = self.ren.GetSize()
        aspect = max(w, 1) / max(h, 1)
        half_h = max(ext[1] / 2 * 1.3 + 1.5, (ext[0] / 2 * 1.3 + 1.5) / aspect, 5.0)
        dist = half_h / math.tan(math.radians(cam.GetViewAngle()) / 2.0)
        cam.ParallelProjectionOff()
        cam.SetFocalPoint(c[0], c[1], s.z0)
        cam.SetPosition(c[0], c[1], s.z0 + dist)
        cam.SetViewUp(0, 1, 0)
        self.ren.ResetCameraClippingRange()
        s.render()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._inited:
            self._inited = True
            self._view.Initialize()
            QtCore.QTimer.singleShot(0, self.reset_view)

    def copy_smiles(self):
        text = self.smiles(True) if self.sketch.selected else ""
        text = text or self.smiles()
        if not text:
            self.message.emit("Nothing to copy.")
            return ""
        QtWidgets.QApplication.clipboard().setText(text)
        self.message.emit(f"Copied SMILES: {text}")
        return text

    def paste_smiles(self):
        text = QtWidgets.QApplication.clipboard().text().strip()
        if not text:
            return
        try:
            self.set_smiles(text.split()[0])
        except ValueError:
            self.message.emit("Clipboard does not contain a valid SMILES.")

    def _on_smiles(self, smi):
        self.smilesChanged.emit(smi)
        self.moleculeChanged.emit()

    def _start_pan(self, ev, x, y):
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self.sketch.pan_last = (x, y)
            self._view.setCursor(Qt.ClosedHandCursor)
            return True
        return False

    def _mouse_press(self, ev, x, y):
        if self._start_pan(ev, x, y):
            return
        if not self._editable:
            return
        if ev.button() == Qt.LeftButton:
            self.sketch.press("left", x, y, bool(ev.modifiers() & Qt.ShiftModifier))
        elif ev.button() == Qt.RightButton:
            self.sketch.press("right", x, y)

    def _mouse_move(self, ev, x, y):
        if self._panning:
            self.sketch._pan_to(x, y)
            return
        mods = ev.modifiers()
        if ((mods & Qt.ControlModifier) and (mods & Qt.AltModifier) and not (mods & Qt.ShiftModifier)
                and not ev.buttons() and self._editable):
            sk = self.sketch
            if sk.rotating is not None or sk.begin_rotate(x, y):
                sk.rotate_to(x, y)
                return
        if self.sketch.rotating is not None:
            self.sketch.end_rotate()
        if (mods & Qt.ControlModifier) and (mods & Qt.ShiftModifier) and not ev.buttons() and self._editable:
            sk = self.sketch
            if sk.scaling is not None or sk.begin_scale(x, y):
                sk.scale_to(x, y)
                return
        if self.sketch.scaling is not None:
            self.sketch.end_scale()
        if not self._editable:
            return
        b = ev.buttons()
        self.sketch.move(x, y, bool(b & Qt.LeftButton), bool(b & Qt.RightButton))

    def _mouse_release(self, ev, x, y):
        if self._panning:
            self._panning = False
            self._view.unsetCursor()
            return
        if not self._editable:
            return
        if ev.button() == Qt.LeftButton:
            self.sketch.release("left", x, y)
        elif ev.button() == Qt.RightButton:
            self.sketch.release("right", x, y)

    def _wheel(self, ev, x, y):
        dy = ev.angleDelta().y()
        if not dy:
            return
        f = 1.15 ** (dy / 120.0)
        cam = self.ren.GetActiveCamera()
        dist = cam.GetDistance()
        if (f > 1 and dist / f < 3.0) or (f < 1 and dist / f > 300.0):
            return
        s = self.sketch
        w0 = s._world_at(x, y)
        cam.Dolly(f)
        self.ren.ResetCameraClippingRange()
        d = w0 - s._world_at(x, y)
        cam.SetPosition(*(np.array(cam.GetPosition()) + d))
        cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) + d))
        self.ren.ResetCameraClippingRange()
        s.render()

    def _key(self, ev):
        ctrl = bool(ev.modifiers() & Qt.ControlModifier)
        shift = bool(ev.modifiers() & Qt.ShiftModifier)
        k, s = ev.key(), self.sketch
        if ctrl and k == Qt.Key_C:
            self.copy_smiles()
        elif ctrl and k == Qt.Key_V:
            if self._editable:
                self.paste_smiles()
        elif ctrl and k == Qt.Key_Z:
            s.undo()
        elif ctrl and k == Qt.Key_A and shift:
            s.add_selected_to_table()
        elif ctrl and k == Qt.Key_A:
            s.select_all()
        elif ctrl and k == Qt.Key_R:
            s.dock_selected()
        elif ctrl and k == Qt.Key_W:
            self.show_shortcuts()
        elif ctrl and k == Qt.Key_E:
            self.export_sdf_dialog()
        elif ctrl and k == Qt.Key_Q:
            self.save_image_dialog()
        elif ctrl and k == Qt.Key_S:
            self.save_scene()
        elif ctrl and k == Qt.Key_O:
            self.load_scene()
        elif ctrl and k == Qt.Key_B:
            self.toggle_background()
        elif ctrl and k == Qt.Key_P:
            self.screenshot()
        elif ctrl and k == Qt.Key_D:
            s.duplicate_selected()
        elif ctrl and k == Qt.Key_G:
            s.toggle_mono_letters()
        elif ctrl and k == Qt.Key_H:
            self.cycle_stereo_display()
        elif ctrl and k == Qt.Key_M:
            s.minimize_selection()
        elif (ctrl and k == Qt.Key_X) or k in (Qt.Key_Delete, Qt.Key_Backspace):
            s.delete_selected()
        elif k == Qt.Key_Escape:
            s.clear_selection()
        elif k == Qt.Key_Home:
            self.reset_view()
        else:
            return False
        return True


SHORTCUTS = [
    ("Draw", "Place an atom / ring / group (several compounds)", "Left-click empty space"),
    ("Draw", "Draw a bond or chain", "Left-drag from an atom"),
    ("Draw", "Replace atom element / attach ring or group", "Left-click an atom"),
    ("Draw", "Cycle bond order (single / double / triple)", "Left-click a bond"),
    ("Draw", "Menu: atoms, rings, groups, bond type, bond thickness, stereo", "Right-click"),
    ("Draw", "Set R/S or E/Z, atom charge", "Right-click an atom / double bond"),
    ("Reaction scheme", "Add arrow, equilibrium, double-headed arrow, plus, heat symbol, text", "Right-click > Reaction scheme"),
    ("Reaction scheme", "Edit a text label (with degree symbol button)", "Right-click the label > Edit"),
    ("Reaction scheme", "Delete an arrow / plus / label", "Left-click it, or right-click > Delete"),
    ("Select", "Box select atoms and labels", "Right-drag"),
    ("Select", "Select everything", "Ctrl+A"),
    ("Select", "Clear the selection", "Esc"),
    ("Edit", "Undo (everything)", "Ctrl+Z"),
    ("Edit", "Delete the selection", "Ctrl+X / Delete / Backspace"),
    ("Edit", "Duplicate the selection", "Ctrl+D"),
    ("Edit", "Energy-minimise the selection", "Ctrl+M"),
    ("Edit", "Copy SMILES (selection, else everything)", "Ctrl+C"),
    ("Edit", "Paste SMILES", "Ctrl+V"),
    ("Transform", "Move the selection", "Shift + left-drag"),
    ("Transform", "Resize selection (compound, text, arrow length, plus)", "Ctrl+Shift + move the mouse"),
    ("Transform", "Rotate selected arrows (and atoms)", "Ctrl+Alt + move the mouse"),
    ("View", "Translate (pan) the view", "Middle-button drag"),
    ("View", "Zoom", "Mouse wheel"),
    ("View", "Reset the view", "Home"),
    ("View", "Cycle stereo display: none / labels / wedge-dash", "Ctrl+H"),
    ("View", "Background black <-> white", "Ctrl+B"),
    ("View", "Atom / functional-group letters: black <-> coloured", "Ctrl+G"),
    ("View", "Bond thickness (default thin)", "Right-click > Bond thickness"),
    ("View", "Text label thickness (thin / bold, separate from bonds)", "Right-click > Text label thickness"),
    ("Files", "Save the ChemPad scene (.pad)", "Ctrl+S"),
    ("Files", "Open a ChemPad scene (.pad)", "Ctrl+O"),
    ("Files", "Export 3D SDF (MMFF94, UFF fallback)", "Ctrl+E"),
    ("Files", "Save 2D image (PNG)", "Ctrl+Q"),
    ("Files", "Screenshot of the ChemPad scene (PNG)", "Ctrl+P"),
    ("Help", "This table", "Ctrl+W"),
]


def make_icon(size=128):
    pm = QtGui.QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    c, r = size / 2, size * 0.36
    pts = [QtCore.QPointF(c + r * math.cos(math.radians(60 * k + 30)),
                          c + r * math.sin(math.radians(60 * k + 30))) for k in range(6)]
    p.setPen(QtGui.QPen(QtGui.QColor("#1f6feb"), size * 0.06, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPolygon(QtGui.QPolygonF(pts))
    p.drawEllipse(QtCore.QPointF(c, c), r * 0.55, r * 0.55)
    p.end()
    return QtGui.QIcon(pm)


class _MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mol sketch")
        self.setWindowIcon(make_icon())
        self.sk = MolSketchWidget(self)
        self.setCentralWidget(self.sk)
        sb = self.statusBar()
        sb.addWidget(QtWidgets.QLabel(
            "Right-click: menu   Ctrl+C: copy SMILES   Ctrl+V: paste SMILES   "
            "Ctrl+Z: undo   Home: reset view   Pan: middle-button drag"))
        self.smi_label = QtWidgets.QLabel("")
        self.smi_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sb.addPermanentWidget(self.smi_label)
        self.sk.smilesChanged.connect(lambda s: self.smi_label.setText(s))
        self.sk.message.connect(lambda m: sb.showMessage(m, 5000))
        self.sk.tableRequested.connect(self._to_clipboard)
        self.sk.dockRequested.connect(self._to_clipboard)
        self.resize(900, 650)

    def closeEvent(self, ev):
        if self.sk.confirm_close():
            ev.accept()
        else:
            ev.ignore()

    def _to_clipboard(self, comps):
        text = ".".join(c["smiles"] for c in comps)
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Copied SMILES: {text}", 5000)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = _MainWindow()
    win.show()
    win.sk.setFocus()
    if len(sys.argv) > 1:
        try:
            win.sk.set_smiles(sys.argv[1])
        except ValueError as exc:
            win.statusBar().showMessage(str(exc), 5000)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
