# MolSketchWidget - an embeddable ACS-1996 style molecule sketcher for PySide6 (QPainter + RDKit)
#
# FILES (keep them in the same folder)
#   chempad_render.py   ACS drawing engine (labels Arial 10, bonds 14.4 pt, arrows, charges, lone pairs)
#   chempad_core.py     sketch engine (editing, clean-up, templates, amino acids, 3D embedding)
#   chempad_widget.py   this file, the QWidget you embed
#
# INSTALL
#   pip install PySide6 rdkit numpy            (VTK is no longer needed)
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
#   win.resize(1100, 700)
#   win.show()
#   sys.exit(app.exec())
#
#   There is no toolbar: right-click gives every menu.
#
# SIGNALS     smilesChanged(str)  moleculeChanged()  message(str)
#
# SMILES      sketcher.smiles(selected_only=False)   sketcher.set_smiles("CC(=O)O")   copy_smiles() / paste_smiles()
# MOL         sketcher.mol()  molblock()  set_mol(mol)  set_molblock(block)  add_mol(mol)
# MOL3D       sketcher.mol3d()                       RDKit Mol with Hs and an MMFF94/UFF optimised 3D conformer
# EXPORT      sketcher.export("sdf"|"mol"|"smi"|"inchi"|"inchikey"|"pdb"|"xyz", path=None)
#             sketcher.export_images("figure")       figure.png (1200 dpi) + figure.svg, ACS proportions
#             sketcher.save("out.png")               format from the extension (.png 1200 dpi, .svg, .xyz ...)
#             sketcher.save_sdf_3d(folder)           sketcher.save_image_2d(folder)
# SCENE       sketcher.save_scene("work.pad")  load_scene("work.pad")  (old VTK .pad files still open)
import json
import math
import os
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtSvg import QSvgGenerator

from rdkit import Chem

from chempad_core import Sketch, build_mol, embed_3d, el_str, rd_xy, short
from chempad_render import SceneRenderer

Qt = QtCore.Qt


class _Canvas(QtWidgets.QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMinimumSize(320, 240)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        self.owner.sketch.paint(p, self.width(), self.height())

    def focusNextPrevChild(self, nxt):
        return False

    def mousePressEvent(self, ev):
        self.setFocus()
        self.owner._mouse_press(ev)

    def mouseMoveEvent(self, ev):
        self.owner._mouse_move(ev)

    def mouseReleaseEvent(self, ev):
        self.owner._mouse_release(ev)

    def wheelEvent(self, ev):
        self.owner._wheel(ev)

    def keyReleaseEvent(self, ev):
        sk = self.owner.sketch
        if ev.key() in (Qt.Key_Control, Qt.Key_Shift) and sk.scaling is not None:
            sk.end_scale()
        if ev.key() in (Qt.Key_Control, Qt.Key_Alt) and sk.rotating is not None:
            sk.end_rotate()
        super().keyReleaseEvent(ev)

    def focusOutEvent(self, ev):
        sk = self.owner.sketch
        if sk.scaling is not None:
            sk.end_scale()
        if sk.rotating is not None:
            sk.end_rotate()
        super().focusOutEvent(ev)

    def leaveEvent(self, ev):
        if self.owner.sketch.hover_id is not None or self.owner.sketch.hover_bond is not None:
            self.owner.sketch.hover_id = self.owner.sketch.hover_bond = None
            self.update()


class MolSketchWidget(QtWidgets.QWidget):
    smilesChanged = QtCore.Signal(str)
    moleculeChanged = QtCore.Signal()
    message = QtCore.Signal(str)

    def __init__(self, parent=None, dark=False, editable=True):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.canvas = _Canvas(self)
        self.sketch = Sketch(self.canvas, self)
        self.sketch.smilesChanged.connect(self._on_smiles)
        self.sketch.statusMessage.connect(self.message)
        self.sketch.toolChanged.connect(self._tool_changed)
        self._editable = bool(editable)
        self._last_dir = ""
        self._panning = False
        self._fitted = False
        lay.addWidget(self.canvas, 1)
        self.set_dark(dark)
        self.sketch.set_editing(self._editable)
        self._saved_key = self._state_key()
        self._build_shortcuts()
        self._tool_changed()

    def sizeHint(self):
        return QtCore.QSize(1000, 680)

    def _fill_file(self, menu, ctx=None):
        sk = self.sketch
        sk._act(menu, "Open scene...   Ctrl+O", self.load_scene)
        sk._act(menu, "Save scene...   Ctrl+S", self.save_scene)
        menu.addSeparator()
        sk._act(menu, "Export PNG and SVG...   Ctrl+P", self.export_images_dialog)
        sk._act(menu, "Save 2D PNG to a folder...   Ctrl+Q", self.save_image_dialog)
        sk._act(menu, "Export 3D SDF (MMFF94)...   Ctrl+E", self.export_sdf_dialog)
        menu.addSeparator()
        sk._act(menu, "Copy SMILES   Ctrl+C", self.copy_smiles)
        sk._act(menu, "Paste SMILES / MOL block   Ctrl+V", self.paste_smiles)
        menu.addSeparator()
        sk._act(menu, "Dark / light background   Ctrl+B", self.toggle_background)
        sk._act(menu, "Shortcuts...   Ctrl+W", self.show_shortcuts)
        sk._act(menu, "Clear everything", self.clear)

    def _tool_changed(self):
        self.canvas.setCursor(Qt.CrossCursor if self.sketch.arrow_tool else Qt.ArrowCursor)

    def _commands(self):
        s = self.sketch
        return [
            ("Ctrl+C", "Edit", "Copy SMILES (selection, else everything)", self.copy_smiles),
            ("Ctrl+V", "Edit", "Paste SMILES / MOL block - adds the structure, keeps the drawing", self.paste_smiles),
            ("Ctrl+Z", "Edit", "Undo", s.undo),
            ("Ctrl+A", "Select", "Select everything", s.select_all),
            ("Esc", "Select", "Clear the selection / cancel an arrow tool", self._escape),
            ("Ctrl+X", "Edit", "Delete the selection", s.delete_selected),
            ("Delete", "Edit", "Delete the selection", s.delete_selected),
            ("Backspace", "Edit", "Delete the selection", s.delete_selected),
            ("Ctrl+D", "Edit", "Duplicate the selection", s.duplicate_selected),
            ("Ctrl+M", "Edit", "Minimise + ACS clean-up of the selection (regular rings, 14.4 pt, 120°)", s.clean_structure),
            ("Ctrl+K", "Edit", "Colour the selected atoms", s.color_selected),
            ("Ctrl+Tab", "Transform", "Selected compound horizontal ⇄ vertical", s.toggle_orient),
            ("Ctrl+3", "3D", "Selected compound(s) 2D \u21c4 3D (drag an atom to rotate)", s.toggle_3d),
            ("Ctrl+P", "Files", "Export PNG and SVG", self.export_images_dialog),
            ("Ctrl+Q", "Files", "Save 2D PNG into a folder", self.save_image_dialog),
            ("Ctrl+E", "Files", "Export 3D SDF (MMFF94, UFF fallback)", self.export_sdf_dialog),
            ("Ctrl+S", "Files", "Save the ChemPad scene (.pad)", self.save_scene),
            ("Ctrl+O", "Files", "Open a ChemPad scene (.pad)", self.load_scene),
            ("Ctrl+H", "View", "Cycle stereo display: none / labels / wedge-dash", s.cycle_stereo_display),
            ("Ctrl+G", "View", "Atom letters: black (ACS) ⇄ element colours", s.toggle_mono_letters),
            ("Ctrl+L", "View", "Show / hide lone pairs", s.toggle_lone_pairs),
            ("Ctrl+J", "View", "Formal charges plain ⇄ in circles", s.toggle_circled_charges),
            ("Ctrl+B", "View", "Background black ⇄ white", self.toggle_background),
            ("Home", "View", "Zoom to fit", self.reset_view),
            ("Ctrl+W", "Help", "This table", self.show_shortcuts),
        ]

    def _build_shortcuts(self):
        self._shortcuts = []
        for seq, _, _, fn in self._commands():
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda f=fn: self._run(f))
            self._shortcuts.append(sc)

    def _run(self, fn):
        try:
            fn()
        except Exception as exc:
            self.message.emit("Error")

    def _escape(self):
        sk = self.sketch
        if sk.arrow_tool:
            sk.arrow_tool = None
            sk.toolChanged.emit()
        sk.clear_selection()

    def _collect(self, ids=None):
        s = self.sketch
        e, b, p, c = s.rd_inputs(ids)
        return (e, b, p, c) if e else None

    def _ids(self, selected_only):
        s = self.sketch
        if selected_only and s.selected:
            ids = set()
            for comp in s.selected_components():
                ids |= comp
            return ids
        return None

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
            m = self.mol3d(selected_only) if f in ("pdb", "xyz") else self.mol(selected_only)
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
            self.export_png(str(path), 1200, selected_only)
        elif ext == "svg":
            self.export_svg(str(path), selected_only)
        else:
            self.export(ext, path, selected_only)

    def _export_renderer(self, selected_only=None):
        s = self.sketch
        if selected_only is None:
            selected_only = bool(s.selected or s.sel_annots)
        if selected_only and (s.selected or s.sel_annots):
            ids = set()
            for comp in s.selected_components():
                ids |= comp
            r = SceneRenderer(s, ui=False, dark=False, atom_ids=ids or s.selected, annot_ids=s.sel_annots)
        else:
            r = SceneRenderer(s, ui=False, dark=False)
        b = r.bounds()
        if b is None:
            raise ValueError("nothing drawn")
        pad = 8.0
        return r, (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)

    def export_png(self, path, dpi=1200, selected_only=None, transparent=False):
        r, (x0, y0, x1, y1) = self._export_renderer(selected_only)
        sc = dpi / 72.0
        w, h = int(math.ceil((x1 - x0) * sc)), int(math.ceil((y1 - y0) * sc))
        while w * h > 220e6:
            sc *= 0.8
            dpi = int(round(sc * 72))
            w, h = int(math.ceil((x1 - x0) * sc)), int(math.ceil((y1 - y0) * sc))
        img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.QColor(0, 0, 0, 0) if transparent else QtGui.QColor(255, 255, 255))
        p = QtGui.QPainter(img)
        p.scale(sc, sc)
        p.translate(-x0, -y0)
        r.draw(p)
        p.end()
        dpm = int(round(dpi / 0.0254))
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        if not img.save(path, "PNG"):
            raise IOError(f"could not write {path}")
        return path, dpi

    def export_svg(self, path, selected_only=None):
        r, (x0, y0, x1, y1) = self._export_renderer(selected_only)
        w, h = x1 - x0, y1 - y0
        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(QtCore.QSize(int(math.ceil(w)), int(math.ceil(h))))
        gen.setViewBox(QtCore.QRectF(0, 0, w, h))
        gen.setResolution(72)
        gen.setTitle("ChemPad structure")
        gen.setDescription("ACS Document 1996: bond length 14.4 pt, line width 0.6 pt, Arial 10 pt")
        p = QtGui.QPainter(gen)
        p.translate(-x0, -y0)
        r.draw(p)
        p.end()
        return path

    def export_images(self, base, dpi=1200, selected_only=None):
        """Writes base.png (dpi) and base.svg; returns (png_path, svg_path)."""
        base = str(base)
        for ext in (".png", ".svg"):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
        png, used = self.export_png(base + ".png", dpi, selected_only)
        svg = self.export_svg(base + ".svg", selected_only)
        return png, svg

    def export_images_dialog(self):
        suggested = os.path.join(self._last_dir or os.path.expanduser("~"), "chempad")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export image: PNG and SVG",
            suggested, "Image name (*.png *.svg)")
        if not path:
            return None
        self._last_dir = os.path.dirname(path)
        try:
            png, svg = self.export_images(path)
            self.message.emit("Saved")
            return png, svg
        except Exception as exc:
            self.message.emit("Failed")
            return None

    @staticmethod
    def _unique_path(directory, stem, ext):
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

    def save_image_2d(self, directory, name="molecule", selected_only=False, size=None):
        path = self._unique_path(directory, name, "png")
        self.export_png(path, 1200, selected_only or None)
        return path

    def export_sdf_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder to save the 3D SDF", self._last_dir)
        if not d:
            return None
        self._last_dir = d
        try:
            path, ff = self.save_sdf_3d(d, selected_only=bool(self.sketch.selected))
            self.message.emit("Saved")
            return path
        except Exception as exc:
            self.message.emit("Failed")
            return None

    def save_image_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder to save the 2D image", self._last_dir)
        if not d:
            return None
        self._last_dir = d
        try:
            path = self.save_image_2d(d, selected_only=bool(self.sketch.selected))
            self.message.emit("Saved")
            return path
        except Exception as exc:
            self.message.emit("Failed")
            return None

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
        """Replace the drawing by this molecule (ACS layout)."""
        self.sketch.clear()
        self.sketch.add_mol(mol, center=(0.0, 0.0), select=False)
        self.reset_view()

    def add_mol(self, mol, select=True):
        """Add a molecule next to what is drawn; nothing is removed."""
        return self.sketch.add_mol(mol, select=select)

    def set_dark(self, dark):
        self.sketch.set_dark(dark)

    @property
    def _dark(self):
        return self.sketch.dark

    def set_editable(self, on):
        self._editable = bool(on)
        self.sketch.set_editing(self._editable)

    def _state_key(self):
        d = self.sketch.scene_dict()
        d.pop("view", None)
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

    def show_shortcuts(self):
        rows = [(g, a, k) for k, g, a, _ in self._commands()]
        rows += MOUSE_HELP
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Shortcuts and actions")
        lay = QtWidgets.QVBoxLayout(dlg)
        tbl = QtWidgets.QTableWidget(len(rows), 3)
        tbl.setHorizontalHeaderLabels(["Group", "Action", "Shortcut / mouse"])
        for r, (grp, action, keys) in enumerate(rows):
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
        dlg.resize(900, 700)
        dlg.exec()

    def toggle_background(self):
        self.set_dark(not self.sketch.dark)

    def save_scene(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ChemPad", "scene.pad", "ChemPad (*.pad)")
            if not path:
                return None
        if not str(path).lower().endswith(".pad"):
            path = str(path) + ".pad"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.sketch.scene_dict(), fh, indent=1)
        self._saved_key = self._state_key()
        self.message.emit("Saved")
        return path

    def load_scene(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open ChemPad File", "", "ChemPad (*.pad *.ca)")
            if not path:
                return None
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        had_view = self.sketch.load_scene_dict(d)
        if not had_view:
            self.reset_view()
        self._saved_key = self._state_key()
        self.message.emit("Loaded")
        return path

    def screenshot(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", "scene.png", "PNG image (*.png)")
            if not path:
                return None
        if not str(path).lower().endswith(".png"):
            path = str(path) + ".png"
        self.canvas.grab().save(str(path), "PNG")
        self.message.emit("Saved")
        return path

    def cycle_stereo_display(self):
        self.sketch.cycle_stereo_display()

    toggle_stereo_labels = cycle_stereo_display

    def reset_view(self):
        self.sketch.fit_view()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._fitted:
            self._fitted = True
            QtCore.QTimer.singleShot(0, self.reset_view)

    def copy_smiles(self):
        text = self.smiles(True) if self.sketch.selected else ""
        text = text or self.smiles()
        if not text:
            self.message.emit("Empty")
            return ""
        QtWidgets.QApplication.clipboard().setText(text)
        self.message.emit("Copied")
        return text

    def paste_smiles(self):
        """Ctrl+V: the structure from the clipboard is ADDED; everything already drawn stays."""
        if not self._editable:
            return
        text = QtWidgets.QApplication.clipboard().text().strip()
        if not text:
            self.message.emit("Empty")
            return
        pt = self.canvas.mapFromGlobal(QtGui.QCursor.pos())
        at = (pt.x(), pt.y()) if self.canvas.rect().contains(pt) else None
        self.sketch.paste_text(text, at)

    def _on_smiles(self, smi):
        self.smilesChanged.emit(smi)
        self.moleculeChanged.emit()

    def _xy(self, ev):
        p = ev.position()
        return p.x(), p.y()

    def _mouse_press(self, ev):
        x, y = self._xy(ev)
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self.sketch.pan_last = (x, y)
            self.canvas.setCursor(Qt.ClosedHandCursor)
            return
        if not self._editable:
            return
        if ev.button() == Qt.LeftButton:
            self.sketch.press("left", x, y, bool(ev.modifiers() & Qt.ShiftModifier))
        elif ev.button() == Qt.RightButton:
            self.sketch.press("right", x, y, bool(ev.modifiers() & Qt.ShiftModifier))

    def _mouse_move(self, ev):
        x, y = self._xy(ev)
        sk = self.sketch
        if self._panning:
            sk.pan_to(x, y)
            return
        mods = ev.modifiers()
        ctrl, alt, shift = bool(mods & Qt.ControlModifier), bool(mods & Qt.AltModifier), bool(mods & Qt.ShiftModifier)
        if ctrl and alt and not ev.buttons() and self._editable:
            if sk.rotating is not None or sk.begin_rotate(x, y):
                sk.rotate_to(x, y, free=shift)
                return
        elif sk.rotating is not None:
            sk.end_rotate()
        if ctrl and shift and not alt and not ev.buttons() and self._editable:
            if sk.scaling is not None or sk.begin_scale(x, y):
                sk.scale_to(x, y)
                return
        elif sk.scaling is not None:
            sk.end_scale()
        if not self._editable:
            return
        b = ev.buttons()
        sk.move(x, y, bool(b & Qt.LeftButton), bool(b & Qt.RightButton))

    def _mouse_release(self, ev):
        x, y = self._xy(ev)
        if self._panning and ev.button() == Qt.MiddleButton:
            self._panning = False
            self._tool_changed()
            return
        if not self._editable:
            return
        if ev.button() == Qt.LeftButton:
            self.sketch.release("left", x, y)
        elif ev.button() == Qt.RightButton:
            self.sketch.release("right", x, y, bool(ev.modifiers() & Qt.ShiftModifier))

    def _wheel(self, ev):
        dy = ev.angleDelta().y()
        if dy:
            x, y = self._xy(ev)
            self.sketch.zoom_at(x, y, 1.15 ** (dy / 120.0))


MOUSE_HELP = [
    ("Draw", "Place an atom / ring / group / amino acid", "Left-click empty space"),
    ("Draw", "Draw a bond or chain (30° steps, 14.4 pt)", "Left-drag from an atom"),
    ("Draw", "Replace atom element / attach ring or group / add R, X, Y", "Left-click an atom"),
    ("Draw", "Cycle bond order (single / double / triple)", "Left-click a bond"),
    ("Draw", "All menus (atoms, rings, groups, amino acids, arrows ...)", "Right-click"),
    ("Draw", "Formal charge / R-S / E-Z", "Right-click an atom / double bond"),
    ("Arrows", "Draw an arrow or curved arrow", "Choose it in Arrows, then drag on the page"),
    ("Arrows", "Edit an arrow (length, angle, curve)", "Click it, then drag its square handles"),
    ("Arrows", "Add conditions above / below an arrow", "Right-click the arrow"),
    ("Arrows", "Delete an arrow / text (never by a plain click)", "Select it, then Delete; or right-click > Delete"),
    ("Select", "Box select atoms and objects", "Right-drag"),
    ("Select", "Add another box to the selection", "Shift + right-drag"),
    ("3D", "Show the selected compound(s) in 3D, same canvas", "Ctrl+3, or right-click > Show selected in 3D"),
    ("3D", "Rotate a 3D compound", "Left-drag any of its atoms"),
    ("3D", "Back to 2D", "Ctrl+3 again, or right-click > Back to 2D"),
    ("Transform", "Move the selection", "Shift + left-drag"),
    ("Transform", "Rotate the selection in 15° steps", "Ctrl+Alt + move the mouse"),
    ("Transform", "Rotate the selection freely (any angle)", "Ctrl+Alt+Shift + move the mouse"),
    ("Transform", "Rotate a substituent about its bond", "Ctrl+Alt with the mouse over a (non-ring) bond"),
    ("Transform", "Resize selection (structures return to ACS format)", "Ctrl+Shift + move the mouse"),
    ("View", "Pan", "Middle-button drag"),
    ("View", "Zoom", "Mouse wheel"),
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
        self.setWindowTitle("ChemPad")
        self.setWindowIcon(make_icon())
        self.sk = MolSketchWidget(self)
        self.setCentralWidget(self.sk)
        sb = self.statusBar()
        self.smi_label = QtWidgets.QLabel("")
        self.smi_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sb.addPermanentWidget(self.smi_label)
        self.sk.smilesChanged.connect(lambda s: self.smi_label.setText(s))
        self.sk.message.connect(lambda m: sb.showMessage(m, 2000))
        self.resize(1100, 700)

    def closeEvent(self, ev):
        if self.sk.confirm_close():
            ev.accept()
        else:
            ev.ignore()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = _MainWindow()
    win.show()
    win.sk.canvas.setFocus()
    if len(sys.argv) > 1:
        try:
            win.sk.set_smiles(sys.argv[1])
        except ValueError as exc:
            win.statusBar().showMessage("Invalid", 2000)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
