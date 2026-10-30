import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from rdkit import Chem

from chempad_widget import MolSketchWidget

if getattr(sys, 'frozen', False):
    import pyi_splash

mainpath = os.path.dirname(__file__)
iconx = os.path.join(mainpath, r'icony.ico')


class TestWindow(QtWidgets.QMainWindow):
    """All keyboard shortcuts (Ctrl+P, Ctrl+M, Ctrl+K, Ctrl+Tab, Ctrl+E, Ctrl+Q ...) live in the
    sketcher itself, so no window-level shortcuts are defined here (they would clash)."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChemPad")
        self.setWindowIcon(QtGui.QIcon(iconx))
        self.sketcher = MolSketchWidget()
        self.setCentralWidget(self.sketcher)
        self.sketcher.message.connect(lambda m: self.statusBar().showMessage(m, 2000))
        self.resize(1100, 700)

    def closeEvent(self, ev):
        if self.sketcher.confirm_close():
            ev.accept()
        else:
            ev.ignore()


def selftest():
    import tempfile
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ChemPad")
    app.setDesktopFileName("chempad")
    w = MolSketchWidget()
    w.resize(900, 600)
    w.show()
    app.processEvents()

    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    w.set_smiles(aspirin)
    m = w.mol()
    assert m is not None and m.GetNumHeavyAtoms() == 13, "aspirin heavy atoms"
    assert w.smiles() == Chem.MolToSmiles(Chem.MolFromSmiles(aspirin)), "aspirin SMILES round trip"
    assert "V2000" in w.molblock(), "molblock"
    assert w.export("sdf").endswith("$$$$\n"), "sdf"
    assert "ATM" in w.export("pdb") or "ATOM" in w.export("pdb"), "3D pdb"
    assert w.export("inchi").startswith("InChI="), "inchi"
    with tempfile.TemporaryDirectory() as td:
        path, ff = w.save_sdf_3d(td)
        assert ff == "MMFF94" and os.path.getsize(path) > 0, "3D sdf MMFF94"
        sup = Chem.SDMolSupplier(path, removeHs=False)
        assert sup[0].GetConformer().Is3D(), "sdf is 3D"
        png, svg = w.export_images(os.path.join(td, "fig"))
        img = QtGui.QImage(png)
        assert abs(img.dotsPerMeterX() - 47244) < 3, "PNG is 1200 dpi"
        assert "<svg" in open(svg, encoding="utf-8").read(), "svg"
        w.set_smiles("[Na+].[Cl-]")
        path, ff = w.save_sdf_3d(td)
        print("NaCl forcefield:", ff)
        w.set_smiles(aspirin)

    ala = "C[C@H](N)C(=O)O"
    w.set_smiles(ala)
    assert w.smiles() == Chem.MolToSmiles(Chem.MolFromSmiles(ala)), "stereo round trip"

    w.copy_smiles()
    assert QtWidgets.QApplication.clipboard().text() == w.smiles(), "Ctrl+C clipboard"

    QtWidgets.QApplication.clipboard().setText("c1ccccc1")
    w.paste_smiles()
    assert w.mol().GetNumHeavyAtoms() == 6 + 6, "paste keeps the old drawing"

    w.clear()
    assert w.mol() is None and w.smiles() == "", "clear"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("ChemPad")
        app.setDesktopFileName("chempad")
        win = TestWindow()
        if getattr(sys, 'frozen', False):
            pyi_splash.close()
        win.show()
        sys.exit(app.exec())
