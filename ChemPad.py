import sys

from PySide6 import QtCore, QtGui, QtWidgets

from rdkit import Chem

from chempad_widget import MolSketchWidget
import os

if getattr(sys, 'frozen', False):
   import pyi_splash

mainpath = os.path.dirname(sys.executable)
iconx = os.path.join(mainpath, r'icony.ico')

class TestWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChemPad")
        self.setWindowIcon(QtGui.QIcon(iconx))
        self.sketcher = MolSketchWidget()
        self.setCentralWidget(self.sketcher)
        self.sketcher.message.connect(lambda m: self.statusBar().showMessage(m, 5000))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, activated=self.export_sdf)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, activated=self.save_image)
        self.resize(900, 650)

    def closeEvent(self, ev):
        if self.sketcher.confirm_close():
            ev.accept()
        else:
            ev.ignore()

    def export_sdf(self):
        self.sketcher.export_sdf_dialog()

    def save_image(self):
        self.sketcher.save_image_dialog()


def selftest():
    app = QtWidgets.QApplication(sys.argv)
    w = MolSketchWidget()
    w.resize(600, 450)
    w.show()
    app.processEvents()

    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    w.set_smiles(aspirin)
    m = w.mol()
    assert m is not None and m.GetNumHeavyAtoms() == 13, "aspirin heavy atoms"
    assert w.smiles() == Chem.MolToSmiles(Chem.MolFromSmiles(aspirin)), "aspirin SMILES round trip"
    assert "V2000" in w.molblock(), "molblock"
    assert w.export("sdf").endswith("$$$$\n"), "sdf"
    assert "ATOM" in w.export("pdb"), "3D pdb"
    assert w.export("inchi").startswith("InChI="), "inchi"
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path, ff = w.save_sdf_3d(td)
        assert ff == "MMFF94" and os.path.getsize(path) > 0, "3D sdf MMFF94"
        sup = Chem.SDMolSupplier(path, removeHs=False)
        assert sup[0].GetConformer().Is3D(), "sdf is 3D"
        png = w.save_image_2d(td)
        assert os.path.getsize(png) > 0, "png"
        w.set_smiles("[Na+].[Cl-]")
        path, ff = w.save_sdf_3d(td)
        print("NaCl forcefield:", ff)
        w.set_smiles(aspirin)

    ala = "C[C@H](N)C(=O)O"
    w.set_smiles(ala)
    assert w.smiles() == Chem.MolToSmiles(Chem.MolFromSmiles(ala)), "stereo round trip"

    w.copy_smiles()
    assert QtWidgets.QApplication.clipboard().text() == w.smiles(), "Ctrl+C clipboard"

    w.clear()
    assert w.mol() is None and w.smiles() == "", "clear"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        app = QtWidgets.QApplication(sys.argv)
        win = TestWindow()
        if getattr(sys, 'frozen', False):
                  pyi_splash.close()
        win.show()
        sys.exit(app.exec())
