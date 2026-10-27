<p align="center">
  <img src="assets/icon.png" alt="ChemPad icon" width="128" height="128">
</p>

<h1 align="center">ChemPad</h1>

<p align="center"><b>A Notepad for Chemistry</b></p>

<p align="center">
  Draw molecules, read the SMILES, and export 2D and 3D structures. Built with PySide6, VTK and RDKit.
</p>

---

## Features

- Draw atoms, single/double/triple bonds, chains and fused structures with the mouse
- Ring library (carbocycles, aromatic and saturated heterocycles, fused rings)
- Functional group library, attached to an atom straight from the right-click menu
- Formal charges, any element through the periodic table dialog
- Stereochemistry: R/S and E/Z labels, or wedge/dash bonds
- Reaction scheme tools: arrows, equilibrium arrows, plus signs, heat symbol, text labels
- Live canonical SMILES, copy and paste SMILES
- Energy minimisation of the whole drawing or a selection
- Export 3D SDF (MMFF94, UFF fallback), 2D PNG, and screenshots
- Save and reopen drawings as `.pad` files
- Black or white background, undo, box select, duplicate, move, resize and rotate

## Requirements

- Python 3.10+
- PySide6
- vtk
- rdkit
- numpy

```
pip install PySide6 vtk rdkit numpy
```

## Run

```
python ChemPad.py
```

Self-test:

```
python ChemPad.py --selftest
```

## Project files

| File | Purpose |
|---|---|
| `ChemPad.py` | Application window and entry point |
| `chempad_widget.py` | `MolSketchWidget`, the embeddable PySide6 widget |
| `chempad_core.py` | Sketch engine: drawing, bonds, rings, groups, 3D embedding |

## Use the widget in your own PySide6 app

```python
import sys
from PySide6 import QtWidgets
from chempad_widget import MolSketchWidget

app = QtWidgets.QApplication(sys.argv)
sketcher = MolSketchWidget()
sketcher.smilesChanged.connect(print)
sketcher.show()
sys.exit(app.exec())
```

```python
sketcher.smiles()                  # canonical SMILES
sketcher.mol()                     # RDKit Mol, 2D
sketcher.mol3d()                   # RDKit Mol, 3D with hydrogens
sketcher.set_smiles("CC(=O)Oc1ccccc1C(=O)O")
sketcher.export("sdf", "out.sdf")  # mol, sdf, smi, inchi, inchikey, pdb, xyz
```

The header of `chempad_widget.py` documents the full API.

## Mouse and keyboard

| Action | Control |
|---|---|
| Place atom, ring or group | Left-click empty space |
| Draw a bond or chain | Left-drag from an atom |
| Change an atom's element | Left-click the atom |
| Cycle bond order | Left-click a bond |
| Menu: atoms, rings, groups, bonds, stereo | Right-click |
| Box select | Right-drag |
| Move selection | Shift + left-drag |
| Resize selection | Ctrl+Shift + move mouse |
| Rotate selection | Ctrl+Alt + move mouse |
| Pan | Middle-button drag |
| Zoom | Mouse wheel |
| Reset view | Home |

| Shortcut | Action |
|---|---|
| Ctrl+C / Ctrl+V | Copy / paste SMILES |
| Ctrl+Z | Undo |
| Ctrl+A | Select all |
| Ctrl+X / Delete | Delete selection |
| Ctrl+D | Duplicate selection |
| Ctrl+M | Energy-minimise selection |
| Ctrl+H | Cycle stereo display |
| Ctrl+G | Atom letters black / coloured |
| Ctrl+B | Background black / white |
| Ctrl+S / Ctrl+O | Save / open `.pad` file |
| Ctrl+E | Export 3D SDF |
| Ctrl+Q | Save 2D image |
| Ctrl+P | Screenshot |
| Ctrl+W | Show all shortcuts |

## Build a Windows executable

```
pyinstaller --name ChemPad --icon iconx.ico --noconsole --onedir ChemPad.py
```

The window icon is loaded from `icony.ico` next to the executable, so copy it into `dist/ChemPad/`.

## Contributing

Issues and pull requests are welcome.
