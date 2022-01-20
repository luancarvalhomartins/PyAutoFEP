# PyAutoFEP
**PyAutoFEP: an automated FEP workflow for GROMACS integrating enhanced sampling methods**

PyAutoFEP is a tool to automate Free Energy Perturbations (FEP) calculations to estimate Relative Free Energies of Binding (RFEB) of small molecules to 
macromolecular targets. It automates the generation of perturbation maps, the building of dual-topologies for ligand pairs, the setup of MD systems, and 
the analysis. Distinctivelly, PyAutoFEP supports multiple force fields, integrates enhanced sampling methods, and allows flexible λ windows schemes. 
Furthermore, it aims to be as flexible as possible, giving the user great control over all steps. Nevertheless, reasonable defaults and automation are 
provided, so that PyAutoFEP can be used by non experts. PyAutoFEP is written in Python3 and uses GROMACS.

## Announcements
**Commit fe41f7d (19.01.2022)**<br/>
This commit introduces a bunch of new features and code changes. Even though I tested newly implemented and rewritten code, **things may break**. Please, fill 
issues should you experience any problem. Main changes:
- Modifications to the atom matching functions in PyAutoFEP were done to implement support for user-supplied atom maps.
- User-supplied atom maps are also available for the superimpose pose loader
- Selecting 3D MCS for merge_topologies.merge_topologies is now supported (3D MCS in generate_perturbation_map.py and superimpose loader coming soon)
- 3D MCS code rewritten for better support for ligand pairs with multiple atom matches

## Requirements
- Common GNU programs: Bash, awk, tar
- [GROMACS](https://www.gromacs.org/) 2016 or newer
- Python 3.6+
- [rdkit](https://www.rdkit.org/) 2019.03+
- [networkx](https://networkx.org) 2.X (1.X versions are not supported)
- [alchemlyb](https://github.com/alchemistry/alchemlyb) 0.6.0 & [pymbar](https://github.com/choderalab/pymbar) 3.0.5 OR [alchemlyb](https://github.com/alchemistry/alchemlyb) 0.3.0 & [pymbar](https://github.com/choderalab/pymbar) 3.0.3 (Because of https://github.com/choderalab/pymbar/issues/419)
- [openbabel](http://openbabel.org/wiki/Main_Page) 2.4 (sparsely used, mainly to load receptor files in *prepare_perturbation_map.py*. openbabel 3.X is not currently not supported, but eventually will)
- matplotlib (required only in *analyze_results.py*, optional in *generate_perturbation_map.py*)
- numpy (required only in *analyze_results.py*)

Optional requirements. The following are not required to run basic calculations in PyAutoFEP, but are needed for specific functions.

- [biopython](https://biopython.org/) (allows sequence alignment when reading initial pose data)
- [mdanalysis](https://www.mdanalysis.org/) (allows use of atom selection language in some contexts)
- pytest (required to run Python tests)
- packaging (used to compare package versions, falling back to distutils)

## Install
To install PyAutoFEP, please, clone this repository using

```bash
git clone https://github.com/luancarvalhomartins/PyAutoFEP.git 
```

Required dependencies can be installed using Anaconda, except for pymbar and alchemlyb, which must be installed using pip.

```bash
# Create a conda environment and activate it.
conda create -n PyAutoFEP
conda activate PyAutoFEP

# Install stuff
conda install -c rdkit rdkit
conda install -c openbabel openbabel # ver 3.X is not supported, make sure to install 2.4.X ver
conda install matplotlib networkx pip

# Use pip to install pymbar and alchemlyb
pip install pymbar alchemlyb==0.6.0
```

## Documentation
### Manual
[PyAutoFEP manual](https://github.com/luancarvalhomartins/PyAutoFEP/blob/master/docs/Manual.pdf) describes in detail its functions and options.

### Tutorials
- [Farnesoid X receptor tutorial](https://github.com/luancarvalhomartins/PyAutoFEP/tree/master/docs/tutorial01) - A tutorial using the Farnesoid X receptor and a series of rigid binders is available.
 
(_We plan to add more tutorials, covering specific aspects of PyAutoFEP in the near future._)

## Issues & pull requests
Issues and pull requests are welcome. When filling a GitHub issue, please include as much details as possible. Inputs and verbose outputs are also useful (if available/relevant). And thanks for reporting bugs!

## Roadmap
Aside from bug squashing, I am currently working on charged perturbations and in a GUI.

- Charged pertubations are being implemented using the alchemical co-ion method, which seems to be both most general and simpler to code. Briefly, and random water molecule will be perturbed to an ion of the opposite charge as the ligand perturbation. At first, only charge differences of +1 and -1 will be supported (this should cover most of the use cases, anyway). Code for regular, non-charged perturbations will not be affected.
- A PyQt5 GUI for PyAutoFEP is being written. So far, a perturbation map editor was mostly implemented and a ligand table is beign worked on. The GUI will be frontend to the scripts, so that no function will depend on using the first. The GUI development is low priority right now, so this is not making into the tree anytime soon.

Further goals
- Covalent perturbations
- Automated cycle-closure histeresis (and likely other analysis as well)
- Support for peptides as ligands
- More tutorials
- A website

## Citation
If PyAutoFEP is used in scientific publications, please cite:

* LC Martins, EA Cino, RS Ferreira. PyAutoFEP: An Automated Free Energy Perturbation Workflow for GROMACS Integrating Enhanced Sampling Methods. _Journal of Chemical Theory and Computation_. **2021** _17_ (7), 4262-4273. [LINK](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00194)

## Legal notice
Copyright © 2021  Luan Carvalho Martins <luancarvalhomartins@gmail.com>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not, see [https://www.gnu.org/licenses](https://www.gnu.org/licenses)
