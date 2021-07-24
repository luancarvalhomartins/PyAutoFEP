# PyAutoFEP
**PyAutoFEP: an automated FEP workflow for GROMACS integrating enhanced sampling methods**

PyAutoFEP is a tool to automate Free Energy Perturbations (FEP) calculations to estimate Relative Free Energies of Binding (RFEB) of small molecules to 
macromolecular targets. It automates the generation of perturbation maps, the building of dual-topologies for ligand pairs, the setup of MD systems, and 
the analysis. Distinctivelly, PyAutoFEP supports multiple force fields, integrates enhanced sampling methods, and allows flexible λ windows schemes. 
Furthermore, it aims to be as flexible as possible, giving the user great control over all steps. Nevertheless, reasonable defaults and automation are 
provided, so that PyAutoFEP can be used by non experts. PyAutoFEP is written in Python3 and uses GROMACS.

## Requirements
- Common GNU programs: Bash, awk, tar
- [GROMACS](https://www.gromacs.org/) 2016 or newer
- Python 3.6+
- [rdkit](https://www.rdkit.org/) 2019.03+
- [networkx](https://networkx.org) 2.X (1.X versions are not supported)
- [alchemlyb](https://github.com/alchemistry/alchemlyb) 0.3 & [pymbar](https://github.com/choderalab/pymbar) 3.0.4
- [openbabel](http://openbabel.org/wiki/Main_Page) 2.4 (sparsely used, mainly to load receptor files in *prepare_perturbation_map.py*. openbabel 3.X is not currently not supported, but eventually will)
- matplotlib (required only in *analyze_results.py*, optional in *generate_perturbation_map.py*)
- numpy (required only in *analyze_results.py*)

Optional requirements. The following are not required to run basic calculations in PyAutoFEP, but are needed for specific functions.

- [biopython](https://biopython.org/) (allows sequence alignment when reading initial pose data)
- [mdanalysis](https://www.mdanalysis.org/) (allows use of atom selection language in some contexts)
- pytest (required to run Python tests)

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
pip install pymbar alchemlyb
```

## Documentation
[PyAutoFEP manual](https://github.com/luancarvalhomartins/PyAutoFEP/blob/master/docs/Manual.pdf) describes in detail its functions and options.

A tutorial using the Farnesoid X receptor and a series of rigid binders is available. We plan to add more tutorials, covering specific aspects of PyAutoFEP using in the near future.

## Issues & pull requests
Issues and pull requests are welcome. When filling a GitHub issue, please include as much details as possible. Inputs and verbose outputs are also useful (if available/relevant). And thanks for reporting bugs!

## Citation
If PyAutoFEP is used in scientific publications, please cite:

* LC Martins, EA Cino, RS Ferreira. PyAutoFEP: An Automated Free Energy Perturbation Workflow for GROMACS Integrating Enhanced Sampling Methods. **Journal of Chemical Theory and Computation**. _In press_. 2021. [LINK](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00194)

## Legal notice
Copyright © 2021  Luan Carvalho Martins <luancarvalhomartins@gmail.com>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not, see [https://www.gnu.org/licenses](https://www.gnu.org/licenses)
