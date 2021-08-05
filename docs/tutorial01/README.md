## PyAutoFEP tutorial 01

This tutorial covers the basics of PyAutoFEP: generating a perturbation map, obtaining ligand topologies and the receptor structure, preparing the perturbation 
inputs, running the MD and analysing the outputs. The Farnesoid X receptor and some of ligands of the D3R's Grand Challenge 2 
(https://drugdesigndata.org/about/grand-challenge-2/fxr) are used as a model system. The OPLS-AA/M force field (http://zarbi.chem.yale.edu/oplsaam.html) is used 
to model the protein and small molecule parameters are generated using LigParGen (http://zarbi.chem.yale.edu/ligpargen). 

### Tutorial requirements
* PyAutoFEP
* GROMACS (version > 2016)
* openbabel (version 2.4.x)
* Tutorial data (workdir.tgz)
