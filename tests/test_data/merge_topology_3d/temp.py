#! /usr/bin/env python3
#

import rdkit.Chem
import rdkit.Chem.rdFMCS
import rdkit
print(rdkit.__version__)

m1 = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles('CC(C)[C@H](C)O'))
m2 = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles('CC(C)[C@@H](C)O'))
rdkit.Chem.rdFMCS.FindMCS([m1, m2], **magicalOptions).smartsString
Out: [H]C([H])([H])C([H])(CO)C([H])([H])[H]

