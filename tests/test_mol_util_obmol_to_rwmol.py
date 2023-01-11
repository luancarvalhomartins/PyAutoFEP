#! /usr/bin/env python3
#
#  test_mol_util_obmol_to_rwmol.py
#
#  Copyright 2023 Luan Carvalho Martins <luancarvalho@ufmg.br>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

from mol_util import obmol_to_rwmol
import rdkit.Chem
from openbabel import pybel


def test_obmol_to_rwmol_smiles():
    """ Test obmol_to_rwmol conversion of molecules build from SMILES. In this case, there is no 3D structure to assign
    stereochemistry from, so that this tests the stereochemistry conversion code.
    """

    mols = [r'[H]/C(F)=C(/[H])C([H])([H])C([H])([H])Br',  # Explicit Hs. Note that a cis bond may be represented
            # with both // or \\ and the molecule would be the same
            'F/C=C/C[C@@H](Cl)Br',  # Implict Hs, chiral center & double
            'F/C=C/C[C@H](Cl)Br',  # Implict Hs, chiral center & double
            'CC=CC',  # Undefined double bond stereochemistry
            'CC(Cl)Br',  # Undefined tetrahedral stereochemistry, implicit H
            '[H]C(C)(Cl)Br',  # Undefined tetrahedral stereochemistry, explicit H
            'C/C=C/C',  # Double bond cis
            r'C/C=C\C',  # Double bond trans
            ]
    for m in mols:
        assert m == rdkit.Chem.MolToSmiles(obmol_to_rwmol(pybel.readstring('smi', m)))


def test_obmol_to_rwmol_opensmiles():
    """ Test obmol_to_rwmol conversion of molecules build from SMILES. In this case, there is no 3D structure to assign
    stereochemistry from, so that this tests the stereochemistry conversion code.
    """

    mols = ['CC(F)(F)F',
            'C1(C2=CC=CC=C2)=CC=CC=C1',
            'C1(CC=C2)=C2C=CC=C1',
            '[NH]1CCCC1',
            'CC#CC',
            'CCC(CC)CO',
            'CC=C=C(C)C',
            'C/N=N/C',
            'CC(N(C)C)=O',
            'C/C(C)=N/C',
            'C/C(N(C)C)=N/C',
            'CC(=O)OC(=O)C',
            'C(=O)Br',
            'C(=O)Cl',
            'C(=O)F',
            'C(=O)I',
            'CC=O',
            'C(=O)N',
            '*N',
            'C12=CC=CC=C1C=C3C(C=CC=C3)=C2',
            'C([N-][N+]#N)',
            'C1=CC=CC=C1',
            'C1=CC=C(C=C1)S',
            'C1CCCCC1C1CCCCC1',
            'Br',
            'CCC=C',
            'CCC#C',
            'O=C=O',
            'C(=O)O',
            'Cl',
            'COCCl',
            'C1=CC=C1',
            'C1CCC1',
            'C1CCCCCC1',
            'C1CCCCC1',
            'C1=CCCC=C1',
            'C1=CCC=CC1',
            'C=1CCCCC=1',
            'C1CCCC1',
            'C1=CCC=C1',
            'C1CC1',
            'C1=CC1',
            '[2H][CH2]C',
            'COC',
            'CCOCC',
            'CC(C)OC(C)C',
            'C&1&1&1&1',
            'C=[N+]=[N-]',
            '[NH4+].[NH4+].[O-]S(=O)(=O)[S-]',
            'N',
            'CC',
            'CCS',
            'CCO',
            'C=C',
            'COC',
            'C(=O)OC',
            'F',
            'C=O',
            'C1OC=CC=1',
            'C&1&1&1',
            'C#N',
            '[OH-]',
            'NO',
            'C1=CC=CC(CCC2)=C12',
            'CC(=O)C',
            'C',
            'CS',
            'CC(OC)=O',
            'CN1CCCC1',
            'CC(C)(C)OC',
            'C12=CC=CC=C1C=CC=C2',
            '[N+](=O)[O-]',
            'C[N+]([O-])=O',
            'C12=CC=CC1=CC=C2',
            'N1CC2CCCC2CC1',
            'OC1CCCCC1',
            'C=1(C=CC=CC1)',
            'c1ccccc1C&1&1',
            'O',
            'N',
            'CC(C)=O',
            'CCC=O',
            'CC=C',
            'CC#C',
            'N1CCCCC1',
            'O=N1CCCCC1',
            'NC',
            'C12(CCCCC1)CCCCC2',
            'S(=O)(=O)',
            'C[N+](C)(C)C',
            'S',
            'OS(=O)(=S)O',
            'CN(C)C',
            'C1(C=CC=C2)=C2C(C=CC=C3)=C3C4=C1C=CC=C4']
    failed = []
    for m in mols:
        try:
            if rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(m)) != \
                    rdkit.Chem.MolToSmiles(obmol_to_rwmol(pybel.readstring('smi', m))):
                failed.append(m)
        except:
            pass
    print(f'Passed: {(1-len(failed)/len(mols))*100:.2f}%')
    print('Failed: {}'.format(', '.join(failed)))
    assert True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tests obmol_to_rwmol in mol_util.py')
    parser.add_argument('tests', type=str, nargs='*', default=['smiles', 'open_smiles'],
                        help="Run these tests (default: run all tests)")
    arguments = parser.parse_args()

    for each_test in arguments.tests:
        if each_test == 'smiles':
            test_obmol_to_rwmol_smiles()
        elif each_test == 'open_smiles':
            test_obmol_to_rwmol_opensmiles()
        else:
            raise ValueError('Unknown test {}'.format(each_test))
