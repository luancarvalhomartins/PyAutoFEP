#! /usr/bin/env python3
#
#  test_find_mcs.py
#
#  Copyright 2019 Luan Carvalho Martins <luancarvalho@ufmg.br>
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

import rdkit.Chem
from merge_topologies import find_mcs, find_mcs_3d


def test_find_mcs_3d_carbohydrate():
    gal = rdkit.Chem.MolFromMol2File('test_data/find_mcs_3d/glucose.mol2', removeHs=False)
    glu = rdkit.Chem.MolFromMol2File('test_data/find_mcs_3d/galactose.mol2', removeHs=False)
    mcs = find_mcs_3d(glu, gal, num_conformers=200, tolerance=0.5, verbosity=-1).smartsString
    assert mcs == '[H]OC([H])([H])[C@]1([H])CC[C@@]([H])(O[H])[C@@]([H])(O[H])O1'


def test_find_mcs_3d_macrocycle():
    azi = rdkit.Chem.MolFromMol2File('test_data/find_mcs_3d/azithromycin.mol2', removeHs=False)
    azi_dia1 = rdkit.Chem.MolFromMol2File('test_data/find_mcs_3d/azithromycin_diastero1.mol2', removeHs=False)
    mcs = find_mcs_3d(azi, azi_dia1, num_conformers=200, tolerance=0.5, verbosity=3).smartsString
    assert mcs == '[H]OC([H])([H])[C@]1([H])CC[C@@]([H])(O[H])[C@@]([H])(O[H])O1'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tests find_mcs and find_mcs_3d')
    parser.add_argument('tests', type=str, nargs='*',
                        default=['find_mcs_3d_carbohydrate', 'find_mcs_3d_macrocycle'],
                        help="Run these tests (options: find_mcs_3d_carbohydrate, find_mcs_3d_macrocycle; "
                             "default: (run all tests)")
    arguments = parser.parse_args()

    for each_test in arguments.tests:
        if each_test == 'find_mcs_3d_carbohydrate':
            test_find_mcs_3d_carbohydrate()
        elif each_test == 'find_mcs_3d_macrocycle':
            test_find_mcs_3d_macrocycle()
        else:
            raise ValueError('Unknown test {}'.format(each_test))
