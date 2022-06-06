#! /usr/bin/env python3
#
#  test_read_small_molecule_from_pdbqt.py
#
#  Copyright 2022 Luan Carvalho Martins <luancarvalho@ufmg.br>
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

from mol_util import read_small_molecule_from_pdbqt


def test_read_meeko():
    from glob import glob
    filelist = glob('test_data/read_small_molecule_from_pdbqt/meeko/*.pdbqt')
    for each_file in filelist:
        m = read_small_molecule_from_pdbqt(each_file)
        assert m is not None


def test_pdbqt_smiles():
    from glob import glob
    filelist = glob('test_data/read_small_molecule_from_pdbqt/smiles/*.pdbqt')
    for each_file in filelist:
        m = read_small_molecule_from_pdbqt(each_file)
        assert m is not None


def test_pdbqt_no_smiles():
    from glob import glob
    filelist = glob('test_data/read_small_molecule_from_pdbqt/no_smiles/*.pdbqt')
    for each_file in filelist:
        m = read_small_molecule_from_pdbqt(each_file)
        assert m is not None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tests mol_util.read_small_molecule_from_pdbqt')
    parser.add_argument('tests', type=str, nargs='*',
                        default=['read_meeko', 'pdbqt_smiles', 'pdbqt_no_smiles'],
                        help="Run these tests (options: read_meeko, pdbqt_smiles, pdbqt_no_smiles; "
                             "default: (run all tests)")
    arguments = parser.parse_args()

    for each_test in arguments.tests:
        if each_test == 'read_meeko':
            test_read_meeko()
        elif each_test == 'pdbqt_smiles':
            test_pdbqt_smiles()
        elif each_test == 'pdbqt_no_smiles':
            test_pdbqt_no_smiles()
        else:
            raise ValueError('Unknown test {}'.format(each_test))
