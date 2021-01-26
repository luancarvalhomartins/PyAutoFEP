#! /usr/bin/env python3
#
#  test_prepare_dual_topology.py
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

import prepare_dual_topology
import savestate_util
import os


def test_parse_ligands_data():
    result = {'FXR_10': {'molecule': 'FXR_10.mol2', 'topology': ['FXR_10.top']},
              'FXR_12': {'molecule': 'FXR_12.mol2', 'topology': ['FXR_12.top']}}
    tests_list = ['FXR_10: FXR_10.mol2, FXR_10.top; FXR_12: FXR_12.mol2, FXR_12.top',
                  'test_data/parse_ligands_data/inputfile.txt',
                  'test_data/parse_ligands_data/files_only',
                  ['FXR_10.mol2', 'FXR_12.mol2', 'FXR_12.top', 'FXR_10.top']]
    for each_test in tests_list:
        assert prepare_dual_topology.parse_ligands_data(each_test, verbosity=-1) == result

    assert prepare_dual_topology.parse_ligands_data('test_data/parse_ligands_data/dirs', verbosity=-1) == \
           {'FXR_10': {'molecule': 'test_data/parse_ligands_data/dirs/FXR_10/FXR_10.mol2',
                       'topology': ['test_data/parse_ligands_data/dirs/FXR_10/FXR_10.top']},
            'FXR_12': {'molecule': 'test_data/parse_ligands_data/dirs/FXR_12/FXR_12.mol2',
                       'topology': ['test_data/parse_ligands_data/dirs/FXR_12/FXR_12.top']}}
    assert prepare_dual_topology.parse_ligands_data('FXR_10: FXR_10.mol2; FXR_12: FXR_12.top',
                                                    savestate_util=savestate_util.SavableState(
                                                        'test_data/parse_ligands_data/molecules_test1.pkl'),
                                                    verbosity=-1)
    assert prepare_dual_topology.parse_ligands_data('FXR_12: FXR_12.top, FXR_12.mol2',
                                                    savestate_util=savestate_util.SavableState(
                                                        'test_data/parse_ligands_data/molecules_test2.pkl'),
                                                    verbosity=-1)


def test_prepare_output_scripts_data():
    assert False


def test_make_index_internal():
    from tempfile import TemporaryDirectory
    from filecmp import cmp

    dir = TemporaryDirectory()
    newfile = os.path.join(dir.name, '3ekv_index.ndx')
    prepare_dual_topology.make_index(newfile, 'test_data/gromacs_index/3ekv.pdb')
    assert cmp(newfile, 'test_data/gromacs_index/3ekv_gmx.ndx')
    prepare_dual_topology.make_index(newfile, 'test_data/gromacs_index/3ekv.pdb',
                                     index_data={'Protein_LIG': '"Protein" | "LIG"'})
    assert cmp(newfile, 'test_data/gromacs_index/3ekv_gmx_protlig.ndx')

    dir.cleanup()


def test_make_index_mdanalysis():
    from tempfile import TemporaryDirectory
    from filecmp import cmp

    dir = TemporaryDirectory()
    newfile = os.path.join(dir.name, '3ekv_index.ndx')
    prepare_dual_topology.make_index(newfile, 'test_data/gromacs_index/3ekv.pdb', method='mdanalysis')
    assert cmp(newfile, 'test_data/gromacs_index/3ekv_mdanalysis.ndx')
    prepare_dual_topology.make_index(newfile, 'test_data/gromacs_index/3ekv.pdb',
                                     index_data={'Around_LIG': 'byres (around 3.5 resname LIG)'}, method='mdanalysis')
    assert cmp(newfile, 'test_data/gromacs_index/3ekv_mdanalysis_aroundlig.ndx')

    dir.cleanup()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tests prepare_dual_topology.align_ligands')
    parser.add_argument('tests', type=str, nargs='*',
                        default=['parse_ligands_data', 'prepare_output_scripts_data', 'make_index'],
                        help="Run these tests (options: parse_ligands_data; default: (run all tests)")
    arguments = parser.parse_args()

    for each_test in arguments.tests:
        if each_test == 'parse_ligands_data':
            test_parse_ligands_data()
        elif each_test == 'prepare_output_scripts_data':
            test_prepare_output_scripts_data()
        if each_test == 'make_index':
            test_make_index_internal()
        else:
            raise ValueError('Unknown test {}'.format(each_test))
