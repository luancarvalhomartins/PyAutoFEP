#!env python3
#
#  test_detect_solute_molecule_name.py
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

from all_classes import TopologyData

base_path = 'test_data/detect_solute_molecule_name'


def test_detect_solute_molecule_name_pdb(gmx_bin='gmx'):
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/complex_SOL.pdb',
                                                    gmx_bin=gmx_bin) == 'SOL'
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/complex_BLA.pdb',
                                                    gmx_bin=gmx_bin, test_sol_molecules='BLA') == 'BLA'
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/memb_prot_TIP3.pdb',
                                                    gmx_bin=gmx_bin) == 'TIP3'


def test_detect_solute_molecule_name_top(gmx_bin='gmx'):
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/top_SOL_1chain.top',
                                                    gmx_bin=gmx_bin) == 'SOL'
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/top_HOH_1chain.top',
                                                    gmx_bin=gmx_bin) == 'HOH'
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/top_HOH_import.top',
                                                    gmx_bin=gmx_bin) == 'WAT'
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/top_WAT_SOL_TIP3_import.top',
                                                    gmx_bin=gmx_bin, test_sol_molecules='WAT') == 'WAT'
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/top_WAT_SOL_TIP3_import.top',
                                                    gmx_bin=gmx_bin, test_sol_molecules='TIP3') == 'TIP3'


def test_detect_solute_molecule_name_tpr(gmx_bin='gmx'):
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/water_leg_TIP3.tpr',
                                                    gmx_bin=gmx_bin, test_sol_molecules='TIP3') == 'TIP3'
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/water_leg_TIP3.tpr',
                                                    gmx_bin=gmx_bin, test_sol_molecules='CLA') == 'CLA'


def test_detect_solute_molecule_name_ndx(gmx_bin='gmx'):
    assert TopologyData.detect_solute_molecule_name(input_file=f'{base_path}/index_TIP3.ndx',
                                                    gmx_bin=gmx_bin) == 'TIP3'
    test_detect_solute_molecule_name_pdb(gmx_bin=gmx_bin)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tests prepare_dual_topology.TopologyData.detect_solute_molecule_name')
    parser.add_argument('tests', type=str, nargs='*', default=['pdb', 'top', 'tpr', 'ndx'],
                        help="Run these tests (options: pdb, top, tpr, ndx; default: (run all tests)")
    parser.add_argument('--gmx_bin', type=str, default='gmx', help="Use this GROMACS executable; default: gmx")
    arguments = parser.parse_args()

    for each_test in arguments.tests:
        if each_test == 'pdb':
            test_detect_solute_molecule_name_pdb(arguments.gmx_bin)
        elif each_test == 'top':
            test_detect_solute_molecule_name_top(arguments.gmx_bin)
        elif each_test == 'tpr':
            test_detect_solute_molecule_name_tpr(arguments.gmx_bin)
        elif each_test == 'ndx':
            test_detect_solute_molecule_name_ndx(arguments.gmx_bin)
        else:
            raise ValueError('Unknown test {}'.format(each_test))
