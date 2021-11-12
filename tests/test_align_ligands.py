#! /usr/bin/env python3
#
#  test_align_ligands.py
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

# NOTE: changing this import order caused segfaults on rdkit.Chem.Get3DDistanceMatrix on conda 4.6.1, rdkit 2019.09.2,
# openbabel 2.4.1
import rdkit.Chem
import numpy
import pybel
import pickle

from prepare_dual_topology import align_ligands

dist_tolerance = 0.001


def extract_coordinates(each_mol):
    coord_array = numpy.array([[each_mol.GetConformer().GetAtomPosition(a.GetIdx()).x,
                                each_mol.GetConformer().GetAtomPosition(a.GetIdx()).y,
                                each_mol.GetConformer().GetAtomPosition(a.GetIdx()).z]
                               for a in each_mol.GetAtoms()])
    return coord_array


def test_align_ligands_pdb1():
    ligands_dict = {'FXR_10': 'test_data/align_ligands_pdb/FXR_10.pdb',
                    'FXR_12': 'test_data/align_ligands_pdb/FXR_12.pdb'}

    align_data = align_ligands(pybel.readfile('pdb', 'test_data/align_ligands_pdb/1dvwb.pdb').__next__(),
                               ligands_dict, poses_reference_structure='test_data/align_ligands_pdb/1dvwb.pdb',
                               pose_loader='pdb', ligand_residue_name='LIG', verbosity=-1)

    with open('test_data/align_ligands_pdb/results_pdb1.pkl', 'rb') as fh:
        reference_position = pickle.load(fh)
    for each_name, each_mol in align_data.items():
        assert (extract_coordinates(each_mol) - reference_position[each_name]).sum() < dist_tolerance


def test_align_ligands_pdb2():
    ligands_dict = {'FXR_10': 'test_data/align_ligands_pdb/FXR_10.pdb',
                    'FXR_12': 'test_data/align_ligands_pdb/FXR_12.pdb'}

    align_data = align_ligands(pybel.readfile('pdb', 'test_data/align_ligands_pdb/1dvwb_altered.pdb').__next__(),
                               ligands_dict, poses_reference_structure='test_data/align_ligands_pdb/1dvwb.pdb',
                               pose_loader='pdb', ligand_residue_name='LIG', verbosity=-1)

    with open('test_data/align_ligands_pdb/results_pdb2.pkl', 'rb') as fh:
        reference_position = pickle.load(fh)

    for each_name, each_mol in align_data.items():
        assert (extract_coordinates(each_mol) - reference_position[each_name]).sum() < dist_tolerance


def test_align_ligands_autodock4():
    from rdkit.rdBase import DisableLog
    DisableLog('rdApp.*')

    ligands_dict = {'FXR_10': 'test_data/align_ligands_autodock4/FXR_10.dlg',
                    'FXR_12': 'test_data/align_ligands_autodock4/FXR_12.dlg'}

    align_data = align_ligands(pybel.readfile('pdb', 'test_data/align_ligands_autodock4/1dvwb_altered.pdb').__next__(),
                               ligands_dict,
                               poses_reference_structure='test_data/align_ligands_autodock4/FXR_receptor.pdbqt',
                               pose_loader='autodock4', cluster_docking_data={'FXR_12': 2}, verbosity=-1)

    with open('test_data/align_ligands_autodock4/results_autodock4.pkl', 'rb') as fh:
        reference_position = pickle.load(fh)

    for each_name, each_mol in align_data.items():
        assert (extract_coordinates(each_mol) - reference_position[each_name]).sum() < dist_tolerance


def test_align_ligands_superimpose1():
    from rdkit.rdBase import DisableLog
    DisableLog('rdApp.*')

    from mol_util import adjust_query_properties
    from rdkit.Chem.rdMolAlign import GetBestRMS

    ligands_dict = {'FXR_10': 'test_data/align_ligands_superimpose/FXR_10.mol2',
                    'FXR_12': 'test_data/align_ligands_superimpose/FXR_12.mol2'}
    result_file = 'test_data/align_ligands_superimpose/FXR_10_result.pdb'

    align_data = align_ligands(pybel.readfile('pdb', 'test_data/align_ligands_pdb/1dvwb_altered.pdb').__next__(),
                               ligands_dict,
                               reference_pose_superimpose='test_data/align_ligands_superimpose/FXR_10_ref.mol2',
                               poses_reference_structure='test_data/align_ligands_pdb/1dvwb.pdb',
                               pose_loader='superimpose', verbosity=-1)

    resultmol = adjust_query_properties(rdkit.Chem.MolFromPDBFile(result_file))
    assert (GetBestRMS(resultmol, align_data['FXR_10'])) < 0.1
    fxr12rms = GetBestRMS(resultmol, align_data['FXR_12'], map=[[(30, 0), (11, 1), (23, 2), (20, 3), (26, 35), (32, 34),
                                                                 (28, 6), (12, 7), (1, 8), (2, 9), (16, 10), (3, 11),
                                                                 (8, 12), (18, 13), (7, 14), (19, 15), (9, 16), (0, 17),
                                                                 (14, 18), (13, 19), (4, 27), (17, 28), (27, 29),
                                                                 (33, 30), (29, 31), (15, 32), (5, 33), (31, 5),
                                                                 (25, 4)]])
    assert fxr12rms < 1


def test_align_ligands_superimpose2():
    from rdkit.rdBase import DisableLog
    DisableLog('rdApp.*')

    from mol_util import adjust_query_properties
    from rdkit.Chem.rdMolAlign import GetBestRMS

    ligands_dict = {'FXR_10': 'test_data/align_ligands_superimpose/FXR_10.mol2',
                    'FXR_12': 'test_data/align_ligands_superimpose/FXR_12.mol2'}
    result_file = 'test_data/align_ligands_superimpose/FXR_10_result.pdb'

    align_data = align_ligands(pybel.readfile('pdb', 'test_data/align_ligands_pdb/1dvwb_altered.pdb').__next__(),
                               ligands_dict,
                               reference_pose_superimpose='test_data/align_ligands_superimpose/FXR_10_ref.pdb',
                               poses_reference_structure='test_data/align_ligands_pdb/1dvwb.pdb',
                               pose_loader='superimpose', verbosity=-1)
    resultmol = adjust_query_properties(rdkit.Chem.MolFromPDBFile(result_file))
    rdkit.Chem.MolToPDBFile(align_data['FXR_10'], 'error_pdb_fxr10.pdb')
    assert (GetBestRMS(resultmol, align_data['FXR_10'])) < 0.1
    fxr12rms = GetBestRMS(resultmol, align_data['FXR_12'], map=[[(30, 0), (11, 1), (23, 2), (20, 3), (26, 35), (32, 34),
                                                                 (28, 6), (12, 7), (1, 8), (2, 9), (16, 10), (3, 11),
                                                                 (8, 12), (18, 13), (7, 14), (19, 15), (9, 16), (0, 17),
                                                                 (14, 18), (13, 19), (4, 27), (17, 28), (27, 29),
                                                                 (33, 30), (29, 31), (15, 32), (5, 33), (31, 5),
                                                                 (25, 4)]])
    assert fxr12rms < 1


def test_align_ligands_generic():
    ligands_dict = {'FXR_10': 'test_data/align_ligands_generic/FXR_10.pdb',
                    'FXR_12': 'test_data/align_ligands_generic/FXR_12.pdb'}

    align_data = align_ligands(pybel.readfile('pdb', 'test_data/align_ligands_generic/1dvwb_altered.pdb').__next__(),
                               ligands_dict, poses_reference_structure='test_data/align_ligands_generic/1dvwb.pdb',
                               pose_loader='generic', verbosity=-1)

    with open('test_data/align_ligands_generic/result_generic.pkl', 'rb') as fh:
        reference_position = pickle.load(fh)

    for each_name, each_mol in align_data.items():
        assert (extract_coordinates(each_mol) - reference_position[each_name]).sum() < dist_tolerance


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tests prepare_dual_topology.align_ligands')
    parser.add_argument('tests', type=str, nargs='*', default=['pdb', 'autodock4', 'superimpose', 'generic'],
                        help="Run these tests (options: pdb, autodock4, superimpose, generic; default: (run all tests)")
    arguments = parser.parse_args()

    for each_test in arguments.tests:
        if each_test == 'pdb':
            test_align_ligands_pdb1()
            test_align_ligands_pdb2()
        elif each_test == 'autodock4':
            test_align_ligands_autodock4()
        elif each_test == 'superimpose':
            test_align_ligands_superimpose1()
            test_align_ligands_superimpose2()
        elif each_test == 'generic':
            test_align_ligands_generic()
        else:
            raise ValueError('Unknown test {}'.format(each_test))
