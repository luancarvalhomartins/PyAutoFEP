#! /usr/bin/env python3
#
#  pdb_loader.py
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

import pybel
import rdkit.Chem.PropertyMol
from time import strftime

import os_util
from align_utils import align_protein
from mol_util import obmol_to_rwmol


def extract_pdb_poses(poses_data, reference_structure, ligand_residue_name='LIG', save_state=None, verbosity=0,
                      **kwargs):
    """
    :param dict poses_data: dict with the files bearing the poses and the receptor, potentially in a different
                            orientation and conformation
    :param pybel.Molecule reference_structure:
    :param str ligand_residue_name: the residues name of the ligand
    :param int verbosity: sets verbosity level
    :rtype: dict
    """

    os_util.local_print('{:=^50}\n{:<15} {:<20}'.format(' Poses read ', 'Name', 'File'),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    # Test for data from a previous run
    saved_pose_data = {}
    if save_state:
        if 'pdbpose_data' in save_state:
            try:
                saved_reference_structure = save_state['pdbpose_data']['reference_pose_path']
            except KeyError:
                # Incorrect behavior, there is no reference_pose_path, so we cannot trust in save_state data at all
                os_util.local_print('Unexpected data strucuture in {}. The entry for PDB pose data is corrupted.'
                                    ' Trying to fix and going on.'
                                    ''.format(save_state.data_file),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
            else:
                if (reference_structure.data['file_path'] and
                        saved_reference_structure == reference_structure.data['file_path']):
                    # Reference pose is the same, we can use the data
                    if len(save_state['pdbpose_data']['ligand_dict']) == 0:
                        os_util.local_print('No ligand poses were saved from previous run in file {}. I found a entry '
                                            'for pdb pose data, but it is empty.'
                                            ''.format(save_state.data_file),
                                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                    else:
                        os_util.local_print('Reading poses data from {}.'.format(save_state.data_file),
                                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                        saved_pose_data = save_state['pdbpose_data']['ligand_dict']
                else:
                    os_util.local_print('PDB poses data from {} was created for reference file {}, while this run uses '
                                        '{} as reference file. Cannot use saved data.'
                                        ''.format(save_state.data_file, saved_reference_structure,
                                                  reference_structure.data['file_path']),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    for k, v in {'seq_align_mat': 'BLOSUM80', 'gap_penalty': -1}.items():
        kwargs.setdefault(k, v)

    docking_mol_local = {}
    # Iterate over the dict, reading the poses
    for ligand_name, ligand_dict in poses_data.items():
        # Try to load the ligand data from saved state
        try:
            docking_mol_local[ligand_name] = saved_pose_data[ligand_name]
        except KeyError:
            pass
        else:
            os_util.local_print('Readed {} pose from {}'.format(ligand_name, save_state.data_file),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
            continue

        receptor_format = ligand_dict.split('.')[-1]
        if receptor_format == 'pdbqt':
            receptor_format = 'pdb'

        # pdb and mol2 fills OBResidue, does any other format file do? If so, we have to add it to this list
        if receptor_format not in ['pdb', 'mol2']:
            os_util.local_print('Using pdb_loader requires a pdb or a mol2 file, but you supplied {}. Try using '
                                'generic_loader or converting your input files',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        try:
            this_pdb_data = pybel.readfile(receptor_format, ligand_dict).__next__()
        except IOError as error_data:
            os_util.local_print('Could not read {}. Error: {}'.format(ligand_dict, error_data),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        else:
            # Iterates over all residues looking for ligand_name. Note: this will select the first residue named
            # ligand_name.
            lig_residue = None
            for each_res in pybel.ob.OBResidueIter(this_pdb_data.OBMol):
                if each_res.GetName() == ligand_residue_name:
                    lig_residue = each_res
                    break
            else:
                # For was not break, we did not find ligand_name
                os_util.local_print('Could not find ligand molecule {} in file {} using the residue name {}. I have '
                                    'read the following residues: {}\n'
                                    ''.format(ligand_name, ligand_dict, lig_residue,
                                              ', '.join([this_pdb_data.OBMol.GetResidue(i).GetName()
                                                         for i in range(this_pdb_data.OBMol.NumResidues())])),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
            dellist = [each_atom.GetIdx() for each_atom in pybel.ob.OBMolAtomIter(this_pdb_data.OBMol)
                       if each_atom.GetIdx() not in [atom_in_res.GetIdx() for atom_in_res in
                                                     pybel.ob.OBResidueAtomIter(lig_residue)]]

            ligand_ob_molecule = pybel.ob.OBMol(this_pdb_data.OBMol)
            [ligand_ob_molecule.DeleteAtom(ligand_ob_molecule.GetAtom(a)) for a in reversed(dellist)]

            docking_mol_local[ligand_name] = ligand_ob_molecule
            align_data = align_protein(this_pdb_data, reference_structure,
                                       seq_align_mat=kwargs['seq_align_mat'],
                                       gap_penalty=kwargs['gap_penalty'], verbosity=verbosity)

            docking_mol_local[ligand_name].Translate(align_data['centering_vector'])
            docking_mol_local[ligand_name].Rotate(pybel.ob.double_array(align_data['rotation_matrix']))
            docking_mol_local[ligand_name].Translate(align_data['translation_vector'])

            os_util.local_print('{:<15} {:<20}'.format(ligand_name, ligand_dict),
                                msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

        if save_state:
            save_dict = {'reference_pose_path': reference_structure.data['file_path'],
                         'ligand_dict': {k: rdkit.Chem.PropertyMol.PropertyMol(obmol_to_rwmol(v))
                                         for k, v in docking_mol_local.items()}}
            save_state['pdbpose_data'] = save_dict
            save_state['pdbpose_data_{}'.format(strftime('%d%m%Y_%H%M%S'))] = save_dict.copy()
            save_state.save_data()

    return docking_mol_local
