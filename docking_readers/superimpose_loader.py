#! /usr/bin/env python3
#
#  superimpose_loader.py
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
import merge_topologies
from docking_readers.generic_loader import extract_docking_poses
import os_util


def superimpose_poses(ligand_data, reference_pose_mol, save_state=None, num_threads=0, num_conformers=200,
                      verbosity=0, **kwargs):
    """
    :param dict ligand_data: dict with the ligands
    :param str reference_pose_mol: file with reference pose to be used
    :param int verbosity: be verbosity
    :param savestate_utils.SavableState save_state: object with saved data
    :param int num_threads: use this much threads
    :param int num_conformers: generate this much trial conformers to find a best shape match
    :param int verbosity: sets the verbosity level
    :rtype: dict
    """

    os_util.local_print('Entering superimpose_poses(ligand_data={}, reference_pose_superimpose={}, save_state={}, '
                        'verbosity={}, kwargs={})'
                        ''.format(ligand_data, reference_pose_mol, save_state, verbosity, kwargs),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Set default to no MCS
    kwargs.setdefault('mcs', None)

    # Test for data from a previous run
    if save_state:
        rdkit_reference_pose = None
        if 'superimpose_data' in save_state:
            try:
                saved_reference_pose = save_state['superimpose_data']['reference_pose_path']
            except KeyError:
                # Incorrect behavior, there is no reference_pose_path, so we cannot trust in save_state data at all
                os_util.local_print('Unexpected data strucuture in {}. The entry for superimpose data is corrupted.'
                                    ' Trying to fix and going on.'
                                    ''.format(save_state.data_file),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
            else:
                if saved_reference_pose == reference_pose_mol:
                    # Reference pose is the same, we can use the data
                    rdkit_reference_pose = save_state['superimpose_data']['reference_pose_superimpose']
                    if len(save_state['superimpose_data']['ligand_dict']) == 0 and verbosity > 0:
                        os_util.local_print('No ligand poses were saved from previous run in file {}. I found a entry '
                                            'for superimpose data, but it is empty.'
                                            ''.format(save_state.data_file),
                                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                        rdkit_reference_pose = None

        if rdkit_reference_pose is None:
            # Create a new superimpose_data entry
            from time import strftime
            backup_name = 'superimpose_data_{}'.format(strftime('%d%m%Y_%H%M%S'))
            save_state['superimpose_data'] = {}
            save_state['superimpose_data']['reference_pose_path'] = reference_pose_mol
            rdkit_reference_pose = extract_docking_poses({'reference': {'molecule': reference_pose_mol}},
                                                         verbosity=verbosity)['reference']
            save_state['superimpose_data']['reference_pose_superimpose'] = rdkit_reference_pose
            save_state['superimpose_data']['ligand_dict'] = {}
            save_state[backup_name] = save_state['superimpose_data']

        # Save whatever we done
        save_state.save_data()
    else:
        # Not saving any data
        rdkit_reference_pose = extract_docking_poses({'reference': {'molecule': reference_pose_mol}},
                                                     verbosity=verbosity)['reference']

    # Extract data from ligands
    docking_poses_data = extract_docking_poses(ligand_data, verbosity=verbosity)
    new_docking_poses_data = {}

    os_util.local_print('{:=^50}\n{:<15} {:<25} {:<15}'
                        ''.format(' Superimposed poses ', 'Name', 'File', 'Note'),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    for ligand_name, each_ligand_mol in docking_poses_data.items():

        # If possible, load data from previous run
        if save_state:

            try:
                this_ligand = save_state['superimpose_data']['ligand_dict'][ligand_name]
            except KeyError:
                os_util.local_print('Could not find data for ligand {} in {}'
                                    ''.format(ligand_name, save_state.data_file),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
            else:
                new_docking_poses_data[ligand_name] = this_ligand
                os_util.local_print('{:<15} {:<25} {:<15}'
                                    ''.format(ligand_name, str(ligand_data[ligand_name]), 'Read from saved state'),
                                    msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)
                continue

        thismol = merge_topologies.constrained_embed_shapeselect(each_ligand_mol, rdkit_reference_pose,
                                                                 num_threads=num_threads, save_state=save_state,
                                                                 num_conformers=num_conformers, verbosity=verbosity,
                                                                 **kwargs)
        new_docking_poses_data[ligand_name] = thismol

        if save_state:
            # Save rdkit Mol
            save_state['superimpose_data']['ligand_dict'][ligand_name] = thismol
            save_state.save_data()

        os_util.local_print('{:<15} {:<15}'.format(ligand_name, str(ligand_data[ligand_name])),
                            msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    os_util.local_print('=' * 50,
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    return new_docking_poses_data
