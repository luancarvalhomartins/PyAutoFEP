#! /usr/bin/env python3
#
#  vina_loader.py
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

import all_classes
import os_util
import mol_util
from docking_readers.generic_loader import extract_docking_poses


def extract_vina_poses(ligands_dict, poses_data=None, no_checks=False, verbosity=0):
    """ Extracts docking poses from Vina/QVina2 output pdbqt. Currently, the log file itself is not used.

    Parameters
    ----------
    ligands_dict : dict
        Input file names from where poses should be read, in {lig_name: filename} format
    poses_data : dict or str or None
        If provided, can be used to select clusters from the docking results. The first cluster will be used by default.
        Passing a {lig_name: filename} dict or the path to a key: val can be used to select other clusters
    no_checks : bool
        Ignore checks and try to go on
    verbosity : int
        Sets the verbosity level

    Returns
    -------
    dict
        Dictionary indexed by molecule names, containing the read poses, as pybel.Molecule
    """

    try:
        from openbabel import pybel
    except ImportError:
        import pybel

    if verbosity < os_util.verbosity_level.extra_debug:
        pybel.ob.obErrorLog.SetOutputLevel(pybel.ob.obError)
    else:
        os_util.local_print('OpenBabel warning messages are on, expect a lot of output.',
                            msg_verbosity=os_util.verbosity_level.extra_debug, current_verbosity=verbosity)

    docking_poses_data = os_util.parse_simple_config_file(poses_data, verbosity=verbosity)

    for each_name, each_mol in ligands_dict.items():
        # Use OpenBabel to read input pdbqt file
        try:
            # FIXME: use mol_util.read_small_molecule_from_pdbqt here
            this_lig_data = [p for p in pybel.readfile('pdbqt', each_mol)]
            assert this_lig_data
        except (OSError, IOError) as error_data:
            if no_checks:
                os_util.local_print('Failed to read Vina/QVina2 input file {} when parsing docking data for {}. '
                                    'Because you are running with no_checks, I am trying to go on.'
                                    ''.format(each_name, each_name),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                continue
            else:
                os_util.local_print('Failed to read Vina/QVina2 input file {} when parsing docking data for {}.'
                                    ''.format(each_name, each_name),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise error_data
        except AssertionError:
            if no_checks:
                os_util.local_print('Failed to read Vina/QVina2 input file {} when parsing docking data for {}. '
                                    'No pose was found in file. Because you are running with no_checks, I am trying to '
                                    'go on.'
                                    ''.format(each_name, each_name),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                continue
            else:
                os_util.local_print('Failed to read Vina/QVina2 input file {} when parsing docking data for {}. No '
                                    'pose was found in file.'.format(each_name, each_name),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
        else:
            cluster_num = docking_poses_data.get(each_name, 1)
            try:
                this_lig_pose = this_lig_data[cluster_num - 1]
            except IndexError as error_data:
                if no_checks:
                    os_util.local_print('Failed to read Vina/QVina2 input file {} when parsing docking data for {}. '
                                        'Cluster {} not found. Because you are running with no_checks, I am trying to '
                                        'go on.'
                                        ''.format(each_name, each_name, cluster_num),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    continue
                else:
                    os_util.local_print('Failed to read Vina/QVina2 input file {} when parsing docking data for {}. '
                                        'Cluster {} not found.'
                                        ''.format(each_name, each_name, cluster_num),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise error_data
            else:
                original_file_path = ligands_dict[each_name]
                ligands_dict[each_name] = all_classes.Namespace()
                ligands_dict[each_name].filename = original_file_path
                ligands_dict[each_name].format = 'pdbqt'
                ligands_dict[each_name].data = mol_util.obmol_to_rwmol(this_lig_pose, verbosity=verbosity)
                ligands_dict[each_name].comment = 'cluster {}'.format(cluster_num)

    return extract_docking_poses(ligands_dict, verbosity=verbosity)
