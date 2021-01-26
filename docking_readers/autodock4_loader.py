#! /usr/bin/env python3
#
#  autodock4_loader.py
#
#  Copyright 2018 Luan Carvalho Martins <luancarvalho@ufmg.br>
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

import os
import sys
import subprocess
import all_classes
import os_util
from docking_readers.generic_loader import extract_docking_poses


def extract_autodock4_poses(ligands_dict, poses_data=None, no_checks=False, verbosity=0):
    """
    :param dict ligands_dict: dictionary containing ligands data
    :param str poses_data: file with poses to be used
    :param bool no_checks: ignore checks and tries to go on
    :param int verbosity: be verbosity
    :rtype: dict
    """

    awk_extract_poses = """
    BEGIN {{
        Found = 0
        FoundM = 0 
    }}
    $0 == "\tLOWEST ENERGY DOCKED CONFORMATION from EACH CLUSTER" {{
        Found = 1
    }}
    Found == 1 && $1 == "MODEL" {{
        FoundM+=1
        if (FoundM > {0}) {{
            exit
        }}
    }}
    FoundM == {0} {{
        print $0
        if ($0 == "ENDMDL") {{
            exit
        }}
    }}
    """

    #FIXME: fix this method

    if isinstance(poses_data, str):
        raw_data = os_util.read_file_to_buffer(poses_data, die_on_error=True, return_as_list=True,
                                               error_message='Failed to read poses data file.', verbosity=verbosity)
        docking_poses_data = {}
        for each_line in raw_data:
            if (len(each_line) <= 1) or (each_line[0] in [';', '#']):
                continue
            lig_data = each_line.split('=')
            try:
                docking_poses_data[lig_data[0].rstrip()] = int(lig_data[1])
            except (ValueError, IndexError) as error_data:
                os_util.local_print('Could not read line "{}" from file {} with error {}'
                                    ''.format(each_line, poses_data, error_data),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
    elif isinstance(poses_data, dict):
        docking_poses_data = poses_data
    else:
        docking_poses_data = {}

    os_util.local_print('{:=^50}\n{:<15} {:<15} {:<15}'.format(' Autodock4 poses ', 'Name', 'File', 'Cluster #'),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    for each_name, each_mol in ligands_dict.items():
        # Extract cluster data and reads it
        cluster_num = docking_poses_data.get(each_name, 1)

        try:
            docking_cluster_pdb = subprocess.check_output(['awk', awk_extract_poses.format(cluster_num), each_mol])
        except subprocess.CalledProcessError as error_data:
            os_util.local_print('Could not run external program. Error: {}'.format(error_data),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            if not no_checks:
                raise SystemExit(1)
            else:
                os_util.local_print('{:<15} {:<18} {:<15} ERROR!!!'
                                    ''.format(each_name, each_mol, cluster_num),
                                    msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)
                continue
        else:
            mol_awk_result = docking_cluster_pdb.decode(sys.stdout.encoding)
            if len(mol_awk_result) < 3:
                os_util.local_print('Failed to read cluster {} from file {}.'
                                    ''.format(each_mol, cluster_num),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                if not no_checks:
                    raise SystemExit(1)
                else:
                    os_util.local_print('{:<15} {:<18} {:<15} ERROR!!!'
                                        ''.format(each_name, each_mol, cluster_num),
                                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)
                    continue

            original_file_path = ligands_dict[each_name]
            ligands_dict[each_name] = all_classes.Namespace()
            ligands_dict[each_name].format = 'pdbqt'
            ligands_dict[each_name].data = mol_awk_result
            ligands_dict[each_name].comment = '{} cluster {}'.format(original_file_path, cluster_num)

    return extract_docking_poses(ligands_dict, verbosity=verbosity)


def extract_docking_receptor(receptor_file, verbosity=0):
    """ Reads a docking receptor file

    :param str receptor_file: receptor file
    :param int verbosity: be verbosity
    :rtype: pybel.OBMol
    """

    import pybel
    if verbosity <= 3:
        pybel.ob.obErrorLog.SetOutputLevel(pybel.ob.obError)

    receptor_format = os.path.splitext(receptor_file)[1].lstrip('.')
    if receptor_format == 'pdbqt':
        receptor_format = 'pdb'

    os_util.local_print('Reading receptor data from {} as a {} file'.format(receptor_file, receptor_format),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    try:
        receptor_mol_local = pybel.readfile(receptor_format, receptor_file).__next__()
    except ValueError as error_data:
        os_util.local_print('Could not understand format {} (guessed from extension). Error was: {}'
                            ''.format(receptor_format, error_data),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise SystemExit(1)
    except (IOError, StopIteration) as error_data:
        os_util.local_print('Could not read file {} using format {} (guessed from extension). Error was: {}'
                            ''.format(receptor_file, receptor_format, error_data),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise SystemExit(1)
    else:
        return receptor_mol_local
