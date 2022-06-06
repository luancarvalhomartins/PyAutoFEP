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

    # TODO: replace awk with internal parsing of Autodock4 output
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

    docking_poses_data = os_util.parse_simple_config_file(poses_data, verbosity=verbosity)

    for each_name, each_mol in ligands_dict.items():
        # Extract cluster data and reads it
        cluster_num = docking_poses_data.get(each_name, 1)

        try:
            docking_cluster_pdb = subprocess.check_output(['awk', awk_extract_poses.format(cluster_num), each_mol])
        except (subprocess.CalledProcessError, FileNotFoundError) as error_data:
            os_util.local_print('Could not run external program "awk" when extracting docking data from file {}. '
                                'Error was:\n{}'.format(error_data, each_mol),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            if not no_checks:
                raise error_data
            else:
                continue
        else:
            mol_awk_result = docking_cluster_pdb.decode(sys.stdout.encoding)
            if len(mol_awk_result) < 3:
                os_util.local_print('Failed to read cluster {} from file {}.'
                                    ''.format(each_mol, cluster_num),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                if no_checks:
                    continue
                else:
                    raise IOError("Failed to read data from file {}".format(each_name))

            original_file_path = ligands_dict[each_name]
            ligands_dict[each_name] = all_classes.Namespace()
            ligands_dict[each_name].filename = original_file_path
            ligands_dict[each_name].format = 'pdb'
            ligands_dict[each_name].data = mol_awk_result
            ligands_dict[each_name].comment = 'cluster {}'.format(cluster_num)

    return extract_docking_poses(ligands_dict, verbosity=verbosity)


def extract_docking_receptor(receptor_file, verbosity=0):
    """ Reads a docking receptor file

    :param str receptor_file: receptor file
    :param int verbosity: be verbosity
    :rtype: pybel.OBMol
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
