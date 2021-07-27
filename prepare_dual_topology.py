#! /usr/bin/env python3
#
#  prepare_dual_topology.py
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

import argparse
import os
import shutil
import rdkit.Chem.rdMolAlign
import pybel
import subprocess
import tempfile
import time
import re
import tarfile
import configparser
import itertools
from collections import OrderedDict

import all_classes
import process_user_input
import savestate_util
import mol_util
import align_utils
import merge_topologies
import os_util


def guess_water_box(solvent_box, pdb2gmx_topology='', verbosity=0):
    """ Guess solvent box to be used in gmx solvate from pdb2gmx topology or from number of points of water model

    :param [int, str] solvent_box: use this solvate box in gmx solvate; if int, will use a box with a model with
                                   solvent_box points, if str, will pass to gmx solvate -cp, if None, will try to guess
                                   from pdb2gmx output (Default: None, guess).

    :param str pdb2gmx_topology: guess water model from this topology (required ir solvent_box is None)
    :param int verbosity: set verbosity
    :rtype: str
    """

    water_box_data = {'spc216.gro': ['spc.itp', 'tip3p.itp', 'scpe.itp', 'tips3p.itp'], 'tip4p.gro': ['tip4p.itp'],
                      'tip5p.grp': ['tip5p.itp', 'tip5pe.itp']}
    water_box_data_numeric = {3: 'spc216.gro', 4: 'tip4p.itp', 5: 'tip5p.itp'}

    build_base_dir = os.path.dirname(pdb2gmx_topology)
    if not pdb2gmx_topology:
        ValueError('pdb2gmx_topology required if solvent_box is None')

    if solvent_box is None:
        # 4.1 User wants automatic detection of water type. Get water topology from pdb2gmx topology file
        pdb2gmx_topoology_data = os_util.read_file_to_buffer(pdb2gmx_topology, die_on_error=True,
                                                             error_message='Failed to read topology from pdb2gmx. '
                                                                           'Cannot continue. Verify the system '
                                                                           'builder intermediate files in {}'
                                                                           ''.format(build_base_dir),
                                                             return_as_list=True, verbosity=verbosity)
        try:
            water_topology = pdb2gmx_topoology_data[pdb2gmx_topoology_data.index('; Include water topology\n') + 1]
        except ValueError:
            os_util.local_print('Failed to find a water topology in file {}. Cannot guess the correct water box to '
                                'use. Cannot continue. Please, check build system files in {} or set solvent_box.'
                                ''.format(pdb2gmx_topology, build_base_dir),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(-1)

        # 4.2 Try to detect to which box this model corresponds
        for water_box, water_models in water_box_data.items():
            for each_model in water_models:
                if water_topology.find(each_model) != -1:
                    solvent_box = water_box
                    os_util.local_print('Using water box {} because the water line in topology {} is importing water '
                                        'model {}. If this is wrong, please, set solvent_box explicitly.'
                                        ''.format(water_box, water_box_data, each_model),
                                        current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.info)
                    break
            else:
                continue
            break
        if solvent_box is None:
            os_util.local_print('Failed to parse water topology line ({}) from file {}. Cannot guess the correct water '
                                'box to use. Cannot continue. Please, check build system files in {} or set '
                                'solvent_box.'.format(water_topology.replace('\n', '\\n'),
                                                      pdb2gmx_topology, build_base_dir),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(-1)
    elif isinstance(solvent_box, int):
        # User wants a box with a solvent_box-points water
        try:
            solvent_box = water_box_data_numeric[solvent_box]
        except KeyError:
            os_util.local_print('You asked me to use a water box for water molecules with "{}" points, but I cannot do '
                                'so. I can only use the following boxes: "{}". Cannot continue. If this is what your '
                                'really want, you will need a pre-solvated system.'
                                ''.format(solvent_box, ', '.join(['{} ({} points)'.format(j, i)
                                                                  for i, j in water_box_data_numeric.items()])),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(-1)

    return solvent_box


def set_default_solvate_data(solvate_data):
    if not solvate_data:
        solvate_data = {}
    # This will use default water for the FF
    solvate_data.setdefault('box_type', 'dodecahedron')
    solvate_data.setdefault('water_model', None)
    solvate_data.setdefault('water_shell', 10.0)
    solvate_data.setdefault('nname', 'CL')
    solvate_data.setdefault('pname', 'NA')
    solvate_data.setdefault('ion_concentration', 0.15)
    return solvate_data


def prepare_complex_system(structure_file, base_dir, ligand_dualmol, topology='FullSystem.top',
                           index_file='index.ndx', forcefield=1, index_groups=None,
                           selection_method='internal', gmx_bin='gmx', extradirs=None, extrafiles=None,
                           solvate_data=None, ligand_name='LIG', gmx_maxwarn=1, no_checks=False, verbosity=0, **kwargs):
    """ Builds a system using gmx tools

    :param str structure_file: receptor file (pdb)
    :param str base_dir: basedir to build system to
    :param all_classes.MergedTopologies ligand_dualmol: merged molecule to be added to the system
    :param str topology: Gromacs-compatible topology file
    :param str index_file: Gromacs-compatible index file
    :param [int, str] forcefield: force field to be passed to pdb2gmx
    :param dict index_groups: groups to be added to index file and their selection string
    :param str selection_method: use gmx make_ndx (internal, default) or mdanalysis (mdanalysis) to generate index
    :param str gmx_bin: gmx executable to be run
    :param list extradirs: copy these dirs to base_dir
    :param list extrafiles: copy these files to base_dir
    :param dict solvate_data: dictionary containing further data to solvate: 'water_model': water model to be used,
                              'water_shell': size of the water shell in A, 'ion_concentration': add ions to this conc,
                              'pname': name of the positive ion, 'nname': name of the negative ion}
    :param str ligand_name: use this as ligand name
    :param int gmx_maxwarn: passed to gmx grompp to suppress GROMACS warnings during system setup
    :param bool no_checks: ignore checks and try to go on
    :param int verbosity: sets verbosity
    :rtype: all_classes.Namespace
    """

    os_util.local_print('Entering prepare_complex_system(structure_file={}, ligand_dualmol={}, topology={}, '
                        'index_file={}, forcefield={}, index_groups={}, selection_method={}, gmx_bin={}, extradirs={}, '
                        'extrafiles={}, solvate_data={}, verbosity={})'
                        ''.format(structure_file, ligand_dualmol, topology, index_file, forcefield, index_groups,
                                  selection_method, gmx_bin, extradirs, extrafiles, solvate_data, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    solvate_data = set_default_solvate_data(solvate_data)

    # Save intemediate files used to build the system to this dir
    build_system_dir = os.path.join(base_dir, 'protein',
                                    'build_system_{}'.format(time.strftime('%H%M%S_%d%m%Y')))
    os_util.makedir(build_system_dir)

    if extradirs:
        for each_extra_dir in extradirs:
            shutil.copytree(each_extra_dir, os.path.join(build_system_dir, os.path.split(each_extra_dir)[-1]))
            os_util.local_print('Copying directory {} to {}'.format(each_extra_dir, build_system_dir),
                                msg_verbosity=os_util.verbosity_level.debug,
                                current_verbosity=verbosity)

    if extrafiles:
        for each_source in extrafiles:
            shutil.copy2(each_source, build_system_dir)
            os_util.local_print('Copying file {} to {}'.format(each_source, build_system_dir),
                                msg_verbosity=os_util.verbosity_level.debug,
                                current_verbosity=verbosity)

    # These are the intermediate build files
    timestamp = time.strftime('%H%M%S_%d%m%Y')
    build_files_dict = {index: os.path.join(build_system_dir, filename.format(timestamp))
                        for index, filename in {'protein_pdb': 'protein_step1_{}.pdb',
                                                'protein_top': 'protein_step1_{}.top',
                                                'proteinlig_top': 'proteinlig_step1_{}.top',
                                                'system_pdb': 'protein_step2_{}.pdb',
                                                'system_gro': 'system_step3_{}.gro',
                                                'systemsolv_gro': 'systemsolvated_step4_{}.gro',
                                                'systemsolv_top': 'systemsolv_step4_{}.top',
                                                'genion_mdp': 'genion_step5_{}.mdp',
                                                'mdout_mdp': 'mdout_step6_{}.mdp',
                                                'genion_tpr': 'genion_step6_{}.tpr',
                                                'makeindex_log': 'make_index_step7_{}.log',
                                                'fullsystem_pdb': 'fullsystem_step7_{}.pdb',
                                                'fullsystem_top': 'fullsystem_step7_{}.top',
                                                'pdb2gmx_log': 'gmx_pdb2gmx_{}.log',
                                                'editconf_log': 'gmx_editconf_{}.log',
                                                'solvate_log': 'gmx_solvate_{}.log',
                                                'grompp_log': 'gmx_grompp_{}.log',
                                                'genion_log': 'gmx_genion_{}.log',
                                                'index_ndx': 'index.ndx'
                                                }.items()}

    # These are the final build files
    build_files_dict['full_topology_file'] = topology
    build_files_dict['final_index_file'] = index_file
    build_files_dict['final_structure_file'] = 'FullSystem.pdb'

    output_structure_file = build_files_dict['final_structure_file']

    os_util.local_print('These are the files to be used during automatic system building: {}'
                        ''.format(', '.join(build_files_dict.values())),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # 1. Prepare and run pdb2gmx
    pdb2gmx_list = ['pdb2gmx', '-f', structure_file,
                    '-p', build_files_dict['protein_top'], '-o', build_files_dict['protein_pdb']]

    if solvate_data['water_model'] is not None:
        pdb2gmx_list.extend(['-water', solvate_data['water_model']])
        water_str = None
    else:
        water_str = '1'

    try:
        forcefield_str = int(forcefield)
    except ValueError:
        pdb2gmx_list.extend(['-ff', forcefield])
        forcefield_str = None

    communicate_str = ''
    if forcefield_str is not None:
        communicate_str = '{}\n'.format(forcefield_str)
        if water_str is not None:
            communicate_str += '{}\n'.format(water_str)

    os_util.run_gmx(gmx_bin, pdb2gmx_list, communicate_str, build_files_dict['pdb2gmx_log'], verbosity=verbosity)

    # 2. Assemble the complex

    # 2.1 Copy the receptor PDB data, stripping all but ATOM and TER records
    complex_string = ''.join([each_line for each_line
                              in os_util.read_file_to_buffer(build_files_dict['protein_pdb'],
                                                             die_on_error=True, return_as_list=True,
                                                             error_message='Failed to read converted protein '
                                                                           'file when building the system.',
                                                             verbosity=verbosity)
                              if (each_line.find('ATOM') != -1 or each_line.find('TER') != -1)])

    os_util.local_print('Read {} lines from the pdb file {}'
                        ''.format(complex_string.count('\n'), build_files_dict['protein_pdb']),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # 2.2 Add small molecule PDB
    # FIXME: remove the need for setting molecule_name
    complex_string += '\n{}'.format(merge_topologies.dualmol_to_pdb_block(ligand_dualmol, molecule_name=ligand_name,
                                                                          verbosity=verbosity))

    with open(build_files_dict['system_pdb'], 'w') as fh:
        fh.write(complex_string)

    os_util.local_print('Wrote the protein + ligand to {} ({} lines)'
                        ''.format(build_files_dict['system_pdb'], complex_string.count('\n')),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # 2.3. Edit system topology to add ligand parameters
    system_topology_list = os_util.read_file_to_buffer(build_files_dict['protein_top'],
                                                       die_on_error=True, return_as_list=True,
                                                       error_message='Failed to read system topology file '
                                                                     'when building the system.',
                                                       verbosity=verbosity)

    ligand_dualmol.dual_topology.set_lambda_state(0)
    ligand_top = {'ligand.atp': ligand_dualmol.dual_topology.__str__('atomtypes'),
                  'ligand.itp': ligand_dualmol.dual_topology.__str__('itp')}
    for name, contents in ligand_top.items():
        with open(os.path.join(build_system_dir, name), 'w') as fh:
            fh.write(contents)

    ligatoms_list = ['\n', '; Include ligand atomtypes\n', '#include "ligand.atp"\n', '\n']
    ligatoms_list.reverse()
    ligtop_list = ['\n', '; Include ligand topology\n', '#include "ligand.itp"\n', '\n']
    ligtop_list.reverse()

    position = os_util.inner_search('.ff/forcefield.itp', system_topology_list, apply_filter=';')
    if position is False:
        new_file_name = os.path.basename(build_files_dict['protein_top'])
        shutil.copy2(build_files_dict['protein_top'], new_file_name)
        os_util.local_print('Failed to find a forcefield.itp import in topology file {}. This suggests a problem in '
                            'topology file formatting. Please, check inputs, especially force field. Copying {} to {}.'
                            ''.format(build_files_dict['protein_top'], build_files_dict['protein_top'], new_file_name),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    [system_topology_list.insert(position + 2, each_line) for each_line in ligatoms_list]

    position = os_util.inner_search('[ system ]', system_topology_list, apply_filter=';')
    if position is False:
        new_file_name = os.path.basename(build_files_dict['protein_top'])
        shutil.copy2(build_files_dict['protein_top'], new_file_name)
        os_util.local_print('Failed to find a [ system ] directive in topology file {}. This suggests a problem in '
                            'topology file formatting. Please, check inputs, especially force field. Copying {} to {}.'
                            ''.format(build_files_dict['protein_top'], build_files_dict['protein_top'], new_file_name),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    [system_topology_list.insert(position - 1, each_line) for each_line in ligtop_list]
    system_topology_list.append('{:<20} {}\n'.format(ligand_dualmol.dual_topology.molecules[0].name, '1'))

    with open(build_files_dict['proteinlig_top'], 'w') as fh:
        fh.writelines(system_topology_list)

    os_util.local_print('Topology file {} edited ({} lines)'
                        ''.format(build_files_dict['proteinlig_top'], system_topology_list.count('\n')),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # 2.4 Read the name of protein topology files and add to copyfile dict
    #   This will look for the first occurrence of a #include containing what should be a macromolecule topology, then
    #   read all subsequent #include lines with the same format, until another directive or including of other
    #   components (eg, water, ions) are found.
    copyfiles = {}
    molname = 'protein_step1_{}'.format(timestamp)
    search_fn = re.compile(r'#include\s+\"' + molname + r'_[a-zA-Z0-9_-]+\.itp\"')
    position = os_util.inner_search(search_fn.match, system_topology_list, apply_filter=';')
    if position is False:
        search_fn = re.compile(r'\s*\[\s+moleculetype\s+]')
        if os_util.inner_search(search_fn.match, system_topology_list, apply_filter=';') is False:
            new_file_name = os.path.basename(build_files_dict['protein_top'])
            shutil.copy2(build_files_dict['protein_top'], new_file_name)
            os_util.local_print('Failed to find both a #include directive for protein topologies and a '
                                '[ moleculetype ] directive in the topology file {}. This suggests a problem in '
                                'topology file formatting. Please, your check inputs. Copying {} to {}'
                                ''.format(build_files_dict['protein_top'], build_files_dict['protein_top'],
                                          new_file_name),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
    else:
        for each_line in system_topology_list[position:]:
            if each_line == '\n' or each_line.lstrip().startswith(';'):
                continue
            if not search_fn.match(each_line):
                break
            else:
                protein_itp = re.findall(molname + r'_[a-zA-Z0-9_-]+\.itp', each_line)[0]
                copyfiles[os.path.join(build_system_dir, protein_itp)] = protein_itp
        if not copyfiles:
            if no_checks:
                os_util.local_print('Failed to parse #include directives for protein topologies in topology file {}. '
                                    'This should not happen. Because you are running with no_checks, I will try to '
                                    'go on.'
                                    ''.format(build_files_dict['protein_top']),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            else:
                new_file_name = os.path.basename(build_files_dict['protein_top'])
                shutil.copy2(build_files_dict['protein_top'], new_file_name)
                os_util.local_print('Failed to parse #include directives for protein topologies in topology file {}. '
                                    'This suggests a problem in topology file formatting. Please, your check inputs. '
                                    'Copying {} to {}.'
                                    ''.format(build_files_dict['protein_top'], build_files_dict['protein_top'],
                                              new_file_name),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)

    # 2.5.2 Tries to find restraint files in the topology (this is the case for a single protein chain)
    search_fn = re.compile(r'#ifdef POSRES\s')
    position = os_util.inner_search(search_fn.match, system_topology_list, apply_filter=';')
    if position is not False:
        for each_line in system_topology_list[position + 1:]:
            each_line = each_line.lstrip()
            if each_line.startswith(';'):
                continue
            if each_line.startswith('#include'):
                this_filename = re.findall(r'"\w+.itp"', each_line)
                try:
                    this_filename = this_filename[0]
                except KeyError:
                    new_file_name = os.path.basename(build_files_dict['protein_top'])
                    shutil.copy2(build_files_dict['protein_top'], new_file_name)
                    os_util.local_print('Failed to parse #include directive for restraint file in topology file {}. '
                                        'This suggests a problem in topology file formatting. Please, your check '
                                        'inputs. Copying {} to {}.'
                                        ''.format(build_files_dict['protein_top'], build_files_dict['protein_top'],
                                                  new_file_name),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(1)
                else:
                    this_filename = this_filename.replace('"', '')
                    dest_file_name = os.path.join(build_system_dir, this_filename)
                    shutil.move(this_filename, dest_file_name)
                    copyfiles[dest_file_name] = this_filename
            elif each_line.startswith('#endif'):
                break

    # 2.5.2 From protein topology files, tries to read position restraints files and add them to copyfile dict
    for each_file in copyfiles.copy().values():
        each_file = os.path.join(build_system_dir, each_file)
        this_file_data = os_util.read_file_to_buffer(each_file, die_on_error=True, return_as_list=True,
                                                     error_message='Failed to read topology file when building the '
                                                                   'system.',
                                                     verbosity=verbosity)
        position = os_util.inner_search('#ifdef POSRES', this_file_data, apply_filter=';')
        if position is False:
            os_util.local_print('Protein topology file {} does not have a "#ifdef POSRES" directive. I cannot '
                                'find the position restraint file, so you will may not be able to use position '
                                'restraints in this system'.format(each_file),
                                msg_verbosity=os_util.verbosity_level.warning,
                                current_verbosity=verbosity)
        else:
            line_data = this_file_data[position + 1].split()
            if line_data[0] != '#include':
                os_util.local_print('Could not understand your POSRES directive {} in file {}. I cannot '
                                    'find the position restraint file, so you will may not be able to use '
                                    'position restraints in this system'.format(line_data, each_file),
                                    msg_verbosity=os_util.verbosity_level.warning,
                                    current_verbosity=verbosity)
            else:
                old_file_name = line_data[1].strip('"')
                new_file_name = os.path.join(build_system_dir, old_file_name)
                shutil.move(old_file_name, new_file_name)
                copyfiles[new_file_name] = old_file_name

    # 3. Generate simulation box (gmx editconf) and solvate the complex (gmx solvate)
    box_size = solvate_data['water_shell'] / 10.0
    if box_size < 1:
        os_util.local_print('You selected a water shell smaller than 10 \u00C5. Note that length units in input files '
                            'are \u00C5.',
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    editconf_list = ['editconf', '-f', build_files_dict['system_pdb'], '-d', str(box_size),
                     '-o', build_files_dict['system_gro'], '-bt', solvate_data['box_type']]
    os_util.run_gmx(gmx_bin, editconf_list, '', build_files_dict['editconf_log'],
                    verbosity=verbosity)

    # 4. Solvate simulation box
    solvent_box = guess_water_box(solvate_data['water_model'], build_files_dict['protein_top'], verbosity=verbosity)

    shutil.copy2(build_files_dict['proteinlig_top'], build_files_dict['systemsolv_top'])
    solvate_list = ['solvate', '-cp', build_files_dict['system_gro'], '-cs', solvent_box,
                    '-o', build_files_dict['systemsolv_gro'], '-p', build_files_dict['systemsolv_top']]
    os_util.run_gmx(gmx_bin, solvate_list, '', build_files_dict['solvate_log'],
                    verbosity=verbosity, alt_environment={'GMX_MAXBACKUP': '-1'})

    # Some users experienced a problem during the gmx grompp step below, apparently filesystem-related, which could be
    # due to a delay in the editing of build_files_dict['systemsolv_top'] by gmx solvate subprocess. The os.sync() may
    # fix this by forcing the file to be written to disk.
    try:
        os.sync()
    except AttributeError:
        os_util.local_print('os.sync() not found. Is this a non-Unix system or Python version < 3.3?',
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    # 5. Add ions to system
    # 5.1. Prepare a dummy mdp to run genion
    with open(build_files_dict['genion_mdp'], 'wb') as genion_fh:
        genion_fh.write(b'\n')

    # 5.3. Prepare a tpr to genion
    grompp_list = ['grompp', '-f', build_files_dict['genion_mdp'],
                   '-c', build_files_dict['systemsolv_gro'], '-p', build_files_dict['systemsolv_top'],
                   '-o', build_files_dict['genion_tpr'], '-maxwarn', str(gmx_maxwarn),
                   '-po', build_files_dict['mdout_mdp']]
    os_util.run_gmx(gmx_bin, grompp_list, '', build_files_dict['grompp_log'],
                    verbosity=verbosity)

    # 5.2. Run genion
    shutil.copy2(build_files_dict['systemsolv_top'], build_files_dict['fullsystem_top'])
    genion_list = ['genion', '-s', build_files_dict['genion_tpr'],
                   '-p', build_files_dict['fullsystem_top'], '-o', build_files_dict['fullsystem_pdb'],
                   '-conc', str(solvate_data['ion_concentration']), '-neutral',
                   '-nname', solvate_data['nname'], '-pname', solvate_data['pname']]
    os_util.run_gmx(gmx_bin, genion_list, 'SOL\n', build_files_dict['genion_log'],
                    alt_environment={'GMX_MAXBACKUP': '-1'}, verbosity=verbosity)

    # 6. Make index

    # Include a Protein_{Ligand} group in case users does not include one
    if index_groups is None:
        os_util.local_print('Added a Protein_LIG group to the group index, which will be generated '
                            'automatically', msg_verbosity=os_util.verbosity_level.info,
                            current_verbosity=verbosity)
        if selection_method == 'mdanalysis':
            index_groups = {'Protein_LIG': 'resname Protein or resname {}'.format(ligand_name)}
        elif selection_method == 'internal':
            index_groups = {'Protein_LIG': '"Protein" | "{}"'.format(ligand_name)}
    elif 'Protein_LIG' not in index_groups:
        os_util.local_print('Added a Protein_LIG group to the group index, which contains the following: {}'
                            ''.format([each_key for each_key in index_groups.keys()]),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        if selection_method == 'mdanalysis':
            index_groups.update({'Protein_LIG': 'resname Protein or resname {}'.format(ligand_name)})
        elif selection_method == 'internal':
            index_groups.update({'Protein_LIG': '"Protein" | "{}"'.format(ligand_name)})

    make_index(build_files_dict['index_ndx'], build_files_dict['fullsystem_pdb'], index_groups,
               selection_method, gmx_bin=gmx_bin, logfile=build_files_dict['makeindex_log'], verbosity=verbosity)

    # 7. Copy files do lambda directories
    copyfiles.update({build_files_dict['fullsystem_top']: build_files_dict['full_topology_file'],
                      build_files_dict['fullsystem_pdb']: build_files_dict['final_structure_file'],
                      build_files_dict['index_ndx']: build_files_dict['final_index_file']})

    for each_source, each_dest in copyfiles.items():
        os_util.local_print('Copying file {} to {}'
                            ''.format(each_source, os.path.join(base_dir, 'protein',
                                                                each_dest)),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        shutil.copy2(each_source, os.path.join(base_dir, 'protein', each_dest))

    return all_classes.Namespace({'build_dir': build_system_dir, 'structure': output_structure_file})


def prepare_output_scripts_data(header_template=None, script_type='bash', submission_args=None,
                                custom_scheduler_resources=None, hrex_frequency=0, collect_type='bin', config_file=None,
                                index_file='index.ndx', temperature=298.15, gmx_bin='gmx', n_jobs=1, run_before=None,
                                run_after=None, scripts_templates='templates/output_files_data.ini',
                                analysis_options=None, ligand_name='LIG', verbosity=0):
    """ Prepare data to be used when generating the output scripts

    :param [str, list, dict] header_template: read headers from this file, from this list of files or from this dict
    :param str script_type: type of output scripts
    :param str submission_args: pass this arguments to submission executable
    :param dict custom_scheduler_resources: if script_type is a scheduler, select this resources when submitting the
                                            jobs
    :param int hrex_frequency: attempt to exchange replicas every this much frames (Default: 0, no HREX)
    :param str collect_type: select script or brinary used to collect rerun data; options: bin, python, no_collect (do
                             not collect), or the path for a executable to be used to process.
    :param str config_file: default output data configuration file - regular users do not need to set this (None: use
                            internal default)
    :param str index_file: index file to be used during runs and analysis
    :param float temperature: absolute temperature of the sampling
    :param str gmx_bin: use this gromacs binary on the run node
    :param int n_jobs: use this many jobs during rerun, collect and analysis steps
    :param str run_before: this will be run before everything else on the output, will detect if file
    :param str run_after: this will be run before everything else on the output, will detect if file
    :param str scripts_templates: read default templates from this file
    :param dict analysis_options: extra options for analysis. Defaults {'skip_frames': 100, }
    :param int verbosity: verbosity level
    :param str ligand_name: use this as ligand name
    :rtype: all_classes.Namespace
    """

    if analysis_options is None:
        analysis_options = {}
    analysis_options.setdefault('skip_frames', 100)

    config_file = os.path.join(os.path.dirname(__file__), scripts_templates) if config_file is None else config_file

    # To be able to read multiline options starting with '#'
    outputscript_data = configparser.RawConfigParser(comment_prefixes=';')
    outputscript_data.read(config_file)

    constant_data = outputscript_data['constant_part']

    # Holder for scripts parts (preserving order)
    scripts_names = os_util.detect_type(outputscript_data['default']['scripts_names'], test_for_list=True)

    if not isinstance(scripts_names, list):
        os_util.local_print('Failed to read the output sequence from file {}. This should not happen')
        raise SystemExit(-1)

    output_constantpart = all_classes.Namespace([(s, '') for s in scripts_names])

    output_header = ''

    if not hrex_frequency:
        # User doesn't want HREX
        output_constantpart['run'] = constant_data['runnohrex']
    else:
        # Parse hrex_frequency
        hrex_frequency = os_util.detect_type(hrex_frequency)

    if collect_type == 'python':
        collect_executables = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dist', each_file)
                               for each_file in ['collect_results_from_xvg.py']]
    elif collect_type == 'bin':
        collect_executables = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dist',
                                            'collect_results_from_xvg')]
    elif collect_type == 'no_collect':
        collect_executables = None
    else:
        collect_executables = collect_type

    if collect_executables:
        collect_on_node = os.path.join('..', '..', '..', '..', os.path.basename(collect_executables[0]))
        collect_line = "chmod +x {} && " \
                       "./{} --temperature {} --gmx_bin {} --input *.xvg" \
                       "".format(collect_on_node, collect_on_node, temperature, gmx_bin)
    else:
        collect_line = 'echo "Skipping collecting xvg data "'

    # Substitute these on the run files
    substitution_constant = {'__N_JOBS__': n_jobs, '__LIG_GROUP__': ligand_name, '__COLLECT_BIN__': collect_line,
                             '__INDEX__': index_file, '__GMXBIN__': gmx_bin, '__HREX__': str(hrex_frequency),
                             '__SKIP_FRAMES__': analysis_options['skip_frames']}

    # Reads constant parts from constant_data into output_constantpart
    for step in output_constantpart:
        if step not in constant_data and step == 'run':
            if hrex_frequency <= 0:
                output_constantpart['run'] = constant_data['runnohrex']
            else:
                output_constantpart['run'] = constant_data['runhrex']
        elif step not in constant_data:
            continue
        else:
            output_constantpart[step] = constant_data[step]

        for each_holder, each_data in substitution_constant.items():
            output_constantpart[step] = output_constantpart[step].replace(each_holder, str(each_data))

    if custom_scheduler_resources:
        custom_scheduler_resources = os_util.detect_type(custom_scheduler_resources, test_for_dict=True)
        if not isinstance(custom_scheduler_resources, dict):
            os_util.local_print('Failed to read output_resources as dict (or None or False, ie: selecting '
                                'default resources). Value "{}" was read as a(n) {}'
                                ''.format(custom_scheduler_resources, type(custom_scheduler_resources)),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
            raise TypeError('dict, False, or NoneType expected, got {} instead'
                            ''.format(type(custom_scheduler_resources)))
        else:
            # Update defaults with user supplied resources
            outputscript_data['resources'].update({k: str(v) for k, v in custom_scheduler_resources.items()})

    try:
        template_section = outputscript_data[script_type]
    except KeyError:
        # User provided a custom submit command, so they must have used header_template and template_section
        # will not be used
        template_section = None
        submit_command = script_type
        depend_string = ''
    else:
        submit_command = template_section['submit_command']
        depend_string = template_section['depend_string']
        temp_str = template_section['header']
        for each_substitution, each_value in outputscript_data['resources'].items():
            this_step, this_resource = each_substitution.split('_')
            if this_step == 'all':
                temp_str = temp_str.replace('__{}__'.format(this_resource.upper()), str(each_value))
        output_header = temp_str

    if header_template:
        # User supplied output_template, try to read these as files
        header_template = os_util.detect_type(header_template, test_for_dict=True,
                                              test_for_list=True)
        if isinstance(header_template, list):
            header_data = [os_util.read_file_to_buffer(each_file, die_on_error=True, return_as_list=True,
                                                       error_message='Could not read output_template file.')
                           for each_file in header_template]
            if len(header_data) == 1:
                output_header = os_util.read_file_to_buffer(header_data[0], die_on_error=True, return_as_list=True,
                                                            error_message='Could not read output_template file.')
            else:
                os_util.local_print('I can only understand a single file as header_template (config: output_template). '
                                    'I read {} from your input {}.'.format(len(header_data), len(header_template)),
                                    current_verbosity=verbosity,
                                    msg_verbosity=os_util.verbosity_level.error)

                # os_util.local_print('When not using split output (default), argument or config file option '
                #                     'output_template requires a single file. I read {} from your input {}.'
                #                     ''.format(len(output_header), len(header_template)),
                #                     current_verbosity=verbosity,
                #                     msg_verbosity=os_util.verbosity_level.error)
                raise SystemExit(1)

        elif isinstance(header_template, dict):
            output_header = {key: os_util.read_file_to_buffer(each_file, die_on_error=True, return_as_list=True,
                                                              error_message='Could not read output_template file.')
                             for key, each_file in header_template.items()}
        else:
            os_util.local_print('Failed to read output_template as list or dict (or None or False, ie: selecting '
                                'default template). Value "{}" was read as a(n) {}'
                                ''.format(header_template, type(header_template)),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
            raise TypeError('list, dict, False, or NoneType expected, got {} instead'
                            ''.format(type(header_template)))
    elif template_section is None:
        os_util.local_print('If a custom submit command is provided via output_scripttype (ie: output_scripttype'
                            'is not in [{}]), then you must provide custom template files via output_template'
                            ''.format(', '.join([s for s in outputscript_data.sections() if s != 'constant_part'])),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise SystemExit(1)

    selfextracting_script_output = outputscript_data['default']['selfextracting']

    scripts_names.remove('pack')
    temp_str = template_section['header']
    for each_substitution, each_value in outputscript_data['resources'].items():
        this_step, this_resource = each_substitution.split('_')
        if this_step == 'pack':
            temp_str = temp_str.replace('__{}__'.format(this_resource.upper()), str(each_value))
    pack = temp_str
    pack += '\n\n' + output_constantpart['pack']

    return_data = all_classes.Namespace({'selfextracting_script': selfextracting_script_output,
                                         'constantpart': output_constantpart, 'header': output_header,
                                         'submit_command': submit_command, 'depend_string': depend_string,
                                         'collect_executables': collect_executables, 'pack_script': pack,
                                         'shebang': outputscript_data['default']['shebang'], 'parts': scripts_names})

    return_data['submission_args'] = submission_args if submission_args else ''

    for key, value in {'run_before': run_before, 'run_after': run_after}.items():
        if not value:
            return_data[key] = None
        else:
            file_data = os_util.read_file_to_buffer(value)
            if file_data is False:
                # Reading as a file failed, must be a string with the commands
                return_data[key] = value

    return return_data


def parse_input_molecules(input_data, verbosity=0):
    """ Flexible reader for molecules. Tries to understand

    :param [str, list, dict] input_data: data to be read
    :param int verbosity: set verbosity
    :rtype: dict
    """

    os_util.local_print('Entering parse_input_molecules(input_data={}, verbosity={})'
                        ''.format(input_data, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if isinstance(input_data, str):
        # User provided a file name, tries to read it
        if os.path.splitext(input_data)[1] == '.pkl':
            # Its a pickle file and must contain a dict, read it to input_data
            import pickle
            try:
                with open(input_data, 'rb') as fh:
                    temp_dict = pickle.load(fh)
            except (FileNotFoundError, IOError, EOFError):
                os_util.local_print('Failed to read the input file {}. It was guessed to be a pickle file from its '
                                    'extension.'
                                    ''.format(input_data),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
            else:
                input_data = temp_dict
        else:
            try:
                dir_data = os.listdir(input_data)
            except NotADirectoryError:
                # It's a text file. Read and parse it.
                file_data = os_util.read_file_to_buffer(input_data, die_on_error=False, return_as_list=False,
                                                        error_message='Failed to read input ligand file',
                                                        verbosity=verbosity)
                temp_dict = os_util.detect_type(file_data, test_for_dict=True, test_for_list=True, verbosity=verbosity)
                if not isinstance(temp_dict, (dict, list)):
                    os_util.local_print('Failed to parse file {}. This is the data read:\n{}'
                                        ''.format(input_data, temp_dict),
                                        current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
                else:
                    input_data = temp_dict
            except FileNotFoundError:
                os_util.verbosity_level('Could not parse input ligand data "{}". Please see documentation'
                                        ''.format(input_data),
                                        current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
                raise SystemExit(1)
            else:
                # Read the directory structure, add all files with the same name and different extensions to a value
                # under key = filename to the dict, if a dir is found, add its contents to a value
                temp_dict = {}
                [temp_dict.setdefault(os.path.splitext(each_file)[0], []).append(os.path.join(input_data, each_file))
                 if not os.path.isdir(os.path.join(input_data, each_file))
                 else temp_dict.setdefault(os.path.splitext(each_file)[0], []).extend(
                    [os.path.join(input_data, each_file, inner_file)
                     for inner_file in
                     os.listdir(os.path.join(input_data, each_file))])
                 for each_file in dir_data]
                input_data = temp_dict

    # User either provided a list or I parsed data to a list, try to figure out names from it
    if isinstance(input_data, list):
        # User provided a list. Read files from the list add all files with the same name and different extensions to
        # a value under key = filename to the dict
        ligand_dict = {}
        [ligand_dict.setdefault(os.path.splitext(each_file)[0], []).append(each_file) for each_file in input_data]
    elif isinstance(input_data, dict):
        ligand_dict = input_data
    elif input_data is None:
        ligand_dict = None
    else:
        os_util.local_print('Failed to read molecules data from {}. Could not parse data with type {}.\nInput data '
                            ''.format(input_data, type(input_data)),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(-1)

    return ligand_dict


def parse_poses_data(poses_data, no_checks=False, verbosity=0):
    """

    :param dict poses_data: poses data to be processed
    :param bool no_checks: ignore checks and try to go on
    :param int verbosity: set verbosity
    :rtype: dict
    """

    os_util.local_print('Entering parse_poses_data(poses_data={}, no_checks={}, verbosity={})'
                        ''.format(poses_data, no_checks, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if poses_data is None:
        return None

    temp_data = {}
    for each_name, each_data in poses_data.items():
        each_data = os_util.detect_type(each_data)
        if isinstance(each_data, str):
            temp_data[each_name] = each_data
        elif isinstance(each_data, list):
            if no_checks:
                os_util.local_print('More than one pose file read for ligand {}. Data read was "{}". Because you used '
                                    '"no_checks", selecting the first one ({}) and going on.'
                                    ''.format(each_name, each_data, each_data[0]),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                temp_data[each_name] = each_data[0]
            else:
                os_util.local_print('More than one pose file read for ligand {}. Data read was "{}". Please, check '
                                    'your input'
                                    ''.format(each_name, each_data),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(-1)
        else:
            os_util.local_print('Failed to parse pose data for molecule {}. Read data "{}" as a {}, but this is not '
                                'supported'.format(each_name, each_data, type(each_data)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(-1)

    return temp_data


def parse_ligands_data(input_ligands, savestate_util=None, no_checks=False, verbosity=0):
    """ Parse input ligands data from a text file, a pickle file, a list, or a dict

    :param [str, dict, list] input_ligands: input to be read
    :param savestate_util.SavableState savestate_util: progress data
    :param bool no_checks: ignore checks and try to keep going
    :param int verbosity: sets the verbosity level
    :rtype: dict
    """

    os_util.local_print('Entering parse_ligands_data(input_ligands={}, savestate_util={}, no_checks={}, verbosity={}))'
                        ''.format(input_ligands, savestate_util, no_checks, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    input_ligands = os_util.detect_type(input_ligands, test_for_dict=True, test_for_list=True, verbosity=verbosity)

    ligand_dict = parse_input_molecules(input_ligands, verbosity=verbosity)
    old_ligand_dict = ligand_dict.copy()

    # Reprocess the dict to convert lists in inner dicts
    for each_name, each_molecule in ligand_dict.items():
        if isinstance(each_molecule, dict):
            continue

        if isinstance(each_molecule, str):
            each_molecule = os_util.detect_type(each_molecule, test_for_list=True)
            if isinstance(each_molecule, str):
                each_molecule = [each_molecule]

        if isinstance(each_molecule, list):
            # Tries to detect if the file is a molecule file (ie: mol2 or pdb) or a topology (everything else)
            new_dict = {}
            for each_field in each_molecule:
                this_ext = os.path.splitext(each_field)[1]
                if this_ext in ['.mol2', '.mol']:
                    temp_mol = rdkit.Chem.MolFromMol2File(each_field, removeHs=False) if this_ext == '.mol2' \
                        else rdkit.Chem.MolFromMolFile(each_field, removeHs=False)
                    if temp_mol is None:
                        os_util.local_print('Failed to read molecule {} as a {} file. Please, check your input.'
                                            ''.format(each_field, this_ext),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise SystemExit(-1)
                    new_dict['molecule'] = temp_mol
                elif this_ext == '.pdb':
                    if not no_checks:
                        os_util.local_print('PDB format is not supported. In order to avoid bond-order '
                                            'related problems, please, use a mol2 file. Alternatively, rerun with '
                                            'no_checks to force reading of a PDB.',
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                    else:
                        os_util.local_print('You are using a PDB file for a ligand ({}). This is is likely a bad idea. '
                                            'Because you used no_checks, I will try to use it anyway. Be aware that '
                                            'bond orders, aromaticity, atoms types, charges and pretty much anything '
                                            'else may be incorrectly detected.'
                                            ''.format(each_field),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                        new_dict['molecule'] = rdkit.Chem.MolFromPDBFile(each_field, removeHs=False)
                elif this_ext in ['.itp', '.atp', '.top', '.str']:
                    new_dict.setdefault('topology', []).append(each_field)
                elif os.path.isdir(each_field):
                    new_dict.setdefault('topology', []).append(each_field)
                else:
                    # Unknown extension, but I will assume it's a topology file anyway
                    os_util.local_print('Unknown file extension for {}, assuming it is a GROMACS-compatible topology '
                                        'file.'.format(each_field),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    new_dict.setdefault('topology', []).append(each_field)
            ligand_dict[each_name] = new_dict
        else:
            os_util.local_print('Failed to read ligand input. I expected a dict of lists or a dict of dicts, but I '
                                'found a {}. The data is:\n{}'.format(type(each_molecule), ligand_dict),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise TypeError('Expected dict or list, got {}'.format(type(each_molecule)))

    # User either provided a dict or I parsed data to a dict. Complete it with data form savestate_util, if available
    if savestate_util:
        savestate_util['ligands_data'] = os_util.recursive_update(savestate_util.setdefault('ligands_data', {}),
                                                                  ligand_dict)
        ligand_dict = savestate_util['ligands_data']
        savestate_util.save_data()

    # Check the data, convert if needed
    # TODO: do the .str->GROMACS conversion without a subprocess
    for each_name, each_data in ligand_dict.items():
        if 'molecule' not in each_data:
            os_util.local_print('Could not find molecule data for ligand {}. Please check your input file or '
                                'command line arguments.\n\tLigand data read is:\n\t{}\n\tData in saved state '
                                'is:\n\t{}'
                                ''.format(each_name, ligand_dict, savestate_util['ligands_data']),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        if 'topology' not in each_data or len(each_data['topology']) == 0:
            os_util.local_print('Could not find topology data for ligand {}. Please check your input file or '
                                'command line arguments.\n\tLigand data read is:\n\t{}\n\tData in saved state '
                                'is:\n\t{}'
                                ''.format(each_name, ligand_dict, savestate_util['ligands_data']),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        elif len(each_data['topology']) == 2:
            files_set = {os.path.splitext(each_data['topology'][0])[1], os.path.splitext(each_data['topology'][1])[1]}
            if files_set == {'.str', '.ff'}:
                # Get a ff dir and the input str
                force_field_dir = [filename for filename in each_data['topology']
                                   if os.path.splitext(filename)[1] == '.ff'][0]
                str_file = [filename for filename in each_data['topology']
                            if os.path.splitext(filename)[1] == '.str'][0]
                # Try to find a .mol2 file in the input, check if successful
                mol2_file = [filename for filename in old_ligand_dict[each_name]
                             if os.path.splitext(filename)[1] == '.mol2']
                if len(mol2_file) == 1:
                    mol2_file = mol2_file[0]
                    os_util.local_print('Based on your input topology for molecule {}, I detected you are using CGenFF '
                                        'topology in str format and supplied a force field ({}). Starting automatic '
                                        'conversion using {}. If you need finer control over topology conversion, '
                                        'please do it before calling {}'
                                        ''.format(each_name, force_field_dir, mol2_file, os.path.basename(__file__)),
                                        msg_verbosity=os_util.verbosity_level.warning,
                                        current_verbosity=verbosity)
                else:
                    if no_checks:
                        mol2_file = os.path.join(os.path.dirname(str_file),
                                                 os.path.splitext(str_file)[0] + os.extsep + 'mol2')
                        os_util.local_print('You supplied a CGenFF .str file ({}) as ligand topology, but no '
                                            'correspondingly .mol2 file was found. Because you are running with '
                                            'no_checks, I will try to generate a {} file from the inputs. Be aware '
                                            'that this may fail.'.format(str_file, mol2_file),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                        mol_util.obmol_to_rwmol(each_data['molecule']).write('mol2', mol2_file)
                    else:
                        os_util.local_print('You supplied a CGenFF .str file ({}) as ligand topology, but no '
                                            'correspondingly .mol2 file was found. A .mol2 file is required to convert '
                                            'a CGenFF topology to a GROMACS-compatible one. You can rerun with '
                                            'no_checks so I will try to generate a .mol2 file from the inputs, but '
                                            'this is not ideal.'.format(str_file),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                        raise SystemExit(1)
                output_files = str_to_gmx(str_file=str_file, mol2_file=mol2_file, mol_name=each_name,
                                          force_field_dir=force_field_dir, output_dir=os.path.dirname(str_file),
                                          no_checks=no_checks, verbosity=verbosity)
                ligand_dict[each_name]['topology'] = output_files['topology']

                temp_mol = rdkit.Chem.MolFromMol2File(output_files['molecule'], removeHs=False)
                if temp_mol is None:
                    os_util.local_print('Failed to read molecule {} as a {} file. Please, check your input.'
                                        ''.format(each_field, this_ext),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(-1)
                ligand_dict[each_name]['molecule'] = temp_mol
                os_util.local_print('Replaced molecule with the one in {}, following the'
                                    ''.format(output_files['molecule']),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        elif len(each_data['topology']) == 1 and os.path.splitext(each_data['topology'][0])[1] == '.str':
            os_util.local_print('You supplied a CGenFF .str file ({}) as ligand topology, but no '
                                'force field directory was found. A .ff directory is required to convert '
                                'a CGenFF topology to a GROMACS-compatible one. Please, add a CHARMM force field '
                                'directory (eg, charmm36-xxx0000.ff) to your input_ligands for molecule {}.'
                                ''.format(each_name, each_data['topology'][0]),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

    ligand_table = '{:=^50}\n{:^10}{:^20}{:^20}\n'.format(' Input ligands ', ' Name ', 'Molecule', 'Topology')
    table_rows = []
    for this_ligand, lig_data in ligand_dict.items():
        # Safe formatting for ligand name
        lig_mol = lig_data['molecule'] if isinstance(lig_data['molecule'], str) \
            else rdkit.Chem.MolToSmiles(lig_data['molecule'])
        table_rows.append('{:^10} {:^19} {:^19}'.format(this_ligand, lig_mol, ', '.join(lig_data['topology'])
        if len(lig_data['topology']) > 1 else lig_data['topology'][0]))
    ligand_table += '\n'.join(table_rows)
    ligand_table += '\n' + '=' * 50
    os_util.local_print(ligand_table,
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    if len(ligand_dict) <= 1:
        os_util.local_print('Problem while reading the input ligands. Too few ligands were found.\n\tThis is the data '
                            'read (command line + config files + progress file):\n\t{}\n\tThis is your (parsed) '
                            'input_ligands (from either command line and config '
                            'files):\n\t{}'.format(ligand_dict, input_ligands),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise SystemExit(1)

    return ligand_dict


def read_index_data(index_file, verbosity=0):
    """
    :param str index_file: name of the index file
    :param int verbosity: sets the verbosity level
    :rtype: dict
    """

    index_data = os_util.read_file_to_buffer(index_file, die_on_error=False, return_as_list=False)
    if index_data:
        index_data = index_data.replace('\n', ' ')
        index_dict = dict(zip(re.findall(r'\[\s+(.+?)\s+\]', index_data),
                              [list(map(int, group.split())) for group in re.split(r'\[\s+.+?\s+\]', index_data)[1:]]))
        os_util.local_print('Read index groups: {}'.format(', '.join(['{}: {}'.format(k, len(v))
                                                                      for k, v in index_dict.items()])),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

        return index_dict
    else:
        return None


def uniquify_index_file(filename, verbosity=0):
    """ Edits an index file, removing duplicated groups. Warning: will edit in place

    :param str filename: file to be edited
    :param int verbosity: verbosity level
    """

    index_data = read_index_data(filename, verbosity=verbosity)
    output = []
    for name, atoms in index_data.items():
        output.append('[ {} ]'.format(name))
        new_line = ''
        for atom in atoms:
            if len('{} {}'.format(new_line, atom)) > 120:
                output.append(new_line)
                new_line = str(atom)
            else:
                new_line += ' {}'.format(atom)
        output.append(new_line)
    output.append("")

    with open(filename, 'w') as fh:
        fh.write('\n'.join(output))


def read_md_protein(structure_file, structure_format='', last_protein_atom=-1, verbosity=0):
    """ Loads MD data from structure_file up to last_protein_atom atom

    :param str structure_file: filename of md data
    :param str structure_format: pybel-compatible file type of md data
    :param int last_protein_atom: read this much atoms (-1: entire file)
    :param int verbosity: be verbosity
    """

    os_util.local_print('Entering read_md_protein(structure_file={}, structure_format={}, last_protein_atom={}, '
                        'verbosity={})'.format(structure_file, structure_format, last_protein_atom, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if structure_format == '':
        structure_format = structure_file.split('.')[-1]

    protein_data = os_util.read_file_to_buffer(structure_file, die_on_error=True, return_as_list=True,
                                               error_message='Failed to read the system file {}. Cannot continue. '
                                                             'Please, use the structure option'
                                                             ''.format(structure_file),
                                               verbosity=verbosity)

    if last_protein_atom == -1:
        protein_data = pybel.readstring(structure_format, structure_file)
        return protein_data
    else:
        # TODO: proper handling of the protein selection

        test_atom = int(protein_data[last_protein_atom][6:11])
        shift_lines = last_protein_atom - test_atom
        protein_data = ''.join(protein_data[:last_protein_atom + shift_lines + 1])

        try:
            # Warning: assumes a single-strucutre PDB
            protein_data = pybel.readstring(structure_format, protein_data)
        except ValueError:
            os_util.local_print('Could not understand format {} (guessed from extension)'.format(structure_format),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        except IOError:
            os_util.local_print('Could not read file {}'.format(structure_file),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        except StopIteration:
            os_util.local_print('Could not understand file {} using format {}'.format(structure_file, structure_format),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        else:
            msg_string = '\n'.join(['{:=^30}'.format(' MD Data '), 'Loaded data from file {} using format {}'
                                                                   ''.format(structure_file, structure_format)])
            os_util.local_print(msg_string, msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)
            return protein_data


def align_ligands(receptor_structure, poses_input=None, poses_reference_structure=None,
                  pose_loader='generic', reference_pose_superimpose=None, superimpose_loader_ligands=None,
                  ligand_residue_name='LIG', cluster_docking_data=None, save_state=False, verbosity=0, **kwargs):
    """ Align ligands from a docking structure to the MD-equilibrated frame or the input structure

    :param pybel.Molecule receptor_structure: md frame to be used in the alignment
    :param dict poses_input: dict containing the ligands to be read, keys=names and values=file path (required for all
                             loaders but superimpose loader)
    :param str poses_reference_structure: reference structure in respect to poses
    :param str pose_loader: format of the docking data ('pdb', 'autodock4', 'superimpose', 'generic')
    :param str reference_pose_superimpose: use this file as reference pose (superimpose loader only)
    :param dict superimpose_loader_ligands: molecules to be superimposed to reference
    :param str ligand_residue_name: residue name for the ligand (PDB loader only)
    :param [str, dict] cluster_docking_data: a file containing ligand names and the selected clusters to be used in the
                                             FEP, if the file is absent or a ligand name is absent from file, cluster 1
                                             will be used (autodock4 loader only)
    :param savestate_utils.SavableState save_state: object with saved data
    :param int verbosity: verbosity level
    :rtype: rdkit.RWMol

    """
    # TODO: read data from other docking programs (eg: autodock-vina, rdock and dock 3.7)

    for k, v in {'seq_align_mat': 'blosum80', 'gap_penalty': -1}.items():
        kwargs.setdefault(k, v)

    if not poses_input and pose_loader not in ['superimpose']:
        os_util.local_print('Missing poses_input, required for pose loader {}. Please, see manual'
                            ''.format(arguments.pose_loader),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        return False

    if poses_reference_structure is None:
        poses_reference_structure = receptor_structure
        if pose_loader != 'pdb':
            os_util.local_print('No poses_reference_structure supplied, using structure ({}) as reference '
                                'macromolecular structure.'
                                ''.format(receptor_structure.title),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    # Loads poses data
    if pose_loader == 'pdb':
        # This loader reads pdb files, selects a ligand using molecule name and align the ligands to the reference
        # structure using the Ca from the receptor in each pdb.

        os_util.local_print('Initial poses are being read from pdb files containing ligand and receptor structures',
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        from docking_readers.generic_loader import read_reference_structure
        from docking_readers.pdb_loader import extract_pdb_poses

        docking_receptor_mol = read_reference_structure(poses_reference_structure, verbosity=verbosity)
        docking_mol_local = extract_pdb_poses(poses_input, docking_receptor_mol,
                                              ligand_residue_name=ligand_residue_name, verbosity=verbosity)

    elif pose_loader == 'autodock4':
        # This loader loads data from autodock4 output, it tries to detect results files and receptor, but can be given
        # the actual file locations

        os_util.local_print('Initial poses are being read from Autodock4 data {}'.format(poses_input),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        from docking_readers.autodock4_loader import extract_autodock4_poses, extract_docking_receptor

        docking_receptor_mol = extract_docking_receptor(poses_reference_structure, verbosity=verbosity)
        docking_mol_local = extract_autodock4_poses(poses_input, cluster_docking_data, verbosity=verbosity)

    elif pose_loader == 'superimpose':
        # Superimpose a series of ligands into a pdb structure using rdkit functions

        os_util.local_print('Initial poses are being generated by superimposition from data {}'.format(poses_input),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        from docking_readers.superimpose_loader import superimpose_poses
        from docking_readers.generic_loader import read_reference_structure

        if not reference_pose_superimpose:
            os_util.local_print('Missing reference_pose_superimpose (config: poses_reference_pose_superimpose), '
                                'required for pose loader superimpose. Please, see documentation',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        if not superimpose_loader_ligands:
            os_util.local_print('Missing superimpose_loader_ligands, required for pose loader superimpose. Please, '
                                'see python reference.',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        docking_mol_local = superimpose_poses(superimpose_loader_ligands, reference_pose_superimpose,
                                              save_state=save_state, verbosity=verbosity, **kwargs)
        docking_receptor_mol = read_reference_structure(poses_reference_structure, verbosity=verbosity)

    elif pose_loader == 'generic':
        # A generic loader, which simply reads ligand files and a receptor file

        os_util.local_print('Initial poses are being read using generic loader',
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        from docking_readers.generic_loader import extract_docking_poses, read_reference_structure

        docking_receptor_mol = read_reference_structure(poses_reference_structure, verbosity=verbosity)
        docking_mol_local = extract_docking_poses(poses_input, verbosity=verbosity)

    else:
        os_util.local_print('Unknown pose_loader {}. Please select among superimpose, pdb, autodock4 and '
                            'generic'.format(pose_loader), msg_verbosity=os_util.verbosity_level.error,
                            current_verbosity=verbosity)
        return False

    if verbosity == 0:
        os_util.local_print('{:=^50}'.format(' Align poses '), msg_verbosity=os_util.verbosity_level.default,
                            current_verbosity=verbosity)

    align_data = align_utils.align_protein(docking_receptor_mol, receptor_structure,
                                           seq_align_mat=kwargs['seq_align_mat'],
                                           gap_penalty=kwargs['gap_penalty'], verbosity=verbosity)
    centering_vector = align_data['centering_vector']
    rot_vector_1d = align_data['rotation_matrix']
    proteinmd_center = align_data['translation_vector']

    docking_mol_transformed = {}
    for ligand_name, each_ligand in docking_mol_local.items():
        each_ligand = mol_util.rwmol_to_obmol(each_ligand, verbosity=verbosity)
        each_ligand.Translate(centering_vector)
        each_ligand.Rotate(pybel.ob.double_array(rot_vector_1d))
        each_ligand.Translate(proteinmd_center)
        ligand_mol = mol_util.obmol_to_rwmol(each_ligand, verbosity=verbosity)
        docking_mol_transformed[ligand_name] = ligand_mol
        os_util.local_print('Molecule {} aligned'.format(ligand_name),
                            msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    os_util.local_print('A total of {} ligands were aligned.'.format(len(docking_mol_local)),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    return docking_mol_transformed


def edit_itp(itp_filename):
    """ Edit a itp file, suppress atomtypes directive, remove preprocessor directives

    :param str itp_filename: itp file to be edited
    """
    new_itp_data = []
    try:
        with open(itp_filename) as file_handler:
            itp_data = file_handler.readlines()
    except IOError:
        os_util.local_print('Could not read from {}'.format(itp_filename), msg_verbosity=os_util.verbosity_level.error)
        raise SystemExit(1)
    else:
        types_found = False
        for each_line in itp_data:
            if each_line[0] == '#':
                continue
            if each_line.rstrip() == '[ atomtypes ]':
                types_found = True
                continue
            if types_found:
                if each_line.rstrip() == '[ moleculetype ]':
                    types_found = False
                else:
                    continue
            if not types_found:
                new_itp_data.append(each_line)

        try:
            with open(itp_filename, 'w') as file_handler:
                file_handler.writelines(new_itp_data)
        except IOError:
            os_util.local_print('Could write to {}'.format(itp_filename), msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(1)


def make_index(new_index_file, structure_data, index_data=None, method='internal', gmx_bin='gmx',
               logfile=None, ligand_name='LIG', verbosity=0):
    """ Generate a index from data in structure_data using groups in index_data

    :param str new_index_file: save index to this file
    :param [str, MDAnalysis.Universe] structure_data: structure file or MDAnalysis.Universe of system
    :param dict index_data: generate index for this selection groups
    :param str method: method to be used; options: internal (default), mdanalysis
    :param str gmx_bin: gromacs binary (only used if method == internal; default: gmx)
    :param str logfile: save log to this file. None: do not save
    :param str ligand_name: use this as ligand name
    :param int verbosity: verbosity level
    """

    os_util.local_print('Entering make_index: new_index_file={}, structure_data={}, index_data={}, method={}, '
                        'gmx_bin={}, verbosity={}'
                        ''.format(new_index_file, structure_data, index_data, method, gmx_bin, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if method == 'mdanalysis':
        import MDAnalysis
        import MDAnalysis.selections.gromacs

        if isinstance(structure_data, str):
            os_util.local_print('Reading {} to a MDAnalysis.Universe'.format(structure_data),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            structure_data = MDAnalysis.Universe(structure_data)
        elif not isinstance(structure_data, MDAnalysis.Universe):
            TypeError('Expected str or MDAnalysis.Universe, got {} instead'.format(type(structure_data)))

        # gmx make_ndx has some default groups, while MDAnalysis does not. Create some default groups in case user
        default_groups = {'System': 'all',
                          'Protein': 'protein',
                          'Water': 'resname SOL',
                          'LIG': 'resname {}'.format(ligand_name),
                          'C-alpha': 'name CA',
                          'Protein_LIG': 'protein or resname {}'.format(ligand_name),
                          'Water_and_ions': 'resname SOL or resname NA or resname CL or resname K'}

        index_input = default_groups
        if index_data is not None:
            [index_input.update({k: v}) for k, v in default_groups.items() if k not in index_data]

        # Generate a group per item in index_data and writes to new_index_file
        with MDAnalysis.selections.gromacs.SelectionWriter(new_index_file, mode='w') as ndx:
            for each_name, each_selection in index_data.items():
                this_selection = structure_data.select_atoms(each_selection)
                if this_selection.n_atoms == 0:
                    os_util.local_print('The index group {} created from the selection "{}" would be empty and will '
                                        'not be created. Please, check your input groups and the system file {}'
                                        ''.format(each_name, each_selection, structure_data),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                else:
                    ndx.write(this_selection, name=each_name)

    elif method == 'internal':
        # Uses subprocess and `gmx make_ndx` to generate index

        index_cmd = ['make_ndx', '-f', structure_data, '-o', new_index_file]

        if index_data:
            # User wants custom groups. Because gmx make_ndx complains when selecting non unique groups and non unique
            # groups are generated by make_ndx by default, we need to get rid of the duplicates.

            # First generate a naive file
            dummy_str = 'q\n'
            stdout, stderr = os_util.run_gmx(gmx_bin, index_cmd, input_data=dummy_str, verbosity=verbosity,
                                             alt_environment={'GMX_MAXBACKUP': '-1'}, return_output=True)

            # Then uniquify its names
            uniquify_index_file(new_index_file, verbosity=verbosity)

            # Now add the groups
            groups_str = '{}\nq\n'.format('\n'.join([each_element for each_name, each_element in index_data.items()]))

            index_cmd = ['make_ndx', '-n', new_index_file, '-o', new_index_file]
            stdout, stderr = os_util.run_gmx(gmx_bin, index_cmd, input_data='{}\nq\n'.format(groups_str),
                                             verbosity=verbosity, alt_environment={'GMX_MAXBACKUP': '-1'},
                                             return_output=True)
        else:
            stdout, stderr = os_util.run_gmx(gmx_bin, index_cmd, input_data='q\n', verbosity=verbosity,
                                             return_output=True, alt_environment={'GMX_MAXBACKUP': '-1'})

        if logfile:
            with open(logfile, 'w') as fh:
                fh.write(stderr)
                fh.write(stdout)

    else:
        os_util.local_print('Invalid selection method {}. Please select either "internal" or "mdanalysis".'
                            ''.format(method),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise ValueError('Invalid selection method {}'.format(method))

    ndx_groups = read_index_data(new_index_file, verbosity=0)

    if index_data:
        if not set(index_data.keys()).issubset(set(ndx_groups.keys())):
            os_util.local_print('Missing groups from index files. Please, verify your input files. Index groups '
                                'in index_data are:\n{}\nIndex groups in file:\n{}'
                                ''.format(', '.join(index_data.keys()), ', '.join(ndx_groups.keys())),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

    os_util.local_print('Index file {} generated from {}.\n These are its groups:\n{}'
                        ''.format(new_index_file, structure_data,
                                  '\n'.join(['{}: {} atoms'.format(k, len(v)) for k, v in ndx_groups.items()])),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)


def str_to_gmx(str_file, mol2_file, mol_name='LIG', force_field_dir=None, output_dir=None, no_checks=False,
               verbosity=0):
    """ Converts a CHARMM36 .str topology to a GROMACS compatible one

    :param str str_file: str topology to be converted
    :param str mol2_file: mol2 structure file
    :param str mol_name: name of the molecule, must match both .mol2 and .str residue name
    :param str force_field_dir: read a GROMACS-compatible CHARMM force field from this dir; None: auto detect from pwd
    :param str output_dir: save resulting files to this dir; if None, file contents will be returned, if a path is
     supplied, the path to the converted files will be returned
    :param bool no_checks: ignore checks and try to keep going
    :param int verbosity: verbosity level
    :rtype: dict
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tools',
                                   'cgenff_charmm2gmx_py3.py')
        shutil.copy2(mol2_file, tmpdirname)
        shutil.copy2(str_file, tmpdirname)
        shutil.copytree(force_field_dir, os.path.join(tmpdirname, os.path.basename(force_field_dir)))
        edited_mol2 = mol_name + '_alt' + os.extsep + 'mol2'
        cmdline = [script_path, '--mol_name', mol_name, '--mol2', os.path.basename(mol2_file),
                   '--str', os.path.basename(str_file), '--forcefield', os.path.basename(force_field_dir),
                   '--output_mol2', edited_mol2, '--include_atomtypes']
        os_util.local_print('Executing cgenff_charmm2gmx_py3.py with cmd="{}"'.format(' '.join(cmdline)),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        with subprocess.Popen(cmdline, cwd=tmpdirname, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) as this_process:
            stdout, stderr = this_process.communicate()
            return_code = this_process.returncode
        if return_code != 0:
            os_util.local_print('Failed to convert {} topology using cgenff_charmm2gmx_py3.py. The error '
                                'message was:\n{}\n\nFull cgenff_charmm2gmx_py3.py output:\n{}'
                                ''.format(str_file, stderr, stdout),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            os_util.local_print('{:=^50}\n{}'.format(' cgenff_charmm2gmx_py3.py output ', stdout, '=' * 50),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            raise subprocess.SubprocessError(stderr)
        else:
            output_files = [mol_name + os.extsep + each_extension for each_extension in ['prm', 'itp']]
            output_files += [edited_mol2]
            if output_dir is not None:
                for this_file_name in output_files:
                    try:
                        os_util.file_copy(os.path.join(tmpdirname, this_file_name), output_dir, error_if_exists=True,
                                          verbosity=verbosity)
                    except FileExistsError:
                        if no_checks:
                            os_util.local_print('Converted {} to a GROMACS-compatible topology, but a {} file '
                                                'exists in {}. Because you are running with no_checks, I AM '
                                                'OVERWRITING {}!'
                                                ''.format(str_file, this_file_name, output_dir,
                                                          os.path.join(output_dir, this_file_name)),
                                                msg_verbosity=os_util.verbosity_level.error,
                                                current_verbosity=verbosity)
                            os_util.file_copy(os.path.join(tmpdirname, this_file_name), output_dir)
                        else:
                            os_util.local_print('Converted {} to a GROMACS-compatible topology, but a {} file '
                                                'exists in {}, so I will not overwrite it. Use {} as '
                                                'topology or move/remove it and run again. You can also rerun '
                                                'with no_checks, so that files will be overwritten.'
                                                ''.format(str_file, this_file_name, output_dir, this_file_name),
                                                msg_verbosity=os_util.verbosity_level.error,
                                                current_verbosity=verbosity)
                            raise FileExistsError('File {} exists'.format(os.path.join(output_dir, this_file_name)))
                    else:
                        os_util.local_print('Converted {} to the GROMACS-compatible topology {}'
                                            ''.format(str_file, os.path.join(output_dir, this_file_name)),
                                            msg_verbosity=os_util.verbosity_level.info,
                                            current_verbosity=verbosity)
                return_data = {'topology': [os.path.join(output_dir, mol_name + os.extsep + each_extension)
                                            for each_extension in ['prm', 'itp']],
                               'molecule': os.path.join(output_dir, edited_mol2)}
                return return_data
            else:
                return_data = {'topology': [os_util.read_file_to_buffer(os.path.join(tmpdirname, mol_name + os.extsep
                                                                                     + each_extension))
                                            for each_extension in ['prm', 'itp']],
                               'molecule': os_util.read_file_to_buffer(os.path.join(tmpdirname, edited_mol2))}
                return return_data


def prepare_water_system(dual_molecule_ligand, water_dir, topology_file, protein_topology_files,
                         solvate_data=None, protein_dir=None, gmx_bin='gmx', ligand_name='LIG',
                         gmx_maxwarn=1, verbosity=0, **kwargs):
    """ Prepares a system consisting in a dual-topology molecule solvated and with counter ions added

    :param merge_topologies.MergedTopology dual_molecule_ligand: pseudo-molecule to be solvated
    :param str water_dir: store temporary and log files to this directory
    :param str topology_file: name of output topology file
    :param list protein_topology_files: files and directories to be used as source for topology
    :param dict solvate_data: dictionary containing further data to solvate: 'water_model': water model to be used,
                              'water_shell': size of the water shell in A, 'ion_concentration': add ions to this conc,
                              'pname': name of the positive ion, 'nname': name of the negative ion}

    :param str protein_dir: copy topology data from this directory (default: water_dir/../protein)
    :param str gmx_bin: Gromacs binary (default: gmx)
    :param str ligand_name: use this as ligand name
    :param int gmx_maxwarn: passed to gmx grompp -maxwarn
    :param int verbosity: set vebosity level
    """

    # Save intemediate files used to build the water system to this dir
    solvate_data = set_default_solvate_data(solvate_data)

    build_water_dir = os.path.join(water_dir, 'build_water_{}'.format(time.strftime('%H%M%S_%d%m%Y')))
    os_util.makedir(build_water_dir)

    # These are the intermediate build files
    build_files_dict = {index: os.path.join(build_water_dir, filename.format(time.strftime('%H%M%S_%d%m%Y')))
                        for index, filename in {'ligand_pdb': 'ligand_step1_{}.pdb',
                                                'ligand_top': 'ligand_step1_{}.top',
                                                'ligandbox_pdb': 'ligandbox_pdb_step2_{}.pdb',
                                                'ligandsolv_pdb': 'ligandsolv_step3_{}.pdb',
                                                'ligandsolv_top': 'ligandsolv_step3_{}.top',
                                                'genion_mdp': 'genion_step5_{}.mdp',
                                                'mdout_mdp': 'mdout_step5_{}.mdp',
                                                'genion_tpr': 'genion_step5_{}.tpr',
                                                'editconf_log': 'gmx_editconf_{}.log',
                                                'solvate_log': 'gmx_solvate_{}.log',
                                                'grompp_log': 'gmx_grompp_{}.log',
                                                'genion_log': 'gmx_genion_{}.log',
                                                'index_ndx': 'index.ndx',
                                                'makendx_log': 'makendx_{}.log',
                                                'fullsystem_pdb': 'fullsystem_{}.pdb',
                                                'fullsystem_top': 'fullsystem_{}.top'
                                                }.items()}

    # Copy topology files to water dir
    protein_dir = os.path.join(water_dir, '..', 'protein') if protein_dir is None else protein_dir
    for each_file in protein_topology_files:
        try:
            shutil.copy2(each_file, water_dir)
            shutil.copy2(each_file, build_water_dir)
        except IsADirectoryError:
            os_util.local_print('Copying {} to {}\nCopying {} to {}'
                                ''.format(each_file, os.path.join(water_dir, os.path.split(each_file)[-1]),
                                          each_file, os.path.join(build_water_dir, os.path.split(each_file)[-1])),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            try:
                shutil.copytree(each_file, os.path.join(build_water_dir, os.path.split(each_file)[-1]))
            except FileExistsError:
                pass
            try:
                shutil.copytree(each_file, os.path.join(build_water_dir, os.path.split(each_file)[-1]))
            except FileExistsError:
                pass

        except FileExistsError:
            pass

        os_util.local_print('Copying {} to {}'.format(each_file, water_dir),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Reads and edit system topology, removing all molecules but ligand
    system_topology_list = os_util.read_file_to_buffer(os.path.join(protein_dir, os.path.basename(topology_file)),
                                                       die_on_error=True, return_as_list=True,
                                                       error_message='Failed to read system topology file.',
                                                       verbosity=verbosity)

    molecules_pattern = re.compile(r'\[\s+molecules\s+\]')
    water_topology_list = None
    for line_number, each_line in enumerate(system_topology_list):
        if molecules_pattern.match(each_line) is not None:
            water_topology_list = system_topology_list[:line_number + 1]
            break

    if water_topology_list is not None:
        water_topology_list.append('{}          1\n'.format(ligand_name))
        with open(build_files_dict['ligand_top'], 'w+') as fh:
            fh.writelines(water_topology_list)
    else:
        os_util.local_print('Could not process topology file {} when editing it to prepare water perturbation. '
                            'Please, check file'.format(topology_file),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    # Saves a dual topology molecule PDB to file
    # FIXME: remove the need for setting molecule_name
    with open(build_files_dict['ligand_pdb'], 'w') as fh:
        fh.write(merge_topologies.dualmol_to_pdb_block(dual_molecule_ligand, molecule_name=ligand_name,
                                                       verbosity=verbosity))

    # Generates box around ligand and solvates it
    box_size = solvate_data['water_shell'] / 10.0
    editconf_list = ['editconf', '-f', build_files_dict['ligand_pdb'], '-d', str(box_size),
                     '-o', build_files_dict['ligandbox_pdb'], '-bt', solvate_data['box_type']]
    os_util.run_gmx(gmx_bin, editconf_list, '', build_files_dict['editconf_log'], verbosity=verbosity)

    # Solvate the ligand
    solvent_box = guess_water_box(None, build_files_dict['ligand_top'], verbosity=verbosity)
    shutil.copy2(build_files_dict['ligand_top'], build_files_dict['ligandsolv_top'])
    solvate_list = ['solvate', '-cp', build_files_dict['ligandbox_pdb'], '-cs', solvent_box,
                    '-o', build_files_dict['ligandsolv_pdb'], '-p', build_files_dict['ligandsolv_top']]
    os_util.run_gmx(gmx_bin, solvate_list, '', build_files_dict['solvate_log'], verbosity=verbosity,
                    alt_environment={'GMX_MAXBACKUP': '-1'})

    # See the same the comment in prepare_complex_system
    try:
        os.sync()
    except AttributeError:
        os_util.local_print('os.sync() not found. Is this a non-Unix system or Python version < 3.3?',
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    # Prepare a dummy mdp to run genion
    with open(build_files_dict['genion_mdp'], 'w') as genion_fh:
        genion_fh.write('include = -I{}\n'.format(build_water_dir))

    # Prepare a tpr to genion
    grompp_list = ['grompp', '-f', build_files_dict['genion_mdp'],
                   '-c', build_files_dict['ligandsolv_pdb'], '-p', build_files_dict['ligandsolv_top'],
                   '-o', build_files_dict['genion_tpr'], '-maxwarn', str(gmx_maxwarn),
                   '-po', build_files_dict['mdout_mdp']]
    os_util.run_gmx(gmx_bin, grompp_list, '', build_files_dict['grompp_log'], verbosity=verbosity)

    # Run genion
    shutil.copy2(build_files_dict['ligandsolv_top'], build_files_dict['fullsystem_top'])
    genion_list = ['genion', '-s', build_files_dict['genion_tpr'], '-p', build_files_dict['fullsystem_top'],
                   '-o', build_files_dict['fullsystem_pdb'], '-pname', solvate_data['pname'], '-nname',
                   solvate_data['nname'], '-conc', str(solvate_data['ion_concentration']), '-neutral']
    os_util.run_gmx(gmx_bin, genion_list, 'SOL\n', build_files_dict['genion_log'],
                    alt_environment={'GMX_MAXBACKUP': '-1'},
                    verbosity=verbosity)

    # Prepare index
    index_list = ['make_ndx', '-f', build_files_dict['fullsystem_pdb'], '-o', build_files_dict['index_ndx']]
    os_util.run_gmx(gmx_bin, index_list, input_data='q\n', output_file=build_files_dict['makendx_log'],
                    alt_environment={'GMX_MAXBACKUP': '-1'}, verbosity=verbosity)

    copywaterfiles = protein_topology_files + [build_files_dict['fullsystem_top'], build_files_dict['fullsystem_pdb'],
                                               build_files_dict['index_ndx']]

    for each_source in copywaterfiles:
        try:
            shutil.copy2(each_source, water_dir)
        except IsADirectoryError:
            try:
                shutil.copytree(each_source, os.path.join(water_dir, os.path.split(each_source)[-1]))
            except FileExistsError:
                pass
        except FileExistsError:
            pass
        os_util.local_print('Copying {} to {}'.format(each_source, water_dir),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    return all_classes.Namespace({'topology': build_files_dict['fullsystem_top'],
                                  'structure': build_files_dict['fullsystem_pdb']})


def create_fep_dirs(dir_name, base_pert_dir, new_files=None, extra_dir=None, extra_files=None, verbosity=0):
    """ Prepare directory structure for FEP

    :param str dir_name: name of the pair dir
    :param str base_pert_dir: root of perturbations
    :param dict new_files: a dict of the new files to be created in the perturbation dir in the form
                           {file_name: contents}
    :param list extra_dir: also copy contents of these dirs to the perturbation dir
    :param list extra_files: also copy these files to the perturbation dir
    :param int verbosity: verbosity level
    """

    os_util.local_print('Entering create_fep_dirs(dir_name=({}), base_pert_dir={}, new_files={}, extra_dir={}, '
                        'extra_files={}, verbosity={})'
                        ''.format(dir_name, base_pert_dir, ', '.join(['{}: {} lines'.format(k, v.count('\n'))
                                                                      for k, v in new_files.items()]),
                                  extra_dir, extra_files, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug,
                        current_verbosity=verbosity)

    # Create FEP directory structure
    dir_basename = os.path.join(base_pert_dir, dir_name)

    for each_dir in [dir_basename, os.path.join(dir_basename, 'water'), os.path.join(dir_basename, 'protein')]:
        os_util.makedir(each_dir, verbosity=verbosity)

    if new_files is not None:
        for each_file, contents in new_files.items():
            for each_dest in [os.path.join(dir_basename, 'water'), os.path.join(dir_basename, 'protein')]:
                with open(os.path.join(each_dest, each_file), 'w') as fh:
                    fh.write(contents)
                os_util.local_print('Creating file {} in {}'.format(each_file, each_dest),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if extra_dir:
        for each_dest in [os.path.join(dir_basename, 'water'), os.path.join(dir_basename, 'protein')]:
            for each_extra_dir in extra_dir:
                shutil.copytree(each_extra_dir, os.path.join(each_dest, os.path.split(each_extra_dir)[-1]))
                os_util.local_print('Copying directory {} to {}'.format(each_extra_dir, each_dest),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if extra_files:
        for each_source in extra_files:
            for each_dest in [os.path.join(dir_basename, 'water'), os.path.join(dir_basename, 'protein')]:
                shutil.copy2(each_source, each_dest)
                os_util.local_print('Copying file {} to {}'.format(each_source, each_dest),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    os_util.local_print('Done creating FEP dir {}'.format(dir_name), msg_verbosity=os_util.verbosity_level.info,
                        current_verbosity=verbosity)


def add_ligand_to_solvated_receptor(ligand_molecule, input_structure_file, output_structure_file, index_data,
                                    input_topology='', output_topology_file='', num_protein_chains=1, radius=3,
                                    selection_method='internal', ligand_name='LIG', verbosity=0):
    """ Adds the ligand ligand_molecule to the pre-solvated structure input_structure_file, relying in info from
    index_file. If provided, a topology file can be edited to reflect changes.

    :param merge_topologies.MergedTopologies ligand_molecule: molecule to be added
    :param str input_structure_file: pre-solvated structure pdb file
    :param str output_structure_file: saves full structure to this file
    :param dict index_data: Gromacs-style index file (used to mark last protein atom from output_structure_file)
    :param str input_topology: if provided, edits this topology to adjust the number of SOL molecules
    :param str output_topology_file: save topology to this file
    :param int num_protein_chains: number of protein chains (only used in case of input_topology != '')
    :param float radius: exclude water molecule this close to the ligand (default: 3.0 A)
    :param str selection_method: use mdanalysis, sklearn or internal (default) selection method
    :param str ligand_name: use this as ligand name
    :param int verbosity: verbosity level
    :rtype: bool
    """

    last_protein_atom = index_data['Protein'][-1]
    # FIXME: remove the need for setting molecule_name
    dual_topology_pdb = merge_topologies.dualmol_to_pdb_block(ligand_molecule, molecule_name=ligand_name,
                                                              verbosity=verbosity)

    # TODO: proper handling of the protein selection
    fullmd_data = os_util.read_file_to_buffer(input_structure_file, die_on_error=True, return_as_list=True,
                                              error_message='Failed to read input structure file.',
                                              verbosity=verbosity)
    for line_number, each_line in enumerate(fullmd_data[last_protein_atom:]):
        line_number += last_protein_atom
        if len(each_line) < 54:
            continue
        else:
            try:
                test_atom = int(each_line[6:11])
            except ValueError:
                continue
            else:
                if test_atom == last_protein_atom:
                    full_structure = fullmd_data[:line_number + 1]
                    os_util.local_print('Will insert ligand {} after atom {} of the system in file {}.'
                                        ''.format(ligand_molecule.dual_molecule_name, last_protein_atom,
                                                  input_structure_file),
                                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
                    break
    else:
        os_util.local_print('Could not find the end of the protein (atom {}) in file {}.'
                            ''.format(last_protein_atom, input_structure_file),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    for each_line in dual_topology_pdb.splitlines():
        if len(each_line) > 1 and each_line.split()[0] in ('ATOM', 'HETATM'):
            full_structure.extend('{}\n'.format(each_line))

    full_structure.extend(fullmd_data[line_number + 1:])
    with open(output_structure_file, 'w+') as outfile:
        outfile.writelines(full_structure)

    # Remove clashing waters
    if selection_method == 'internal':
        os_util.local_print('Internal selection method not implemented',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        # FIXME: implement this
        raise SystemExit(1)
    elif selection_method == 'sklearn':
        from sklearn.metrics import pairwise_distances_argmin_min
        import numpy

        # Reads data from pdb and ligand atom positions
        atom_dict_list = [{'line': line_number, 'atom_idx': int(line[6:11]), 'atom_name': int(line[12:16]),
                           'atom_position': [float(line[30:38]), float(line[38:46]), float(line[46:54])]}
                          for line_number, line in enumerate(input_structure_file)
                          if (len(line) > 54 and line[0:4] == 'ATOM' or line[0:6] == 'HETATM')]

        # Atom positions to a numpy array
        pdb_atom_positions = numpy.asarray([each_atom['atom_position'] for each_atom in atom_dict_list])
        lig_atom_positions = numpy.vstack([ligand_molecule.molecule_a.GetConformer().GetPositions(),
                                           ligand_molecule.molecule_b.GetConformer().GetPositions()])

        # Find distance between all ligand atoms to all atoms of the system
        closest, dist = pairwise_distances_argmin_min(pdb_atom_positions, lig_atom_positions)

        # Remove clashing waters
        clashing_waters = 0
        for each_atom, each_dist in closest[::-1], dist[::-1]:
            if dist < radius and atom_dict_list[each_atom]['atom_idx']:
                os_util.local_print('Atom {} would be removed. Atom data: {}'
                                    ''.format(each_atom, atom_dict_list[each_atom]),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        os_util.local_print('sklearn selection method not implemented',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
        # FIXME: implement this

    elif selection_method == 'mdanalysis':
        import MDAnalysis

        lig_name = 'UNL'
        fullmd_data = MDAnalysis.Universe(output_structure_file)

        # Count clashing waters
        clashing_waters = fullmd_data.select_atoms("(byres around {:1.3} resname {}) and (resname SOL)"
                                                   "".format(radius, lig_name)).n_residues

        # Construct the system without them
        new_system = fullmd_data.select_atoms("not ((byres around {:1.3} resname {}) and (resname SOL))"
                                              "".format(radius, lig_name))
        with MDAnalysis.Writer(output_structure_file) as file_writer:
            file_writer.write(new_system)

    # Edit topology to reflect the change in the number of water molecules
    with open(input_topology, 'r') as file_handler:
        topology_data = file_handler.readlines()

    new_topology_data = []
    mol_found = False
    prot_count = 0
    for line_number, each_line in enumerate(topology_data):
        if re.match(r"\[\s+molecules\s+].*", each_line) is not None:
            new_topology_data.append(each_line)
            mol_found = True
        elif mol_found:
            if (len(each_line) > 1) and (each_line[0] != ';'):
                try:
                    prot_count += int(each_line.split()[1])
                except TypeError:
                    os_util.local_print('Could not read line {} from file {}. This is the line text: {}'
                                        ''.format(line_number, each_line, input_topology),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(1)
                if prot_count == num_protein_chains:
                    new_topology_data.append(each_line)
                    new_topology_data.append('{}              1\n'.format(ligand_name))
                elif each_line.split()[0] == 'SOL':
                    new_water_count = int(each_line.split()[1]) - clashing_waters
                    new_topology_data.append('SOL              {}\n'.format(new_water_count))
                else:
                    new_topology_data.append(each_line)
            else:
                new_topology_data.append(each_line)
        else:
            new_topology_data.append(each_line)

    with open(output_topology_file, 'w+') as file_handler:
        file_handler.writelines(new_topology_data)

    os_util.local_print('Inserted ligand {} in system read from {}, removed {} waters and save the resulting system to '
                        '{} and topology to {}'.format(ligand_molecule.dual_molecule_name, input_structure_file,
                                                       clashing_waters, output_structure_file, output_topology_file),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)


def process_lambdas_input(lambda_input, no_checks=False, lambda_file=None, verbosity=0):
    """ Process filename input file and reads data into dict

    :param str lambda_input: name of the lambdas input file
    :param bool no_checks: ignore checks and keep going
    :param str lambda_file: read default lambda schemes from this file
    :param int verbosity: verbosity level
    :rtype: dict
    """

    os_util.local_print('Entering process_lambdas_input(lambda_input={}, no_checks={}, verbosity={})'
                        ''.format(lambda_input, no_checks, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if lambda_file is None:
        lambda_file = os.path.join('templates', 'lambdas.ini')

    default_lambdas = configparser.ConfigParser()
    default_lambdas.read(os.path.join(os.path.dirname(__file__), lambda_file))

    lambda_data = os_util.detect_type(lambda_input, test_for_dict=True, verbosity=verbosity)
    if isinstance(lambda_data, dict):
        # User supplied a dict
        pass
    elif isinstance(lambda_data, str):
        if lambda_data in default_lambdas['defaults']:
            lambda_data = os_util.detect_type(default_lambdas['defaults'][lambda_data], test_for_dict=True)
        else:
            # User supplied a filename, read its contents then try to process as dict
            file_data = os_util.read_file_to_buffer(lambda_data, die_on_error=True, return_as_list=False,
                                                    error_message='Failed to read lambdas file.', verbosity=verbosity)
            lambda_data = os_util.detect_type(file_data, test_for_dict=True, verbosity=verbosity)
    else:
        os_util.local_print('Failed to understand lambda_input data. Data read was: {} with type {}'
                            ''.format(arguments.lambda_input, type(arguments.lambda_input)),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    # Sanity check
    if not no_checks:
        for key, value_list in lambda_data.items():
            if key not in all_classes.DualTopologyData.allowed_scaling:
                os_util.local_print('Could not understand entry {} int lambda_input.\nThis is the data read:\n{}.'
                                    ''.format(key, lambda_data),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
            else:
                try:
                    if not all([0.0 <= float(each_value) <= 1.0 for each_value in value_list]):
                        os_util.local_print('Lambda value out of bounds [0.0, 1.0]. Please verify your input. Data '
                                            'read: {}: {}'.format(key, value_list),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise SystemExit(1)

                except (ValueError, TypeError):
                    os_util.local_print('Failed to parse line {} in lambda_input. This is the data read'
                                        ''.format(value_list, lambda_data),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(1)

        if len(set(map(len, list(lambda_data.values())))) > 1:
            os_util.local_print('Inconsistent lambda count. This is the lambda data read:\n{}'
                                ''.format(lambda_data),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

    os_util.local_print('This is the lambda data:\n{}'.format(lambda_data),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    return lambda_data


def edit_mdp_prepare_rerun(mdp_input, mdp_outfile, verbosity=0):
    """ Edit mdp to prepare a rerun tpr

    :param str mdp_input: name of the input file
    :param str mdp_outfile: name of the tpr output file
    :param int verbosity: sets the verbosity level
    """

    edited_mdp = []
    read_data = os_util.read_file_to_buffer(mdp_input, die_on_error=True, return_as_list=True,
                                            error_message='Failed to read mdp file when preparing a mdp for rerun.',
                                            verbosity=verbosity)
    for each_line in read_data:
        if each_line.startswith('continuation'):
            edited_mdp.append('continuation        = no\n')
        elif each_line.startswith('nstvout'):
            edited_mdp.append('nstvout             = 0\n')
        elif each_line.startswith('nstxout'):
            edited_mdp.append('nstxout             = 0\n')
        elif each_line.startswith('nstlog'):
            edited_mdp.append('nstlog              = 0\n')
        else:
            edited_mdp.append(each_line)
    with open(mdp_outfile, 'w+') as file_handler:
        file_handler.writelines(edited_mdp)


def do_solute_scaling(solute_scaling_file, scale_value, scaled_atoms, scaling_bin=None, verbosity=0):
    """ Uses plumed to apply solute scaling

    :param str solute_scaling_file: apply solute scaling to this file (note: will edit in place)
    :param float scale_value: scale the dihedral by this amount
    :param dict scaled_atoms: a dict containing information on atoms to be scaled
    :param str scaling_bin: use this rest2 binary
    :param int verbosity: control verbosity level
    :rtype: bool
    """
    os_util.local_print('Entering do_solute_scaling: solute_scaling_file={}, scale_value={}, scaled_atoms={}, '
                        'scaling_bin={}, verbosity={}'
                        ''.format(solute_scaling_file, scale_value, scaled_atoms, scaling_bin, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if not scaling_bin:
        scaling_bin = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tools', 'rest2.sh')

    input_data = os_util.read_file_to_buffer(solute_scaling_file, return_as_list=True, die_on_error=True,
                                             error_message='Could not read solute scaling file.', verbosity=verbosity)

    original_topology_file = os.path.join(os.path.dirname(solute_scaling_file), 'before_solute_scaling.top')
    shutil.copy2(solute_scaling_file, original_topology_file)

    # Parse topology file, look for atoms to be scaled by REST2 and add and _ after their name
    altered_atom_count = 0
    altered_topology = []
    actual_molecule = ''
    for line_number, each_line in enumerate(input_data):
        if each_line[0] == ';' or each_line[0] == '\n':
            altered_topology.append(each_line)
        elif re.match(r'\[\s+(.+)\s+\]', each_line) is not None:
            altered_topology.append(each_line)
            matched_str = re.match(r'\[\s+(.+)\s+\]', each_line).groups()[0]
            if matched_str == 'moleculetype':
                # Flag to search for molecule name
                actual_molecule = None
            elif matched_str == 'atoms':
                pass
            else:
                # Just keep ignoring everything
                actual_molecule = ''
        elif actual_molecule is None and each_line.split()[0] in scaled_atoms:
            actual_molecule = each_line.split()[0]
            altered_topology.append(each_line)
        elif actual_molecule:
            if len(each_line) < 3 and each_line[0] != ';':
                continue
            # We are reading the atoms of scaled molecule, check if they should be scaled
            try:
                atom_number, atom_type, atom_resno, atom_resnm, atom_name, complement_str = each_line.split(None, 5)
            except ValueError as errordata:
                os_util.local_print('Could not parse line {} with error {}. Line content is:\n\t{}'
                                    ''.format(line_number, errordata, each_line),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                raise SystemExit(1)
            else:
                if int(atom_number) in scaled_atoms[actual_molecule]:
                    altered_topology.append('{:<3} {:<15} {:<5} {:<10} {:<20} {} ; REST2 scaled atom\n'
                                            ''.format(atom_number, '{}_'.format(atom_type), atom_resno, atom_resnm,
                                                      atom_name, complement_str.replace('\n', ' ')))
                    altered_atom_count += 1
                else:
                    for each_pattern in scaled_atoms[actual_molecule]:
                        if isinstance(each_pattern, int):
                            continue
                        if re.search(each_pattern, atom_name) is not None:
                            altered_topology.append('{:<3} {:<15} {:<5} {:<10} {:<20} {} ; REST2 scaled atom\n'
                                                    ''.format(atom_number, '{}_'.format(atom_type), atom_resno,
                                                              atom_resnm, atom_name,
                                                              complement_str.replace('\n', ' ')))
                            altered_atom_count += 1
                            break
                    else:
                        altered_topology.append(each_line)

        else:
            altered_topology.append(each_line)

    os_util.local_print('{} atoms are subject to solute scaling'.format(altered_atom_count),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    altered_topology = ''.join(altered_topology)

    if verbosity >= os_util.verbosity_level.debug:
        altered_topology_file = '{}_altered_topology_backup{}'.format(*os.path.splitext(solute_scaling_file))
        os_util.local_print('Saving a copy of edited topology submitted to REST bin {} in file {}'
                            ''.format(scaling_bin, altered_topology_file),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        with open(altered_topology_file, 'w') as fh:
            fh.write(altered_topology)

    # Now call plumed and supply it the edited file
    process_handler = subprocess.Popen([scaling_bin, str(scale_value)],
                                       stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                       bufsize=-1, universal_newlines=True)
    scaled_output, error_log = process_handler.communicate(altered_topology)

    if error_log != '':
        os_util.local_print('Plumed failed with the message:\n{}'.format(error_log),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    else:
        # Save scaled topology to solute_scaling_file
        try:
            with open(solute_scaling_file, 'w') as output_file:
                output_file.write(scaled_output)
        except IOError as errordata:
            os_util.local_print('Could not write to file {}. Error was: {}'.format(solute_scaling_file, errordata),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        else:
            return True


def generate_scaling_vector(max_val, min_val, nsteps):
    """ Uses plumed to apply solute scaling

    :param float max_val: maximum value (usually, max_val = 1)
    :param float min_val: save result to this file
    :param int nsteps: number of steps
    :rtype: list
    """

    from numpy import linspace, concatenate
    if nsteps % 2 == 0:
        this_scalingvector = linspace(max_val, min_val, int(nsteps / 2))
        this_scalingvector = concatenate([this_scalingvector, this_scalingvector[::-1]])
    else:
        this_scalingvector = linspace(max_val, min_val, int(nsteps / 2) + 1)
        this_scalingvector = concatenate([this_scalingvector, this_scalingvector[-2::-1]])

    return this_scalingvector


def process_scaling_input(input_data, verbosity=0):
    """ Reads a file with atoms to be scaled

    :param str input_data: file te be read
    :param int verbosity: sets the verbosity level
    :rtype: dict
    """

    os_util.local_print('Entering process_scaling_input(input_data={}, verbosity={})'.format(input_data, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    scaling_data = os_util.detect_type(input_data, test_for_dict=True, verbosity=verbosity)
    if isinstance(input_data, dict):
        # User supplied a dict
        pass
    elif isinstance(input_data, str):
        # User supplied a filename, read its contents then try to process as dict
        file_data = os_util.read_file_to_buffer(input_data, die_on_error=True, return_as_list=False,
                                                error_message='Failed to read scaling selection file.'
                                                              '', verbosity=verbosity)
        scaling_data = os_util.detect_type(file_data, test_for_dict=True, verbosity=verbosity)
    else:
        os_util.local_print('Failed to understand scaling selection. Scaling data read was: {} with type {}'
                            ''.format(arguments.lambda_input, type(arguments.lambda_input)),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    for molecule_name, selection_data in scaling_data.items():
        # Try to interpret each value as an int (ie: atom idx) or a str (to be regex in atom name)
        scaling_data[molecule_name] = []
        if isinstance(selection_data, str):
            selection_data = selection_data.split()
        for each_atom in selection_data:
            try:
                scaling_data[molecule_name].append(int(each_atom))
            except ValueError:
                try:
                    scaling_data[molecule_name].append(re.compile(each_atom))
                except re.error:
                    os_util.local_print('Could not parse entry {} for molecule type {} when reading scaling '
                                        'selection data.'
                                        ''.format(each_atom, molecule_name),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(1)

    return scaling_data


def prepare_steps(lambda_value, lambda_dir, dir_list, topology_file, structure_file, index_file='index.ndx',
                  solute_scaling_list=None, solute_scaling_atoms_dict=None, scaling_bin=None, plumed_conf=None,
                  local_gmx_path='gmx', gmx_path='gmx', gmx_maxwarn=1, verbosity=0):
    """ Run Gromacs to minimize and equilibrate a system, if required, apply solute scaling

    :param int lambda_value: lambda value to be used in FEP
    :param str lambda_dir: prepare run in this directory
    :param list dir_list: list of directories to use to minimize and equilibrate
    :param str topology_file: Gromacs-compatible topology file
    :param str structure_file: Gromacs-compatible structure file
    :param str index_file: Gromacs-compatible index file (.ndx)
    :param [list, numpy.array] solute_scaling_list: a list of scaling values along the lambda windows
    :param dict solute_scaling_atoms_dict: selection of atoms to be scaled
    :param str scaling_bin: use this executable to apply solute scaling
    :param str: use data this as plumed configuration file
    :param str local_gmx_path: path to Gromacs binary in the current machine (default: gmx)
    :param str gmx_path: path to Gromacs binary (default: gmx)
    :param int gmx_maxwarn: pass this maxmarn to Gromacs
    :param int verbosity: control verbosity level
    """

    os_util.local_print('Entering prepare_steps(lambda_value={}, lambda_dir={}, dir_list={}, topology_file={}, '
                        'structure_file={}, index_file={}, solute_scaling_list={}, solute_scaling_atoms_dict={}, '
                        'gmx_path={}, gmx_maxwarn={}, verbosity={})'
                        ''.format(lambda_value, lambda_dir, dir_list, topology_file, structure_file, index_file,
                                  solute_scaling_list, solute_scaling_atoms_dict, gmx_path, gmx_maxwarn, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if lambda_dir.endswith(os.sep):
        lambda_dir = lambda_dir[:-1]

    file_names = {}

    # Lambda windows such as 0.0 < l < 1.0 may have non-integer total system charges, so we maxwarn += 1.
    # Currently, the atom names from PDB will be trunked in respect of the atom names in topology an this will also
    # generate a warning, so maxwarn += 1.
    # TODO: fix the atom naming problem; will require another (not PDB) file format
    gmx_maxwarn += 2

    if solute_scaling_list is not None:
        file_names['grompp_ss_log'] = os.path.join(lambda_dir, 'sulutescaling_grompp.log')
        file_names['top'] = os.path.join(lambda_dir, topology_file)
        file_names['backup_topology'] = os.path.join(lambda_dir, 'backup_topology.top')
        file_names['original_topology'] = os.path.join(lambda_dir, '..', topology_file)

        # Replace the symlink to the topology by an actual file
        if os.path.islink(file_names['top']):
            os.unlink(file_names['top'])
            shutil.copy2(os.path.join(lambda_dir, '..', topology_file), file_names['top'])
        else:
            shutil.copy2(os.path.join(lambda_dir, '..', topology_file), file_names['top'])

        # Prepare a scaled topology
        temporary_mdp = os.path.join(lambda_dir, 'dummy.mdp')
        temporary_tpr = os.path.join(lambda_dir, 'dummy.tpr')
        shutil.copy2(file_names['top'], file_names['backup_topology'])
        input_mdp = os.path.join(dir_list[0], '{}.mdp'.format(os.path.basename(dir_list[0])))
        file_names['local_structure_file'] = os.path.join(lambda_dir, structure_file)

        input_list = ["grompp", "-c", file_names['local_structure_file'], "-r", file_names['local_structure_file'],
                      "-f", input_mdp, "-p", file_names['backup_topology'], "-pp", file_names['top'],
                      '-n', os.path.join(lambda_dir, index_file), '-maxwarn', str(gmx_maxwarn), '-o', temporary_tpr,
                      '-po', temporary_mdp]

        os_util.run_gmx(local_gmx_path, input_list, output_file=file_names['grompp_ss_log'], verbosity=verbosity,
                        alt_environment={'GMX_MAXBACKUP': '-1'})
        os.remove(temporary_mdp)
        os.remove(temporary_tpr)
        # Edit atoms in solute_scaling_dict, call plumed and save to file_names['top']
        do_solute_scaling(file_names['top'], solute_scaling_list[lambda_value], solute_scaling_atoms_dict,
                          scaling_bin=scaling_bin, verbosity=verbosity)

    output_data = []
    last_files_names = None
    # Iterate over calculation directories
    for index, each_dir in enumerate(dir_list):
        if os.path.basename(each_dir) == 'md':
            continue
        this_dir = os.path.relpath(each_dir, os.path.join(lambda_dir, '..', '..', '..'))
        file_names.update({key: '{}.{}'.format(os.path.join(this_dir, os.path.basename(this_dir)), key)
                           for key in ['gro', 'cpt', 'tpr', 'mdp']})
        file_names['mdout'] = '{}_mdout.mdp'.format(os.path.join(this_dir, os.path.basename(this_dir)),
                                                    os.path.basename(this_dir))
        file_names['mdrun_log'] = '{}_mdrun_log.log'.format(os.path.basename(this_dir))
        file_names['grompp_log'] = '{}_grompp.log'.format(os.path.join(this_dir, os.path.basename(this_dir)),
                                                          os.path.basename(this_dir))

        if index == 0 and last_files_names is None:
            # First minimization, use structure from input
            last_structure = os.path.join(this_dir, '..', structure_file)
        elif last_files_names is not None:
            # Subsequent minimization or equilibration, use structure from last cycle
            last_structure = last_files_names['gro']
        else:
            os_util.local_print('Could not find info from last step. This is step #{}, and the sequence is {}'
                                ''.format(index, dir_list),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        # Run gmx mdrun to prepare tpr for minimization/equilibration step
        input_list = ["grompp", "-c", last_structure, "-r", last_structure, "-f", file_names['mdp'], "-p",
                      os.path.join(this_dir, '..', topology_file), "-o", file_names['tpr'],
                      '-n', os.path.join(this_dir, '..', index_file), '-po', file_names['mdout'],
                      '-maxwarn', str(gmx_maxwarn)]
        grompp_cmd = os_util.assemble_shell_command(gmx_path, input_list, output_file=file_names['grompp_log'],
                                                    verbosity=verbosity)
        grompp_cmd = '[ ! -f {} ] && {{ {} }}'.format(file_names['tpr'], grompp_cmd)
        output_data.append(grompp_cmd)

        # Then run gmx mdrun
        input_list = ["mdrun", "--deffnm", os.path.basename(each_dir)]
        mdrun_cmd = os_util.assemble_shell_command(gmx_path, input_list, output_file=file_names['mdrun_log'],
                                                   verbosity=verbosity, cwd=this_dir)
        mdrun_cmd = '[ ! -f {}.gro ] && {{ {} }}'.format(os.path.join(this_dir, os.path.basename(each_dir)), mdrun_cmd)

        output_data.append(mdrun_cmd)
        last_files_names = file_names.copy()

    os_util.makedir(os.path.join(lambda_dir, 'md'), verbosity=verbosity)
    os_util.makedir(os.path.join(lambda_dir, '..', 'md'), verbosity=verbosity)
    md_run_dir = os.path.join(lambda_dir, '..', 'md', os.path.basename(lambda_dir))
    os_util.makedir(md_run_dir, verbosity=verbosity)

    if plumed_conf is None:
        plumed_conf = ''
    with open(os.path.join(md_run_dir, 'plumed.dat'), 'w+') as plumed_file:
        # Prepare a dummy plumed file
        plumed_file.write(plumed_conf)

    md_dir = os.path.relpath(os.path.join(lambda_dir, 'md'), os.path.join(lambda_dir, '..', '..'))

    new_file_names_dict = {'md_nocont': os.path.join(morph_dir, md_dir, 'md_nocont.mdp'),
                           'md': os.path.join(morph_dir, md_dir, 'md.mdp'),
                           'md_nocont_tpr': os.path.join(morph_dir, md_dir, 'md_nocont.tpr'),
                           'md_tpr': os.path.join(morph_dir, md_dir, 'md.tpr'),
                           'mdout_nocont_mdp': os.path.join(morph_dir, md_dir, 'mdout_nocont.mdp'),
                           'mdout_mdp': os.path.join(morph_dir, md_dir, 'mdout.mdp'),
                           'grompp_log': os.path.join(morph_dir, md_dir, 'mdout.log'),
                           'grompp_nocont_log': os.path.join(morph_dir, md_dir, 'mdout_nocont.log')}
    edit_mdp_prepare_rerun(os.path.join(os.path.join(lambda_dir, 'md'), 'md.mdp'),
                           os.path.join(os.path.join(lambda_dir, 'md'), 'md_nocont.mdp'))

    # Run gmx grompp to prepare the final run tpr
    input_list = ["grompp", "-c", last_files_names['gro'],
                  "-r", last_files_names['gro'],
                  "-t", last_files_names['cpt'],
                  "-f", new_file_names_dict['md_nocont'],
                  '-po', new_file_names_dict['mdout_nocont_mdp'],
                  "-o", new_file_names_dict['md_nocont_tpr'],
                  '-n', os.path.join(morph_dir, md_dir, '..', '..', index_file),
                  "-p", os.path.join(morph_dir, md_dir, '..', topology_file),
                  '-maxwarn', str(gmx_maxwarn)]
    grompp_cmd = os_util.assemble_shell_command(gmx_path, input_list,
                                                output_file=new_file_names_dict['grompp_nocont_log'],
                                                verbosity=verbosity)
    grompp_cmd = '[ ! -f {} ] && {{ {} }}'.format(new_file_names_dict['md_nocont_tpr'], grompp_cmd)
    output_data.append(grompp_cmd)

    input_list = ["grompp", "-c", last_files_names['gro'],
                  "-r", last_files_names['gro'],
                  "-t", last_files_names['cpt'],
                  "-f", new_file_names_dict['md'],
                  "-p", os.path.join(morph_dir, md_dir, '..', topology_file),
                  "-o", new_file_names_dict['md_tpr'],
                  '-n', os.path.join(morph_dir, md_dir, '..', '..', index_file),
                  '-po', new_file_names_dict['mdout_mdp'],
                  '-maxwarn', str(gmx_maxwarn)]
    grompp_cmd = os_util.assemble_shell_command(gmx_path, input_list,
                                                output_file=new_file_names_dict['grompp_log'],
                                                verbosity=verbosity)
    grompp_cmd = '[ ! -f {} ] && {{ {} }}'.format(new_file_names_dict['md_tpr'], grompp_cmd)
    output_data.append(grompp_cmd)

    mdrun_dir = os.path.relpath(os.path.join(lambda_dir, '..'), os.path.join(lambda_dir, '..', '..', '..'))
    mdrun_dir = os.path.join(mdrun_dir, 'md', os.path.basename(lambda_dir))

    output_data.append('cp {} {}'.format(new_file_names_dict['md_tpr'],
                                         os.path.join(mdrun_dir, 'lambda.tpr')))
    output_data.append('cp {} {}'.format(new_file_names_dict['md_nocont_tpr'],
                                         os.path.join(mdrun_dir, 'lambda_nocont.tpr')))
    output_data.append('')

    return '\n'.join(output_data)


def replace_multi_value_mdp(substitutions, mdp_data, no_checks=False, verbosity=0):
    """ Reads mdp data as list and substitute values where multiple values may be expected

    :param dict substitutions: group substitutions to apply to mdp_data
    :param list mdp_data: mdp data to be edited
    :param bool no_checks: ignore checks and try to go on
    :param int verbosity: set verbosity
    :rtype: list
    """

    substitutions_groups = [('_TEMPERATURE', 'ref-t', 'tc-grps'), ('_PRESSURE', 'ref-p', 'pcoupltype')]
    pressure_coulpe_num = {'isotropic': 1, 'semiisotropic': 2, 'anisotropic': 6, 'surface-tension': 2}

    for (alias_option, mdp_option, group_option) in substitutions_groups:
        if alias_option in substitutions:
            if mdp_option in substitutions:
                if no_checks:
                    os_util.local_print('You used no_checks, so I am going on using temperature argument and ignoring '
                                        '"{}"'.format(mdp_option),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                else:
                    os_util.local_print('You supplied both "{}" and a temperature argument. Select one - or use '
                                        'no_checks, so I will ignore "{}"'.format(mdp_option, mdp_option),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(-1)

            try:
                group_line = [each_line.split(';')[0].split('=')[1].split()
                              for each_line in mdp_data if each_line.lstrip().startswith(group_option)]
            except IndexError:
                os_util.local_print('Failed to parse {} data from mdp file. Please, check the mdp file.'
                                    ''.format(group_option),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise ValueError('Failed to parse mdp option {}'.format(group_option))
            else:
                if alias_option == '_PRESSURE':
                    num_tc_groups = pressure_coulpe_num.setdefault(group_line[0][0], False)
                    if not num_tc_groups:
                        os_util.local_print('Failed to parse {} data from mdp file. Please, check the mdp file.'
                                            ''.format(group_option),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise ValueError('Failed to parse mdp option {}'.format(group_option))
                else:
                    if group_line:
                        num_tc_groups = len([each_group for each_group in group_line[0] if each_group])
                    else:
                        num_tc_groups = 0

            if num_tc_groups:
                substitutions.update({mdp_option: '{} '.format(substitutions[alias_option]) * num_tc_groups})
    return mdp_data


def edit_mdp_file(mdp_file, substitutions, outfile=None, no_checks=False, verbosity=0):
    """ Edits and mdp file applying substitutions

    :param str mdp_file: file to be edited
    :param dict substitutions: perform this substitutions
    :param str outfile: save modified version to this file (default: overwrite mdp_file)
    :param bool no_checks: ignore checks and try to go on
    :param int verbosity: verbosity level
    """

    os_util.local_print('Entering edit_mdp_file(mdp_file={}, substitutions={}, outfile={}, verbosity={})'
                        ''.format(mdp_file, substitutions, outfile, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if not outfile:
        outfile = mdp_file
        os_util.local_print('No outfile was suplied to edit_mdp_file. I will edit {} inplace.'.format(outfile),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    mdp_data = os_util.read_file_to_buffer(mdp_file, die_on_error=True, return_as_list=True,
                                           error_message='Could not read mdp file.', verbosity=verbosity)

    mdp_data = replace_multi_value_mdp(substitutions, mdp_data, no_checks=no_checks, verbosity=verbosity)

    dt = None
    for index, each_line in enumerate(mdp_data[:]):
        if not each_line or each_line[0] == ';' or each_line.find('=') == -1:
            continue

        directive, value = each_line.split(';', 1)[0].split('=')
        directive = directive.lstrip().rstrip()
        if directive in substitutions:
            mdp_data[index] = '{} = {}         ; Edited by PyAutoFEP\n'.format(directive, substitutions[directive])
        if directive == 'dt':
            dt = float(value)

    if '_LENGTH' in substitutions:
        nsteps = int(substitutions['_LENGTH'] / dt)

        for index, each_line in enumerate(mdp_data[:]):
            if not each_line or each_line[0] == ';' or each_line.find('=') == -1:
                continue

            directive, value = each_line.split(';', 1)[0].split('=')
            directive = directive.lstrip().rstrip()
            if directive == 'nsteps':
                mdp_data[index] = '{} = {}         ; Edited by PyAutoFEP\n'.format(directive, nsteps)

    try:
        with open(outfile, 'w') as fh:
            fh.writelines(mdp_data)
    except IOError:
        os_util.local_print('Failed to write to file {} after editing mdp. Cannot continue.'.format(outfile),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)


def process_perturbation_map(perturbation_map_input, verbosity=0):
    """ Reads perturbation data input

    :param str perturbation_map_input: pertubation data
    :param int verbosity: sets verbosity level
    :return: dict
    """

    perturbation_map_input = os_util.detect_type(perturbation_map_input, test_for_dict=True, test_for_list=True)
    if isinstance(perturbation_map_input, dict):
        pass
    elif isinstance(perturbation_map_input, str):
        file_name = perturbation_map_input
        file_data = os_util.read_file_to_buffer(file_name, die_on_error=False, return_as_list=False,
                                                error_message='Failed to read perturbations map file.',
                                                verbosity=verbosity)
        typed_data = os_util.detect_type(file_data, test_for_dict=True, test_for_list=True)
        if not typed_data:
            typed_data = perturbation_map_input

        if isinstance(typed_data, dict):
            # Read a dict from file
            perturbation_map_input = typed_data
        else:
            # Process lines
            perturbation_map_input = OrderedDict()
            for each_line in typed_data.split(os.linesep):
                if len(each_line) < 1 or each_line[0] in ['#', ';'] or each_line.split() == []:
                    continue

                each_line = each_line.split()
                if len(each_line) == 2:
                    perturbation_map_input[(each_line[0], each_line[1])] = {}
                elif len(each_line) > 2:
                    this_data = os_util.detect_type(each_line[2], test_for_dict=True)
                    if isinstance(this_data, dict):
                        perturbation_map_input[(each_line[0], each_line[1])] = this_data
                    elif isinstance(this_data, str):
                        # Default is read only the lambda scheme
                        perturbation_map_input[(each_line[0], each_line[1])] = {"lambda": this_data}
                else:
                    os_util.local_print('Failed to read perturbation_map from file {}. Line "{}" was not understood'
                                        ''.format(file_name, each_line),
                                        current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
                    raise ValueError('At least 2 fields are required'.format(type(perturbation_map_input)))

    else:
        os_util.local_print('Failed to read perturbation_map as dict or string (or None or False, ie: '
                            'read from save state data). Value "{}" was read as a(n) {}'
                            ''.format(perturbation_map_input, type(perturbation_map_input)),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise TypeError('dict, False, or NoneType expected, got {} instead'
                        ''.format(type(perturbation_map_input)))

    perturbation_map = OrderedDict()
    for k, v in perturbation_map_input.items():
        if isinstance(k, str) and isinstance(v, str):
            # Entry is in format 'mol_a': 'mol_b', convert it
            perturbation_map[(k, v)] = {}
        elif isinstance(k, tuple):
            # Entry is in format ('mol_a', 'mol_b'): {} or ('mol_a', 'mol_b'): lambda_XX
            if isinstance(v, dict):
                perturbation_map[k] = v
            else:
                perturbation_map[k] = {'lambda': v}
        else:
            os_util.local_print('Failed to understand perturbation data in {}. Entry {} could not be understood. '
                                'Either a str or tuple was expected, but got a {}'
                                ''.format(perturbation_map_input, k, type(k)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise TypeError('str or tuple expected, got {} instead'.format(type(k)))

    return perturbation_map


if __name__ == '__main__':
    Parser = argparse.ArgumentParser(description='Reads multiple topology and molecule files and a perturbation map, '
                                                 'then generates the perturbation inputs.')
    Parser.add_argument('--topology', type=str, default=None,
                        help='Full system topology (default: FullSystem.top)')
    Parser.add_argument('--structure', type=str, default=None,
                        help='Solvated and equilibrated system or receptor structure')
    Parser.add_argument('--index', type=str, default=None,
                        help='Index file to be created (if not --pre_solvated) or read (if --pre_solvated)')
    Parser.add_argument('--index_string', type=str, default=None, help='Create this/these index group(s)')
    Parser.add_argument('--selection_method', choices=['mdanalysis', 'internal'], default=None,
                        help='Method to be used in the selection of water molecules (internal (default), mdanalysis)')
    Parser.add_argument('--input_ligands', type=str, default=None,
                        help='One of the following: (a) a file containing ligands names, mol2 and topology files; (b) '
                             'a dict structure containing ligands names, mol2 and topology files; (c) path to a '
                             'directory containing files named ligand_name.mol2 and ligand_name.top. I will also try '
                             'to read data absent from this option (and from possible configuration file) from '
                             'progress file.')
    Parser.add_argument('--perturbation_map', type=str, default=None,
                        help='Perturbation map as a dictionary or a text file')
    Parser.add_argument('--perturbations_dir', type=str, default=None,
                        help='Generate perturbation directories to this dir (default: automatically create a directory '
                             'name)')
    Parser.add_argument('--extrafiles', default=None, nargs='*', help='Copy these files to perturbation directories')
    Parser.add_argument('--extradirs', nargs='*', default=None,
                        help='Recursively copy contents of these directories to perturbation directories')

    gmx_opt = Parser.add_argument_group('Gromacs options', 'Options to control Gromacs invoking.')
    gmx_opt.add_argument('--gmx_maxwarn', type=int, default=None, help='Pass this maxwarn to gmx (Default: 1)')
    gmx_opt.add_argument('--gmx_bin_local', type=str, default=None,
                         help='Use this local Gromacs binary to prepare the system (Default: gmx)')
    gmx_opt.add_argument('--gmx_bin_run', type=str, default=None,
                         help='Use this Gromacs binary to run the MD. This should be the Gromacs bin in the run node,'
                              'not in the current machine. (Default: gmx)')

    mcs_opt = Parser.add_argument_group('MCS options', 'Options to control MCS between molecules')
    Parser.add_argument('--mcs_custom_mcs', type=str, default=None,
                        help='Use this/these custom MCS between pairs. Can be either a string (so the same MCS '
                             'will be used for all pairs) or a dictionary (only pairs present in dictionary will use '
                             'a custom MCS)')
    # TODO: implement this
    # Parser.add_argument('--mcs_custom_atommap', type=str, default=None,
    #                     help='Use these custom atoms map between pairs.')

    poses_loader = Parser.add_argument_group('poses_loader', 'Options related to the loading of initial ligand poses')
    poses_loader.add_argument('--poses_input', type=str, default=None,
                              help='Poses data to be read, if needed. See documentation.')
    poses_loader.add_argument('--pose_loader', choices=['generic', 'pdb', 'superimpose', 'autodock4'], default=None,
                              help='Select poses format to read. See manual for further details.')
    poses_loader.add_argument('--poses_reference_structure', type=str, default=None,
                              help='Reference poses structure, eg: docking receptor. Not used for pdb loader. If not '
                                   'supplied, the macromolecule from structure argument or option will be used.')
    poses_loader.add_argument('--poses_reference_pose_superimpose', type=str, default=None,
                              help='Reference poses for superimposition. Uses only if pose_loader = "superimpose".')
    poses_loader.add_argument('--poses_advanced_options', type=str, default=None,
                              help='Advanced options to poses loader. See documentation for further info.')

    build_system = Parser.add_argument_group('build_system', 'Options used to build the system, if --pre_solvated is '
                                                             'not used')
    build_system.add_argument('--buildsys_forcefield', type=int, default=None,
                              help='Value to be passed to  pdb2gmx (Default: 1, AMBER03; this option will be passed '
                                   'to gmx pdb2gmx; you can use a custom FF by passing 1 here (default) and adding a '
                                   'FF directoty in --extradirs)')
    build_system.add_argument('--buildsys_water', default=None,
                              choices=['default', 'spc', 'spce', 'tip3p', 'tip4p', 'tip5p', 'tips3p'],
                              help='Use this water model. If default, "1" will be supplied to pdb2gmx, selecting FF '
                                   'default water model, otherwise the chosen name will be passed as -water option to '
                                   'pdb2gmx.')
    build_system.add_argument('--buildsys_watershell', type=float, default=None,
                              help='Distance, in Angstroms, between the solute and the box (Default: 10 A)')
    build_system.add_argument('--buildsys_boxtype', choices=['triclinic', 'cubic', 'dodecahedron', 'octahedron'],
                              default=None, help='Box type used to build the complex and water systems (Default: '
                                                 'dodecahedron)')
    build_system.add_argument('--buildsys_ionconcentration', type=float, default=None,
                              help='Concentration, in mol/L, of added ions (Default: 0.15 mol/L)')
    build_system.add_argument('--buildsys_nname', type=str, default=None,
                              help='Name of the positive ion (Default: CL)')
    build_system.add_argument('--buildsys_pname', type=str, default=None,
                              help='Name of the negative ion (Default: NA)')

    presolvated_system = Parser.add_argument_group('pre_solvated', 'Options if a pre-solvated system is supplied '
                                                                   '(ie: --pre_solvated is used)')
    presolvated_system.add_argument('--pre_solvated', action='store_const', const=True,
                                    help='Use a pre-solvated structure (Default: no)')
    presolvated_system.add_argument('--presolvated_radius', type=float, default=None,
                                    help='Remove water molecules this close, in Angstroms, to ligand (Default: 1.0 A)')
    presolvated_system.add_argument('--presolvated_protein_chains', type=int, default=None,
                                    help='Number of distinct protein chains')

    perturbation_opts = Parser.add_argument_group('perturbation_options',
                                                  'Options used to control the perturbation scheme')
    perturbation_opts.add_argument('--lambda_input', type=str, default=None,
                                   help='One of the preset lambda schemes (Default: lambda12) or a dictionary of '
                                        'lambda values or input file containing lambda values')
    perturbation_opts.add_argument('--template_mdp', type=str, default=None,
                                   choices=['default_nosc', 'charmm_nosc'],
                                   help='Use this default mdp template files. (Default: default_nosc) ')
    perturbation_opts.add_argument('--complex_mdp', nargs='*', default=None,
                                   help='List of mdp files or file containing the mdp files for complex perturbation')
    perturbation_opts.add_argument('--water_mdp', nargs='*', default=None,
                                   help='List of mdp files or file containing the mdp files for water perturbation')
    perturbation_opts.add_argument('--mdp_substitution', type=str, default=None,
                                   help='Substitute these values on mdp files')

    # TODO: fix the softcore code
    # perturbation_opts.add_argument('--perturbations_softcore', default=None, type=str,
    #                                choices=['off', 'maximum', 'average', 'minimum'],
    #                                help='Use soft-core for Van de Waals potential on the ligand. Selects the function '
    #                                     'used to calculate the effective soft-core lambda. (Default: no, do not use '
    #                                     'soft core)')

    ee_opts = Parser.add_argument_group('enhanced_sampling',
                                        'Options used to control solute tempering/scaling and HREX')
    ee_opts.add_argument('--solute_scaling', type=float, default=None,
                         help='Apply solute scaling (Default: off)')
    ee_opts.add_argument('--solute_scaling_selection', type=str, default=None,
                         help='File containing the atoms to be selected for solute scaling; or dict containing the '
                              'same information')
    ee_opts.add_argument('--solute_scaling_bin', type=str, default=None,
                         help='Path to the a executable capable of applying solute tempering/ scaling on a Gromacs '
                              'topology (Default: use internal (REST2) script')
    ee_opts.add_argument('--hrex', type=int, default=None,
                         help='Attempt frequency of Hamiltonian Replica Exchange (requires a Plumed-patched Gromacs '
                              'binary; default: 0: no HREX, any integer > 0: replica-exchange frequency)')
    ee_opts.add_argument('--plumed_conf', type=str, default=None,
                         help='Custom plumed config file (.dat, see plumed manual)')

    md_opts = Parser.add_argument_group('md_configuration', 'Options used to control the MD sampling')
    md_opts.add_argument('--md_temperature', type=float, default=None,
                         help='Absolute temperature of MD. (Default 298.15).')
    md_opts.add_argument('--md_length', type=float, default=None,
                         help='Simulation length, in ps. (Default: 5000.0 ps = 5.0 ns)')

    ouput_opts = Parser.add_argument_group('Output configuration',
                                           'Options used to control the output scripts and directories')
    ouput_opts.add_argument('--output_packing', choices=['bin', 'tgz', 'dir'], default=None,
                            help='Select output as a self-extracting binary (bin, default), or a regular directory '
                                 '(dir)')
    ouput_opts.add_argument('--output_scripttype', type=str, default=None,
                            help='Select the type of output scripts to be generated. Choices: bash (default, simple '
                                 'bash scripts), slurm, pbs (scripts to be submitted to a job manager), or any value '
                                 '(and this value will be used as the submit command, this can only be used '
                                 'in conjunction with output_template)')
    ouput_opts.add_argument('--output_submit_args', type=str, default=None,
                            help='Pass this extra arguments to the submission script')
    ouput_opts.add_argument('--output_resources', type=str, default=None,
                            help='Define custom resources to be requested to the scheduler (only applicable if '
                                 'output_template is not used, see documentation for details)')
    ouput_opts.add_argument('--output_template', type=str, default=None,
                            help='Use contents of this file as headers for submission or run. If this option is '
                                 'omitted, default templates will be used. This can be used in conjunction with '
                                 'output_scripttype to support other schedulers.')
    ouput_opts.add_argument('--output_collecttype', type=str, default=None,
                            help='Select type of script to be used to process results during the collect stage. '
                                 'See manual for details')
    ouput_opts.add_argument('--output_runbefore', type=str, default=None,
                            help='Add these commands (or read from this file, automatically detected) to be run before '
                                 'actual run in output script.')
    ouput_opts.add_argument('--output_runafter', type=str, default=None,
                            help='Add these commands (or read from this file, automatically detected) to be run after '
                                 'actual run in output script.')
    ouput_opts.add_argument('--output_njobs', type=int, default=None,
                            help='Use this many jobs during rerun, collect and analysis steps (Default: -1: guess)')
    ouput_opts.add_argument('--output_hidden_temp_dir', type=str, default=None,
                            help='Use a hidden temporary directory for file operations. If False, a dir in $PWD will '
                                 'be used (Default: True)')

    process_user_input.add_argparse_global_args(Parser)
    arguments = process_user_input.read_options(Parser, unpack_section='prepare_dual_topology')

    os_util.local_print('These are the input options: {}'.format(arguments),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)

    # Suppress lengthy openbabel warnings, unless user wants to
    if arguments.verbose <= 3:
        pybel.ob.obErrorLog.SetOutputLevel(pybel.ob.obError)

    if arguments.perturbations_dir and os.path.exists(arguments.perturbations_dir):
        if not arguments.no_checks:
            os_util.local_print('Output directory "{}" exists. Cannot continue. Remove/rename {} or use another '
                                'perturbation_dir. Alternatively, you can rerun with no_checks, so I will overwrite '
                                'the output.'
                                ''.format(arguments.perturbations_dir, arguments.perturbations_dir),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise FileExistsError('{} exists'.format(arguments.perturbations_dir))
        else:
            os_util.local_print('Output directory {} exists. Because you are running with no_checks, I will OVERWRITE '
                                '{}!'
                                ''.format(arguments.perturbations_dir, arguments.perturbations_dir),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)

    for each_arg in ['structure']:
        if arguments[each_arg] is None:
            os_util.local_print('Argument "--{}" or config file option "{}" is required. Please see the documentation'
                                ''.format(each_arg, each_arg), msg_verbosity=os_util.verbosity_level.error,
                                current_verbosity=arguments.verbose)
            raise SystemExit(1)

    arguments.mcs_custom_mcs = os_util.detect_type(arguments.mcs_custom_mcs, test_for_list=True)
    if arguments.mcs_custom_mcs is None:
        custom_mcs_data = None
    elif isinstance(arguments.mcs_custom_mcs, str):
        custom_mcs_data = arguments.mcs_custom_mcs
    elif isinstance(arguments.mcs_custom_mcs, list):
        n_elements = len(arguments.mcs_custom_mcs)
        if n_elements % 3 == 0:
            from numpy import arange

            custom_mcs_data = {}
            for (mol_name_1, mol_name_2, mcs_smarts) in [arange(3 * i, 3 * i + 3)
                                                         for i in range(int(n_elements / 3))]:
                key = frozenset([arguments.mcs_custom_mcs[mol_name_1], arguments.mcs_custom_mcs[mol_name_2]])
                custom_mcs_data[key] = arguments.mcs_custom_mcs[mcs_smarts]
        else:
            os_util.local_print('Error while processing mcs_custom_mcs argument or option. If a list in supplied to '
                                'mcs_custom_mcs, it must have 3n elements, formatted as "mol1_name, mol2_name, mcs_a, '
                                'mol3_name, mol4_name, mcs_b". Please, see the manual. Data read from mcs_custom_mcs '
                                'was {}'.format(arguments.mcs_custom_mcs),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(-1)
    else:
        os_util.local_print('Could not understand the mcs_custom_mcs argument or option. List or string expected, but '
                            'type {} found. Data read was {}. Please, see the manual.'
                            ''.format(type(arguments.mcs_custom_mcs), arguments.mcs_custom_mcs),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(-1)

    if arguments.output_packing not in ['bin', 'dir', 'tgz']:
        os_util.local_print('Packing {} not recognized, please, select between "bin", "tgz" or "dir"',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(-1)
    elif arguments.output_packing == 'dir':
        os_util.local_print('You are using raw directory output. This will generate uncompressed data, which maybe a '
                            'waste of resources. You may prefer "bin" or "tgz" packing',
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)

    for each_arg in ['extradirs', 'extrafiles', 'water_mdp', 'complex_mdp']:
        arguments[each_arg] = os_util.detect_type(arguments[each_arg], test_for_list=True, verbosity=arguments.verbose)
        arguments[each_arg] = [arguments[each_arg]] if isinstance(arguments[each_arg], str) else arguments[each_arg]

    if arguments.pre_solvated and not arguments.topology:
        if arguments.no_checks:
            os_util.local_print('You provided a pre-solvated structure, but did not provided a topology. Because '
                                'you are running with no_checks, I will go on. You will need to edit the topology '
                                'manually. Alternatively, use the topology input option',
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)
        else:
            os_util.local_print('You provided a pre-solvated structure, but did not provided a topology. Please, '
                                'use the topology input option. Alternatively, you can run with no_checks to '
                                'bypass this checking.',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)

    arguments.mdp_substitution = os_util.detect_type(arguments.mdp_substitution, test_for_dict=True)
    if arguments.mdp_substitution:
        if isinstance(arguments.mdp_substitution, str):
            mdp_data = os_util.read_file_to_buffer(arguments.mdp_substitution,
                                                   error_message='Failed to read mdp substitution from file.',
                                                   verbosity=arguments.verbose)
            arguments.mdp_substitution = os_util.detect_type(mdp_data, test_for_dict=True)
        if not isinstance(arguments.mdp_substitution, dict):
            os_util.local_print('Failed to understand mdp_substitution value "{}". Either a dict or a parsable file '
                                'are required.'.format(arguments.mdp_substitution),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise TypeError('Expect a str of dict, got {} instead'.format(type(arguments.mdp_substitution)))

    # TODO: alter this to support other systems
    mdp_data = {}
    if bool(arguments.complex_mdp) ^ bool(arguments.water_mdp):
        os_util.local_print('If you use complex_mdp or water_mdp, you have to supply both. ',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(1)
    elif arguments.complex_mdp and arguments.water_mdp:
        mdp_data['protein'] = arguments.complex_mdp
        mdp_data['water'] = arguments.water_mdp
    elif arguments.template_mdp:
        try:
            mdp_data = os_util.detect_type(arguments.internal.mdp[arguments.template_mdp], test_for_dict=True)
        except KeyError:
            os_util.local_print('Template mdp type "{}" not found.'.format(arguments.template_mdp),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)
        else:
            mdp_data = {key: [os.path.join(os.path.dirname(__file__), ele) for ele in val]
                        for key, val in mdp_data.items()}
            # TODO: remove this when supporting other systems
            mdp_data['protein'] = mdp_data.pop('complex')
    else:
        os_util.local_print('One of complex_mdp and water_mdp, or template_mdp is required. See documentation.',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(1)

    formatted_data = ['{:=^50}\n{:^25} {:^25}'.format(' mdp and run steps=', 'Complex', 'Water')]
    formatted_data.extend(['{:^25} {:^25}'.format(os.path.basename(a), os.path.basename(b)) for a, b in
                           itertools.zip_longest(mdp_data['protein'], mdp_data['water'], fillvalue='-')])
    formatted_data.append('=' * 50)
    os_util.local_print('\n'.join(formatted_data),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)

    # Prepare file to save data
    if arguments.progress_file != '':
        progress_data = savestate_util.SavableState(arguments.progress_file)
    else:
        progress_data = savestate_util.SavableState()

    # TODO: change this when allowing flexible systems
    progress_data['systems'] = ['protein', 'water']
    progress_data.save_data()

    if arguments.index_string is not None:
        test_data = os_util.detect_type(arguments.index_string, test_for_dict=True, verbosity=arguments.verbose)
        if isinstance(arguments.index_string, dict):
            new_index_groups_dict = test_data
        else:
            index_data = os_util.read_file_to_buffer(arguments.index_string, die_on_error=True,
                                                     error_message='Could not read index file.',
                                                     verbosity=arguments.verbose)
            index_data = os_util.detect_type(index_data, test_for_dict=True, verbosity=arguments.verbose)
            if isinstance(index_data, dict):
                new_index_groups_dict = index_data
            else:
                os_util.local_print('Could not read data in file {} as an index dictionary. Please, check file format',
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
                raise SystemExit(1)

    else:
        new_index_groups_dict = None

    if arguments.index:
        index_data = read_index_data(arguments.index, verbosity=arguments.verbose)
    else:
        index_data = False

    lambda_dict = process_lambdas_input(arguments.lambda_input, no_checks=arguments.no_checks,
                                        lambda_file=arguments.internal.default['lambdas'],
                                        verbosity=arguments.verbose)

    if not arguments.plumed_conf:
        plumed_conf = ''
    else:
        plumed_conf = os_util.read_file_to_buffer(arguments.plumed_conf, die_on_error=False, return_as_list=False)
        if plumed_conf is False:
            plumed_conf = os_util.detect_type(plumed_conf, test_for_list=True, verbosity=arguments.verbose)
            if isinstance(plumed_conf, list):
                plumed_conf = '\n'.join(plumed_conf)
            elif not isinstance(plumed_conf, str):
                os_util.local_print('Failed to parse plumed_conf argument. Value should be a filename, a list or a '
                                    'string. Read {} and it was parsed as {} ({})'
                                    ''.format(arguments.plumed_conf, plumed_conf, type(plumed_conf)))
                raise ValueError('list or str expected, {} found'.format(type(plumed_conf)))

    if arguments.solute_scaling != -1:
        # Solute scaling has been requested, prepare the scaling coefficients
        try:
            arguments.solute_scaling = float(arguments.solute_scaling)
        except ValueError:
            os_util.local_print('An invalid scaling of {} has been requested. Make sure solute_scaling e [0.0, 1.1]'
                                ''.format(arguments.solute_scaling),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)
        if not arguments.no_checks and (arguments.solute_scaling > 1.0 or arguments.solute_scaling < 0.0):
            os_util.local_print('An invalid scaling of {} has been requested. Make sure solute_scaling e [0.0, 1.1]'
                                ''.format(arguments.solute_scaling),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)
        elif not arguments.no_checks and arguments.solute_scaling_selection == '':
            os_util.local_print('If you selected solute scaling, you have to provide --solute_scaling_selection',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)
        else:
            solute_scaling_list = generate_scaling_vector(1.0, arguments.solute_scaling, len(lambda_dict['coulA']))
            solute_scaling_atoms_dict = process_scaling_input(arguments.solute_scaling_selection)
            if arguments.solute_scaling_bin == 'rest2':
                scaling_bin = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tools', 'rest2.sh')
            elif arguments.solute_scaling_bin == 'rest1':
                scaling_bin = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tools', 'rest1.sh')
            else:
                scaling_bin = arguments.solute_scaling_bin

            try:
                os.chmod(scaling_bin, os.stat(scaling_bin).st_mode | 0o111)
            except OSError:
                os_util.local_print('Fail to make {} an executable. Applying solute scaling may fail.'
                                    ''.format(scaling_bin),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)

            os_util.local_print('This is the solute scaling data:\n\tsolute_scaling_list={}\n\t'
                                'solute_scaling_atoms_dict={}\n\tscaling_bin={}'
                                ''.format(solute_scaling_list, solute_scaling_atoms_dict, scaling_bin),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)

    if arguments.pre_solvated:
        if index_data is False:
            os_util.local_print('Could not read index file {} and you supplied a pre-solvated system. I will generate '
                                'one automatically using gmx make_ndx. If the script fails, supplying a previously '
                                'generated index file may help. Note: a "Protein" group is expected in such file.'
                                ''.format(arguments.index),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)
            os_util.run_gmx(arguments.gmx_bin_local, ['make_ndx', '-f', arguments.structure, '-o', arguments.index],
                            'q\n')
            index_data = read_index_data(arguments.index, verbosity=arguments.verbose)

        receptor_structure_mol = read_md_protein(arguments.structure, last_protein_atom=index_data['Protein'][-1])
    else:
        receptor_structure_mol = pybel.readfile('pdb', arguments.structure).__next__()

    ligands_dict = parse_ligands_data(arguments.input_ligands, savestate_util=progress_data,
                                      no_checks=arguments.no_checks, verbosity=arguments.verbose)

    poses_input = parse_poses_data(parse_input_molecules(arguments.poses_input, verbosity=arguments.verbose),
                                   no_checks=arguments.no_checks, verbosity=arguments.verbose)

    if arguments.poses_advanced_options:
        arguments.poses_advanced_options = os_util.detect_type(arguments.poses_advanced_options, test_for_dict=True)
        if not isinstance(arguments.poses_advanced_options, dict):
            if not arguments.no_checks:
                os_util.local_print('Could not understand poses_advanced_options as a dictionary. Data read was: "{}" '
                                    'as a {} type'
                                    ''.format(arguments.poses_advanced_options, type(arguments.poses_advanced_options)),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
                raise SystemExit(1)
            else:
                os_util.local_print('Could not understand poses_advanced_options as a dictionary. Data read was: "{}" '
                                    'as a {} type. Because you are running with no_checks, I will ignore '
                                    'poses_advanced_options and move on.'
                                    ''.format(arguments.poses_advanced_options, type(arguments.poses_advanced_options)),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
                arguments.poses_advanced_options = {}
    else:
        arguments.poses_advanced_options = {}

    # Reads poses data. If a custom MCS was supplied as str (ie, all MCS should custom), use it during pose loading.
    # Otherwise, user wants only specific pairs (of ligands) to use a custom MCS, so use find_mcs during pose loading.
    poses_mcs = custom_mcs_data if isinstance(custom_mcs_data, str) else None
    poses_mol_data = align_ligands(receptor_structure_mol, poses_input,
                                   poses_reference_structure=arguments.poses_reference_structure,
                                   reference_pose_superimpose=arguments.poses_reference_pose_superimpose,
                                   superimpose_loader_ligands=ligands_dict,
                                   pose_loader=arguments.pose_loader, mcs=poses_mcs, save_state=progress_data,
                                   verbosity=arguments.verbose, **arguments.poses_advanced_options)

    if not poses_mol_data:
        os_util.local_print('Failed during the align ligands step. Cannot continue.',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)

        raise SystemExit(-1)

    # Reads a perturbation file or reads from savable state data
    if arguments.perturbation_map:

        perturbation_map = process_perturbation_map(arguments.perturbation_map, verbosity=arguments.verbose)

        if not perturbation_map:
            os_util.local_print('No perturbation could be read from {}. Please, check your input'
                                ''.format(arguments.perturbation_map),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(-1)

        # User supplied a perturbation map instead of using the saved map in progress file. Create a graph and save it
        # to progress file
        import networkx

        perturbation_graph = networkx.DiGraph()
        [perturbation_graph.add_edge(i, j, **data) for (i, j), data in perturbation_map.items()]
        progress_data['perturbation_map'] = perturbation_graph.copy()
        progress_data['perturbation_map_{}'.format(time.strftime('%H%M%S_%d%m%Y'))] = perturbation_graph.copy()

    else:
        perturbation_graph = progress_data.get('perturbation_map', None)
        if not perturbation_graph:
            try:
                perturbation_graph = progress_data['thermograph']['last_solution']
            except KeyError as error:
                if error.args[0] == 'thermograph':
                    os_util.local_print('Perturbation map data {} corrupt. Please, run generat_perturbation_map '
                                        'or use perturbation_map argument or input file option.'
                                        ''.format(progress_data.data_file), msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)
                    raise SystemExit(1)
                else:
                    os_util.local_print('Could not find a perturbation map in {}. Please, run generat_perturbation_map '
                                        'or use perturbation_map argument or input file option.'
                                        ''.format(progress_data.data_file), msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)
                    raise SystemExit(1)
            else:
                perturbation_graph = perturbation_graph['best_solution']
                perturbation_map = {(node_i, node_j): edge_data
                                    for node_i, node_j, edge_data in perturbation_graph.edges.data()}

                # TODO: this may be become an unsafe assumption at some point, but for now it is only used to save
                #       the center_molecule
                maptype = progress_data['thermograph'].get('runtype', 'optimal')
                if maptype in ['star', 'wheel']:
                    progress_data['center_molecule'] = progress_data['thermograph']['bias']
                progress_data['perturbation_map'] = perturbation_graph.copy()
                progress_data['perturbation_map_{}'.format(time.strftime('%H%M%S_%d%m%Y'))] = perturbation_graph.copy()

                os_util.local_print('Pertubation map read from thermograph.best_solution in {}. Graph:\n\t{}'
                                    ''.format(progress_data.data_file, perturbation_graph.edges),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)

        else:
            perturbation_map = {(node_i, node_j): edge_data
                                for node_i, node_j, edge_data in perturbation_graph.edges.data()}
            os_util.local_print('Pertubation map read from perturbation_map in {}. Graph:\n\t{}'
                                ''.format(progress_data.data_file, perturbation_graph.edges),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)
            progress_data.save_data()

    os_util.local_print('Pertubation map read from {}. Info:\n\tNumber of nodes (molecules): {}'
                        '\n\tNumber of edges (perturbations): {}'
                        ''.format(arguments.perturbation_map,
                                  len(set(map(lambda x: x[0], perturbation_map))
                                      .union(map(lambda x: x[1], perturbation_map))), len(perturbation_map)),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)

    os_util.local_print('{:=^50}\n{:^25}{:^25}'.format(' Perturbations ', 'State A', 'State B'),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)
    [os_util.local_print('{:^25}{:^25}'.format(mol_a, mol_b), msg_verbosity=os_util.verbosity_level.default,
                         current_verbosity=arguments.verbose)
     for mol_a, mol_b in perturbation_map]
    os_util.local_print('=' * 50, msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)

    original_base_pert_dir = arguments.perturbations_dir if arguments.perturbations_dir \
        else 'perturbations_{}'.format(time.strftime('%H%M%S_%d%m%Y'))

    if not arguments.output_hidden_temp_dir:
        tmpdir = all_classes.Namespace({'name': os.getcwd(), 'cleanup': lambda: None})
        os_util.local_print('Preparing perturbations in {}. This directory will not be removed upon completion.'
                            ''.format(os.path.join(tmpdir.name, original_base_pert_dir)),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)
        os_util.makedir(tmpdir.name, verbosity=arguments.verbose)
    else:
        tmpdir = tempfile.TemporaryDirectory()

    base_pert_dir = os.path.join(tmpdir.name, original_base_pert_dir)
    os_util.makedir(base_pert_dir, verbosity=arguments.verbose)

    new_pairs_list = []
    existing_dir_list = []
    for morph_pair in perturbation_map:
        morph_dir = '{}-{}'.format(*morph_pair)
        if os.path.isdir(os.path.join(base_pert_dir, morph_dir)):
            existing_dir_list.append(morph_dir)
        else:
            new_pairs_list.append(morph_pair)

    if existing_dir_list:
        os_util.local_print('Existing directories (will not be overwritten): {}'.format(', '.join(existing_dir_list)),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)
    else:
        os_util.local_print('I found no existing dir to be kept.',
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)

    os_util.local_print('I will work on these perturbations: {}'
                        ''.format(', '.join(['{}\u2192{}'.format(*i) for i in new_pairs_list])),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)
    os_util.local_print('{:=^50}\n{} {} {}'.format(' Working on pairs ', 'Perturbation', 'Pose', 'Coordinates'),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)

    output_data = prepare_output_scripts_data(header_template=arguments.output_template,
                                              script_type=arguments.output_scripttype,
                                              submission_args=arguments.output_submit_args,
                                              custom_scheduler_resources=arguments.output_resources,
                                              hrex_frequency=arguments.hrex,
                                              collect_type=os_util.detect_type(arguments.output_collecttype,
                                                                               test_for_list=True),
                                              temperature=arguments.md_temperature,
                                              gmx_bin=arguments.gmx_bin_run,
                                              index_file=arguments.index,
                                              n_jobs=arguments.output_njobs,
                                              run_before=arguments.output_runbefore,
                                              run_after=arguments.output_runafter,
                                              scripts_templates=arguments.internal.default['output_files_data'],
                                              verbosity=arguments.verbose)

    # This will be the generated script data
    output_script_list = [output_data['shebang'], '', 'lastjid=()']

    # Iterate over perturbations pairs
    for morph_pair in new_pairs_list:

        morph_dir = '{}-{}'.format(*morph_pair)

        os_util.local_print('{:=^50}'.format(' Working on {} '.format(morph_dir)),
                            msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)

        # Get the names of the ligands and prepare the accordingly files
        state_a_name, state_b_name = morph_pair
        try:
            mol2file_a = ligands_dict[state_a_name]['molecule']
            topfile_a = ligands_dict[state_a_name]['topology']
        except KeyError:
            os_util.local_print('Ligand {} not found in the ligand data read. Cannot continue. These are the ligands '
                                'read: {}'.format(state_a_name, ' '.join([n for n in ligands_dict.keys()])),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise KeyError('Ligand {} not found'.format(state_a_name))

        mol2file_b = ligands_dict[state_b_name]['molecule']
        topfile_b = ligands_dict[state_b_name]['topology']

        if not custom_mcs_data:
            this_custom_mcs = None
        elif isinstance(custom_mcs_data, str):
            this_custom_mcs = custom_mcs_data
        elif isinstance(custom_mcs_data, dict):
            if frozenset([state_a_name, state_b_name]) in custom_mcs_data:
                this_custom_mcs = custom_mcs_data[frozenset([state_a_name, state_b_name])]
            else:
                this_custom_mcs = None
        else:
            os_util.local_print("Could not understand custom_mcs_data, read from mcs_custom_mcs option. Please, check "
                                'your input. mcs_custom_mcs is "{}"'.format(arguments.mcs_custom_mcs),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(-1)
        # From individual files, prepare a dual topology molecule and topology
        merged_data = merge_topologies.merge_topologies(ligands_dict[state_a_name]['molecule'],
                                                        ligands_dict[state_b_name]['molecule'],
                                                        ligands_dict[state_a_name]['topology'],
                                                        ligands_dict[state_b_name]['topology'],
                                                        savestate=progress_data, no_checks=arguments.no_checks,
                                                        mcs=this_custom_mcs, verbosity=arguments.verbose)

        if 'lambda' in perturbation_map[tuple(morph_pair)]:
            this_lambda_table = process_lambdas_input(perturbation_map[tuple(morph_pair)]['lambda'],
                                                      no_checks=arguments.no_checks,
                                                      lambda_file=arguments.internal.default['lambdas'],
                                                      verbosity=arguments.verbose)
            merged_data.dual_topology.lambda_table = this_lambda_table
            if arguments.solute_scaling != -1:
                this_solute_scaling_list = generate_scaling_vector(1.0, arguments.solute_scaling,
                                                                   len(this_lambda_table['coulA']))
        else:
            merged_data.dual_topology.lambda_table = lambda_dict
            this_lambda_table = lambda_dict
            if arguments.solute_scaling != -1:
                this_solute_scaling_list = solute_scaling_list

        # Then embed it to the reference pose
        merged_data = merge_topologies.constrained_embed_dualmol(merged_data,
                                                                 rdkit.Chem.RemoveHs(poses_mol_data[state_a_name]),
                                                                 mcs=this_custom_mcs, verbosity=arguments.verbose,
                                                                 savestate=progress_data)

        dual_topology_data = merged_data.dual_topology

        # TODO: allow arbitrary molecule names
        dual_topology_data.molecules[0].name = 'LIG'

        os_util.local_print('{} -> {}'.format(state_a_name, state_b_name), current_verbosity=arguments.verbose,
                            msg_verbosity=os_util.verbosity_level.default)
        dual_topology_data.set_lambda_state(0)

        extra_new_files = {'ligand.atp': dual_topology_data.__str__('atomtypes'),
                           'ligand.itp': dual_topology_data.__str__('itp')}
        create_fep_dirs(morph_dir, base_pert_dir, new_files=extra_new_files, extra_dir=arguments.extradirs,
                        extra_files=arguments.extrafiles, verbosity=arguments.verbose)

        # These files will not be copied to water building dir
        unwanted_files = [arguments.topology, 'FullSystem.pdb', 'md']
        unwanted_files += ['lambda{}'.format(i) for i in range(len(this_lambda_table['coulA']))]

        if arguments.pre_solvated:
            # User provided a pre-solvated strucuture
            output_structure_file = os.path.join(base_pert_dir, morph_dir, 'protein', 'FullSystem.pdb')
            output_topology_file = os.path.join(base_pert_dir, morph_dir, 'protein', 'FullSystem.top')

            # Align ligand
            add_ligand_to_solvated_receptor(merged_data, arguments.structure, output_structure_file,
                                            index_data, input_topology=arguments.topology,
                                            output_topology_file=output_topology_file,
                                            num_protein_chains=arguments.presolvated_protein_chains,
                                            radius=arguments.presolvated_radius,
                                            selection_method=arguments.selection_method,
                                            verbosity=arguments.verbose)

            # FIXME: run genion here to neutralize the system in a case a user supply a system with a different charge
            #  than the sun system (this should be uncommon)
            make_index(os.path.join(base_pert_dir, morph_dir, 'protein', 'index.ndx'), output_structure_file,
                       new_index_groups_dict, method=arguments.selection_method, gmx_bin=arguments.gmx_bin_local,
                       verbosity=arguments.verbose)

        else:
            # User wants automatic complex generation
            os_util.local_print('You are using the automated system building. Check the final system and, if '
                                'needed, the intermediate files. The system building routines are very simple, '
                                'convenience functions and will likely fail for complex or exotic systems. In these '
                                'cases, please, supply a pre-solvated system.',
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)

            os_util.local_print('{:=^50}'.format(' Building system '),
                                msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)

            solvate_data = {'water_model': arguments.buildsys_water, 'water_shell': arguments.buildsys_watershell,
                            'ion_concentration': arguments.buildsys_ionconcentration,
                            'pname': arguments.buildsys_pname, 'nname': arguments.buildsys_nname,
                            'box_type': arguments.buildsys_boxtype}

            build_data = prepare_complex_system(structure_file=arguments.structure,
                                                base_dir=os.path.join(base_pert_dir, morph_dir),
                                                ligand_dualmol=merged_data,
                                                topology=arguments.topology, index_file=arguments.index,
                                                forcefield=arguments.buildsys_forcefield,
                                                index_groups=new_index_groups_dict,
                                                selection_method=arguments.selection_method,
                                                gmx_bin=arguments.gmx_bin_local, extradirs=arguments.extradirs,
                                                gmx_maxwarn=arguments.gmx_maxwarn, extrafiles=arguments.extrafiles,
                                                solvate_data=solvate_data, verbosity=arguments.verbose)

            unwanted_files += [os.path.basename(build_data.build_dir)]
            output_structure_file = build_data.structure

        protein_topology_files = [os.path.join(base_pert_dir, morph_dir, 'protein', each_file)
                                  for each_file in os.listdir(os.path.join(base_pert_dir, morph_dir, 'protein'))
                                  if each_file not in unwanted_files]

        solvate_data = {'water_model': arguments.buildsys_water, 'water_shell': arguments.buildsys_watershell,
                        'ion_concentration': 0, 'pname': arguments.buildsys_pname, 'nname': arguments.buildsys_nname,
                        'box_type': arguments.buildsys_boxtype}

        # Prepare water perturbations (this is irrespective of whether user used a pre-solvated system or not)
        water_data = prepare_water_system(dual_molecule_ligand=merged_data,
                                          water_dir=os.path.join(base_pert_dir, morph_dir, 'water'),
                                          topology_file=arguments.topology,
                                          solvate_data=solvate_data,
                                          protein_topology_files=protein_topology_files,
                                          gmx_maxwarn=arguments.gmx_maxwarn, gmx_bin=arguments.gmx_bin_local,
                                          verbosity=arguments.verbose)

        # Create lambda dirs
        for each_system, each_mdplist in mdp_data.items():

            equilibration_output = []

            for each_value in range(len(this_lambda_table['coulA'])):
                this_basedir = os.path.join(base_pert_dir, morph_dir, each_system, 'lambda{}'.format(each_value))
                # Create a lambdaX dir
                os_util.makedir(this_basedir, verbosity=arguments.verbose)

                for each_file in each_mdplist:
                    stepname = os.path.basename(each_file).replace(".mdp", "")
                    # Create a directory for each minimization and equilibration step inside the lambdaX dir
                    this_each_dir = os.path.join(this_basedir, stepname)
                    os_util.makedir(this_each_dir, verbosity=arguments.verbose)
                    try:
                        shutil.copy2(each_file, this_each_dir)
                    except FileNotFoundError:
                        os_util.local_print('Could not find the mdp file {}. Cannot continue.'
                                            ''.format(each_file), msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=arguments.verbose)
                        raise SystemExit(1)

                    substitutions = {}
                    if arguments.mdp_substitution:
                        if stepname in arguments.mdp_substitution and \
                                isinstance(arguments.mdp_substitution[stepname], dict):
                            substitutions.update(arguments.mdp_substitution[stepname])
                        else:
                            substitutions.update(arguments.mdp_substitution)

                    # if arguments.perturbations_softcore:
                    #     fn0 = {'minimum': min,
                    #            'maximum': max,
                    #            'average': lambda x: sum(x)/len(x)}
                    #
                    #     vdw_values = [this_lambda_table['vdwA'][each_value], this_lambda_table['vdwB'][each_value]]
                    #
                    #     try:
                    #         sc_line = '{}       1.0'.format(fn0[arguments.perturbations_softcore](*vdw_values))
                    #         substitutions.update({'fep-lambdas': sc_line})
                    #     except KeyError:
                    #         os_util.local_print('Could not understand function to obtain soft-core lambda "{}". Please '
                    #                             'select between {}.'
                    #                             ''.format(arguments.perturbations_softcore, list(fn0.keys())),
                    #                             msg_verbosity=os_util.verbosity_level.error,
                    #                             current_verbosity=arguments.verbose)
                    #         raise SystemExit(1)

                    substitutions.update(_TEMPERATURE=arguments.md_temperature)

                    if stepname == 'md':
                        substitutions.update(_LENGTH=arguments.md_length)

                    if substitutions:
                        edit_mdp_file(os.path.join(this_each_dir, os.path.basename(each_file)),
                                      substitutions=substitutions, no_checks=arguments.no_checks,
                                      verbosity=arguments.verbose)

                # Prepare the lambda state for saving dual_topology
                dual_topology_data.set_lambda_state(each_value)
                extra_new_files = {'ligand.atp': dual_topology_data.__str__('atomtypes'),
                                   'ligand.itp': dual_topology_data.__str__('itp')}

                # Symlink each file in pert_dir to inside each lambdaX dir
                for each_file in os.listdir(os.path.join(base_pert_dir, morph_dir, each_system)):
                    if each_file not in \
                            ['lambda{}'.format(i) for i in range(len(this_lambda_table['coulA']))] \
                            + ['md', 'ligand.atp', 'ligand.itp']:
                        os.symlink(os.path.join('..', each_file), os.path.join(this_basedir,
                                                                               os.path.basename(each_file)))
                        os_util.local_print('Symlinking {} -> {}'
                                            ''.format(os.path.join('..', each_file),
                                                      os.path.join(this_basedir, os.path.basename(each_file))),
                                            msg_verbosity=os_util.verbosity_level.debug,
                                            current_verbosity=arguments.verbose)

                # Create ligand topology files in each lambda dir
                for each_file, contents in extra_new_files.items():
                    with open(os.path.join(this_basedir, each_file), 'w') as fh:
                        fh.write(contents)
                    os_util.local_print('Creating file {} in {}'.format(each_file, this_basedir),
                                        msg_verbosity=os_util.verbosity_level.debug,
                                        current_verbosity=arguments.verbose)

                dir_mdp_list = [os.path.join(this_basedir, os.path.basename(each_file).replace(".mdp", ""))
                                for each_file in each_mdplist]

                topology_file = arguments.topology if each_system == 'protein' \
                    else os.path.basename(water_data['topology'])
                structure_file = output_structure_file if each_system == 'protein' \
                    else os.path.basename(water_data['structure'])

                if arguments.solute_scaling == -1:
                    # No solute scaling
                    equilibration_output.append(prepare_steps(each_value, this_basedir, dir_mdp_list,
                                                              topology_file=topology_file,
                                                              structure_file=structure_file,
                                                              index_file=arguments.index,
                                                              plumed_conf=plumed_conf,
                                                              gmx_path=arguments.gmx_bin_run,
                                                              gmx_maxwarn=arguments.gmx_maxwarn,
                                                              verbosity=arguments.verbose))
                else:
                    # User wants solute scaling
                    equilibration_output.append(prepare_steps(each_value, this_basedir, dir_mdp_list,
                                                              topology_file=topology_file,
                                                              structure_file=structure_file,
                                                              index_file=arguments.index, scaling_bin=scaling_bin,
                                                              solute_scaling_list=this_solute_scaling_list,
                                                              solute_scaling_atoms_dict=solute_scaling_atoms_dict,
                                                              plumed_conf=plumed_conf,
                                                              local_gmx_path=arguments.gmx_bin_local,
                                                              gmx_path=arguments.gmx_bin_run,
                                                              gmx_maxwarn=arguments.gmx_maxwarn,
                                                              verbosity=arguments.verbose))

            this_basedir = os.path.join(base_pert_dir, morph_dir, each_system)
            os_util.makedir(os.path.join(this_basedir, 'md'), verbosity=arguments.verbose)
            os_util.makedir(os.path.join(this_basedir, 'md', 'rerun'), verbosity=arguments.verbose)
            os_util.makedir(os.path.join(this_basedir, 'md', 'analysis'), verbosity=arguments.verbose)

            # This will be the output script
            this_runall_script = os.path.join(this_basedir, '..', 'runall_{}_{}.sh'.format(morph_dir, each_system))

            run_command = output_data['constantpart']['run']
            substitution = {'__MDDIR__': os.path.join(morph_dir, each_system, 'md')}
            for each_holder, each_data in substitution.items():
                run_command = run_command.replace(each_holder, str(each_data))

            header_command = output_data['header']
            substitution = {'__JOBNAME__': '{}_{}'.format(morph_dir, each_system)}
            for each_holder, each_data in substitution.items():
                header_command = header_command.replace(each_holder, str(each_data))

            # Write to it
            with open(this_runall_script, 'w') as fh:
                fh.write(header_command + '\n\n')
                if output_data['run_before']:
                    fh.write(output_data['run_before'] + '\n')
                fh.write('\n\necho "$(date): Starting equilibration in {} {} at $(hostname)"\n\n'
                         ''.format(morph_dir, each_system))
                fh.write('\n'.join(equilibration_output))
                fh.write('\n\necho "$(date): Starting run in {} {} at $(hostname)"\n\n'
                         ''.format(morph_dir, each_system))
                fh.write(run_command + '\n')
                fh.write('\n\necho "$(date): Starting rerun in {} {} at $(hostname)"\n\n'
                         ''.format(morph_dir, each_system))
                fh.write(output_data['constantpart']['rerun'] + '\n')
                fh.write('\n\necho "$(date): Starting collect in {} {} at $(hostname)"\n\n'
                         ''.format(morph_dir, each_system))
                fh.write(output_data['constantpart']['collect'] + '\n')
                if each_system not in ['water', 'vacuum', 'solvent']:
                    # Analysis script is for complex only
                    fh.write('\n\necho "$(date): Starting analysis in {} {} at $(hostname)"\n\n'
                             ''.format(morph_dir, each_system))
                    fh.write(output_data['constantpart']['analysis'] + '\n')
                if output_data['run_after']:
                    fh.write(output_data['run_after'] + '\n')

            output_script_list.append('lastjid+=( $({} {} {}) )'
                                      ''.format(output_data['submit_command'], output_data['submission_args'],
                                                os.path.join(os.path.join(morph_dir),
                                                             'runall_{}_{}.sh'.format(morph_dir, each_system))))

    substitution = {'__PACKED_FILE__': '{}.tgz'.format(original_base_pert_dir),
                    '__JOBNAME__': 'pack_{}'.format(os.path.split(base_pert_dir)[-1])}
    pack_data = output_data['pack_script']
    for each_holder, each_data in substitution.items():
        pack_data = pack_data.replace(each_holder, str(each_data))

    # This script packs results and analysis into a tgz file
    with open(os.path.join(base_pert_dir, 'pack.sh'), 'w') as fh:
        fh.write(pack_data)

    output_script_list.append('{} {} {} pack.sh'
                              ''.format(output_data['submit_command'], output_data['submission_args'],
                                        output_data.depend_string.format('$(IFS=,; echo "${lastjid[*]}")')))

    with open(os.path.join(base_pert_dir, 'runall.sh'), 'w+') as fh:
        fh.write('\n'.join(output_script_list))

    if output_data['collect_executables']:
        for each_file in output_data['collect_executables']:
            try:
                shutil.copy2(each_file, base_pert_dir)
            except FileNotFoundError:
                os_util.local_print('Could not find collect executable {}. If you supplied a custom executable (eg: '
                                    'using output_collecttype), check that file exists.'
                                    ''.format(each_file),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbosity)

    progress_data.save_data()
    shutil.copy2(progress_data.data_file, os.path.join(base_pert_dir, os.path.basename(progress_data.data_file)))

    if arguments.output_packing == 'bin':
        with tempfile.SpooledTemporaryFile(mode='wb') as fh:
            with tarfile.open(fileobj=fh, mode='w:gz') as tarfh:
                tarfh.add(base_pert_dir, original_base_pert_dir)
            fh.seek(0)
            gziped_data = fh.read()
        tmpdir.cleanup()

        with open('{}.bin'.format(original_base_pert_dir), 'wb') as fh:
            fh.write(output_data['selfextracting_script'].encode())
            fh.write(b'\n')
            fh.write(gziped_data)

        os_util.local_print('Input data written to {}'.format('{}.bin'.format(original_base_pert_dir)))
    elif arguments.output_packing == 'tgz':
        with tarfile.open('{}.tgz'.format(original_base_pert_dir), mode='w:gz') as tarfh:
            tarfh.add(base_pert_dir, original_base_pert_dir)
        tmpdir.cleanup()
        os_util.local_print('Input data written to {}'.format('{}.tgz'.format(original_base_pert_dir)))
    elif arguments.output_packing == 'dir':
        try:
            shutil.copytree(base_pert_dir, original_base_pert_dir)
        except FileExistsError:
            if arguments.output_hidden_temp_dir is not False:
                if arguments.no_checks:
                    os_util.local_print('You are running with no_checks, so I am OVERWRITING {}.'
                                        ''.format(original_base_pert_dir), msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)
                    shutil.rmtree(original_base_pert_dir, ignore_errors=True)
                    shutil.copytree(base_pert_dir, original_base_pert_dir)
                else:
                    os_util.local_print('Directory {} exists. I cannot write output to an exiting directory.'
                                        ''.format(original_base_pert_dir), msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)
                    raise FileExistsError('{} exists'.format(original_base_pert_dir))
        else:
            os_util.local_print('Input data written to {}'.format(original_base_pert_dir),
                                msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)
    else:
        os_util.local_print('Packing {} not recognized, please, select between "bin", "tgz" and "dir" using '
                            'output_packing',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(-1)
