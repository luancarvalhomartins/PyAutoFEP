#! /usr/bin/env python3
#
#  generate_perturbation_map.py
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

from copy import deepcopy
import networkx
import argparse
import rdkit.Chem
import rdkit.Chem.PropertyMol
import time
import itertools
from merge_topologies import find_mcs
from collections import OrderedDict
import all_classes
import multiprocessing
import savestate_util
from os.path import splitext
from statistics import median
from math import exp
import mol_util
import os_util
import process_user_input


def fill_thermograph(thermograph, molecules, pairlist=None, use_hs=False, threads=1, custom_mcs=None, savestate=None,
                     verbosity=0):
    """

    :param networkx.Graph thermograph: map to be edited
    :param dict molecules: molecules will be read from this dict, format {'molname': rdkit.Chem.Mol}
    :param list pairlist: create edges for these pairs (default: create edges for all possible pairs in molecules)
    :param bool use_hs: consider Hs in the perturbation costs (default: False)
    :param int threads: run this many threads (default = 1)
    :param dict custom_mcs: custom mcs and atom maps to be used
    :param savestate_util.SavableState savestate: saved state data
    :param int verbosity: set verbosity level
    """

    # Perturbations will connect larger molecules to smaller ones, by default.
    if not pairlist:
        pairlist = [(mol_i, mol_j) if molecules[mol_i].GetNumHeavyAtoms() >= molecules[mol_j].GetNumHeavyAtoms()
                    else (mol_j, mol_i) for mol_i, mol_j in itertools.combinations(molecules, 2)]

    if custom_mcs is None:
        custom_mcs = {}

    if not savestate:
        todo_pairs = pairlist
    else:
        todo_pairs = [pair for pair in pairlist if frozenset([rdkit.Chem.MolToSmiles(molecules[pair[0]]),
                                                              rdkit.Chem.MolToSmiles(molecules[pair[1]])])
                      not in savestate.setdefault('mcs_dict', {})]

    for pair in todo_pairs[:]:
        if frozenset(pair) in custom_mcs or '*' in custom_mcs:
            del todo_pairs[pair]

    if len(todo_pairs) > 0:
        if threads == -1:
            wrapper_fn_tmp = lambda args, kwargs: os_util.wrapper_fn(find_mcs, args, kwargs)
            mcs_data = map(wrapper_fn_tmp, [[[molecules[mol_i], molecules[mol_j]], None, verbosity]
                                            for (mol_i, mol_j) in todo_pairs],
                           itertools.repeat({'completeRingsOnly': True, 'matchValences': True,
                                             'ringMatchesRingOnly': True}))
        else:
            with multiprocessing.Pool(threads) as thread_pool:
                mcs_data = os_util.starmap_unpack(find_mcs, thread_pool,
                                                  [[[molecules[mol_i], molecules[mol_j]], None, verbosity]
                                                   for (mol_i, mol_j) in todo_pairs],
                                                  itertools.repeat({'completeRingsOnly': True, 'matchValences': True,
                                                                    'ringMatchesRingOnly': True}))
    else:
        mcs_data = []

    if savestate:
        for each_result, (mol_i, mol_j) in zip(mcs_data, todo_pairs):
            savestate['mcs_dict'][frozenset([rdkit.Chem.MolToSmiles(molecules[mol_i]),
                                             rdkit.Chem.MolToSmiles(molecules[mol_j])])] = each_result
        savestate.save_data()
        search_dict = savestate['mcs_dict']

        for each_pair in pairlist:
            if frozenset(each_pair) in custom_mcs:
                search_dict[frozenset(each_pair)] = custom_mcs[frozenset(each_pair)]
            elif '*' in custom_mcs:
                search_dict[frozenset(each_pair)] = custom_mcs['*']

    else:
        search_dict = {frozenset([rdkit.Chem.MolToSmiles(molecules[mol_i]),
                                  rdkit.Chem.MolToSmiles(molecules[mol_j])]): each_result
                       for each_result, (mol_i, mol_j) in zip(mcs_data, todo_pairs)}
        for each_pair in pairlist:
            if frozenset(each_pair) in custom_mcs:
                search_dict[frozenset(each_pair)] = custom_mcs[frozenset(each_pair)]
            elif '*' in custom_mcs:
                search_dict[frozenset(each_pair)] = custom_mcs['*']

    for each_mol_i, each_mol_j in pairlist:
        this_molkey = frozenset([rdkit.Chem.MolToSmiles(molecules[each_mol_i]),
                                 rdkit.Chem.MolToSmiles(molecules[each_mol_j])])
        if use_hs:
            num_core_atoms = rdkit.Chem.MolFromSmarts(search_dict[this_molkey].smartsString).GetNumAtoms()
            atoms_i = molecules[each_mol_i].GetNumAtoms()
            atoms_j = molecules[each_mol_j].GetNumAtoms()
        else:
            num_core_atoms = rdkit.Chem.MolFromSmarts(search_dict[this_molkey].smartsString).GetNumHeavyAtoms()
            atoms_i = molecules[each_mol_i].GetNumHeavyAtoms()
            atoms_j = molecules[each_mol_j].GetNumHeavyAtoms()

        # The edge cost is the number of perturbed atoms in a hypothetical transformation between the pair.
        perturbed_atoms = (atoms_i - num_core_atoms) + (atoms_j - num_core_atoms)
        if perturbed_atoms == 0:
            os_util.local_print('The perturbation between {} and {} would change no heavy atoms. Currently, this is '
                                'not supported. Should you need to simulate this perturbation, pass perturbation_map '
                                'directly to prepare_dual_topology.py'
                                ''.format(molecules[each_mol_i].GetProp('_Name'),
                                          molecules[each_mol_j].GetProp('_Name')))
            raise SystemExit(1)
        thermograph.add_edge(each_mol_i, each_mol_j, perturbed_atoms=perturbed_atoms, desirability=1.0)

    all_pert_atoms = [i for _, _, i in thermograph.edges(data='perturbed_atoms')]
    # Scale the number of perturbed atoms according to ln(0.2) * median(all_pert_atoms), so that the values are rescaled
    # to be [0, 1] and the median value will be 0.2
    # TODO: configurable beta expression
    beta = -1.6094379 / median(all_pert_atoms)
    for (edge_i, edge_j) in thermograph.edges:
        thermograph[edge_i][edge_j]['cost'] = 1 - exp(beta * thermograph[edge_i][edge_j]['perturbed_atoms'])


def test_center_molecule(map_bias, all_molecules, verbosity=0):
    """ Test center molecule to prepare star or wheel maps

    :param [list, str] map_bias: test this bias string or list
    :param list all_molecules: all molecules read com input
    :param int verbosity: sets the verbosity level
    :rtype: str
    """

    map_bias = os_util.detect_type(map_bias, test_for_list=True)
    if not map_bias:
        os_util.local_print('A star map requires one, and only one, center molecule. You supplied none.',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    if isinstance(map_bias, list) and len(map_bias) > 1:
        os_util.local_print('A star map requires one, and only one, center molecule. You supplied {} ({})'
                            ''.format(len(map_bias), map_bias),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    if isinstance(map_bias, list):
        map_bias = map_bias[0]

    if map_bias not in all_molecules:
        os_util.local_print('The center molecule you supplied ({}) not found in {}.'
                            ''.format(map_bias, ', '.join(all_molecules)),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise ValueError('Molecule not found')

    return map_bias


def run_workers(ant_colony, n_runs=-1, n_threads=1, elitism=-1, comm_freq=20, verbosity=0):
    """ Run optimization using multiprocessing

    :param all_classes.AntSolver ant_colony: optimizing object
    :param int n_runs: number of optimization ants. Default: -1 = automatically determine
    :param int n_threads: number of threads
    :param int elitism: use this many best ants for each parallel run to update pheromone matrix (default: -1: use all)
    :param int comm_freq: communicate between threads this often
    :param int verbosity: sets verbosity level
    """

    if n_runs == -1:
        # Automatically setting n_runs
        if n_threads == -1:
            n_runs = comm_freq * 20
        else:
            n_runs = n_threads * comm_freq * 20

    if 0 < elitism < 1:
        # Elitism was supplied as ratio, convert to int
        elitism = int(n_runs / (comm_freq * n_threads) * elitism)

    if n_threads == -1:
        os_util.local_print('You are using non-threaded code (ie: threads = -1). The implementation of the ACO '
                            'algorithm is slightly different when using the non-threaded code. This should only be '
                            'used for developing purposes.',
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

        for run_n in range(int(n_runs / comm_freq)):
            os_util.local_print('Optimization round {} out of {}'
                                ''.format(run_n + 1, int(n_runs / comm_freq)),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

            # Run a hive
            results_list = [each_group for each_group in
                            map(ant_colony.run_multi_ants, itertools.repeat(comm_freq, times=n_threads))]

            if run_n > 0:
                ant_colony.evaporate_pheromone()

            # Aggregate results, deposit pheromone
            ant_colony.solutions.extend([each_result for each_group in results_list for each_result in each_group])
            [ant_colony.deposit_pheromone(each_result.pheromone_multiplier, each_result.graph)
             for each_group in results_list
             for n, each_result in enumerate(sorted(each_group, key=lambda x: x.cost))
             if n < elitism or elitism == -1]

    else:
        with multiprocessing.Pool(n_threads) as thread_pool:
            for run_n in range(int(n_runs / (comm_freq * n_threads))):
                os_util.local_print('Optimization round {} out of {}'
                                    ''.format(run_n + 1, int(n_runs / (comm_freq * n_threads))),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

                # Run n_threads parallel hives
                results_list = [each_group for each_group in
                                thread_pool.map(ant_colony.run_multi_ants, itertools.repeat(comm_freq, times=n_threads))]

                if run_n > 0:
                    ant_colony.evaporate_pheromone()

                # Aggregate results, deposit pheromone
                ant_colony.solutions.extend([each_result for each_group in results_list for each_result in each_group])
                [ant_colony.deposit_pheromone(each_result.pheromone_multiplier, each_result.graph)
                 for each_group in results_list
                 for n, each_result in enumerate(sorted(each_group, key=lambda x: x.cost))
                 if n < elitism or elitism == -1]

        # Finish running workers (in case n_runs is not a multiple of comm_freq * n_threads)
        if len(ant_colony.solutions) < n_runs:
            results_list = ant_colony.run_multi_ants(n_runs - len(ant_colony.solutions))
            ant_colony.solutions.extend(results_list)
            [ant_colony.deposit_pheromone(each_result.pheromone_multiplier, each_result.graph)
             for each_result in results_list]


def process_custom_mcs(custom_mcs, savestate=None, verbosity=0):
    """ Parses user supplied custom MCS data

    :param [str, dict] custom_mcs: mcs data to be parsed 
    :param savestate_util.SavableState savestate: saved state data 
    :param int  verbosity: controls verbosity level 
    :rtype: dict
    """

    custom_mcs_result = {}
    if custom_mcs:
        custom_mcs = os_util.detect_type(custom_mcs, test_for_dict=True)
        if isinstance(custom_mcs, str):
            if rdkit.Chem.MolFromSmarts(custom_mcs) is not None:
                os_util.local_print('Using user-supplied MCS {} for all molecules.'.format(custom_mcs),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
                custom_mcs_result = {'*': custom_mcs}
            else:
                os_util.local_print('Could not parse you custom MCS "{}".'.format(custom_mcs),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
        elif isinstance(custom_mcs, dict):
            if all([(isinstance(key, frozenset) and len(key) == 2) for key in custom_mcs]):
                custom_mcs_result = custom_mcs
            elif all([(isinstance(key, str) and key.count('-') == 1) for key in custom_mcs]):
                custom_mcs_result = {frozenset(key.split('-')): value
                                    for key, value in custom_mcs.items()}
            else:
                os_util.local_print('Could not parse you custom MCS "{}". If providing a dict, make sure to follow '
                                    'the required format (see documentation).'.format(custom_mcs),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
        else:
            os_util.local_print('Could not parse you custom MCS. A string or dict is required, but your data "{}" '
                                'was parsed as a {} (see documentation for formatting options).'
                                ''.format(custom_mcs, type(custom_mcs)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        if savestate is not None:
            savestate['custom_mcs'] = custom_mcs_result
            savestate.setdefault('mcs_dict', {}).update(custom_mcs_result)
            savestate.save_data()

    return custom_mcs_result


if __name__ == '__main__':
    Parser = argparse.ArgumentParser(description='Generate a perturbation map using a heuristic algorithm')
    Parser.add_argument('-i', '--input', type=str, nargs='+', help='Input molecules')
    Parser.add_argument('--use_hs', default=None, help='Use hydrogens to score perturbations (Default: off)')
    Parser.add_argument('--custom_mcs', type=str, default=None,
                        help='Use this/these custom MCS between pairs. Can be either a string (so the same MCS '
                             'will be used for all pairs) or a dictionary (only pairs present in dictionary will use '
                             'a custom MCS)')

    map_opts = Parser.add_argument_group('General map options', 'General options to control generation of the map')
    map_opts.add_argument('--map_type', type=str, choices=['optimal', 'star', 'wheel'], default=None,
                          help='Type of perturbation map (see manual for more info): optimal (default), star, or '
                               'wheel')
    map_opts.add_argument('--map_runs', type=int, default=None,
                          help='Number of runs (Default: 500, only used for optimal and wheel maps)')
    map_opts.add_argument('--map_communication_frequency', type=int, default=None,
                          help='Communicate every this much steps (Default: 10)')
    map_opts.add_argument('--map_bias', type=str, default=None,
                          help='Bias map toward this/these nodes name (default: no bias; this is required in star and '
                               'connected star maps)')
    map_opts.add_argument('--map_alpha', type=float, default=None,
                          help='Pheromone biasing exponent. Controls the effect of the pheromone on the desirability '
                               'of an edge')
    map_opts.add_argument('--map_beta', type=float, default=None,
                          help='Cost biasing exponent. Controls the effect of the cost on the desirability of an edge')
    map_opts.add_argument('--map_pheromone_intensity', type=float, default=None,
                          help='The intensity of deposited pheromone (Default: 0.1)')
    map_opts.add_argument('--map_evaporating_rate', type=float, default=None,
                          help='How fast the pheromone evaporates (Default: 0.3)')
    map_opts.add_argument('--map_min_desirability', type=float, default=None,
                          help='Minimal desirability of an edge (Default: 0.1)')
    map_opts.add_argument('--map_max_pheromone_deposited', type=float, default=None,
                          help='Deposit at most this much pheromone per run (Default: off)')
    map_opts.add_argument('--map_elitism', type=float, default=None,
                          help='Use this many best solutions to update pheromone matrix (Default: -1: use all)')

    optimal_opts = Parser.add_argument_group('Optimal map options', 'Options to control generation of an optimal map '
                                                                    'via ACO algorithm')
    optimal_opts.add_argument('--optimal_max_path', type=int, default=None, help='Max path length (Default: off)')
    optimal_opts.add_argument('--optimal_perturbation_multiplier', type=float, default=None,
                              help='Multiplier for perturbation score (Default: 20)')
    optimal_opts.add_argument('--optimal_perturbation_exponent', type=float, default=None,
                              help='Exponent for perturbation score (Default: 4)')
    optimal_opts.add_argument('--optimal_length_exponent', type=float, default=None,
                              help='Exponent for length cost (Default: off)')
    optimal_opts.add_argument('--optimal_degree_target', type=int, default=None,
                              help='Constant for degree cost (Default: optimal_min_edges_per_node)')
    optimal_opts.add_argument('--optimal_degree_multiplier', type=float, default=None,
                              help='Multiplier for degree cost (Default: off)')
    optimal_opts.add_argument('--optimal_degree_exponent', type=float, default=None,
                              help='Exponent for degree cost (Default: off)')
    optimal_opts.add_argument('--optimal_min_edges_per_node', type=int, default=None,
                              help='Each edge must have at least this much nodes (Default: 2 = map with closure cycle)')
    optimal_opts.add_argument('--optimal_extra_edge_beta', type=float, default=None,
                              help='Extra edge beta parameter. Larger values allows more edges than the minimum amount '
                                   '(Default: 2)')
    optimal_opts.add_argument('--optimal_unbound_runs', type=float, default=None,
                              help='Minimum number of runs when all edges can be removed (Default: off)')
    optimal_opts.add_argument('--optimal_permanent_edge_threshold', type=float, default=None,
                              help='Edges with this much pheromone become static (Default: off)')
    process_user_input.add_argparse_global_args(Parser)
    arguments = process_user_input.read_options(Parser, unpack_section='generate_perturbation_map')

    progress_data = savestate_util.SavableState(arguments.progress_file)

    if arguments.input is None:
        os_util.local_print('No input files were provided. Please, do so by using --input or input option in your '
                            'configuration file',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(1)

    if isinstance(arguments.map_communication_frequency, int) and arguments.map_communication_frequency > 0:
        comm_freq = arguments.map_communication_frequency
    elif arguments.map_type != 'star':
        os_util.local_print('Could not understand communication frequency (map_communication_frequency) value '
                            '{}. Value must be a positive integer.'
                            ''.format(arguments.map_communication_frequency),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(1)

    custom_user_data = process_custom_mcs(arguments.custom_mcs, savestate=progress_data, verbosity=arguments.verbose)

    # Reads a networkx.Graph from a pickle file
    if not arguments.input and 'ligands_data' in progress_data:

        molecules_dict = progress_data['ligands_data']

        if not isinstance(molecules_dict, dict):
            os_util.local_print('Failed to load molecules from {} (and you did not supply an input file). Cannot '
                                'continue'.format(arguments.progress_file),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)
        elif len(molecules_dict) == 0:
            os_util.local_print('Molecules data in {} is empty (and you did not supply an input file). Cannot continue.'
                                ''.format(arguments.progress_file),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)

        full_thermograph = networkx.DiGraph()

    elif arguments.input:
        # Or reads molecules and prepare a networkx.Graph from it
        molecules_dict = OrderedDict()
        for each_file in arguments.input:
            if arguments.verbose >= 1:
                os_util.local_print('Reading data from file {}'.format(each_file),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)
            file_ext = splitext(each_file)[1]
            if file_ext in ['.smi', '.smiles']:

                mol_supplier = [i for i in rdkit.Chem.SmilesMolSupplier(each_file, titleLine=False)]
                if not mol_supplier[0]:
                    mol_supplier = [i for i in rdkit.Chem.SmilesMolSupplier(each_file, titleLine=True)]

                for index, each_mol in enumerate(mol_supplier):
                    if each_mol is None:
                        if not arguments.no_checks:
                            os_util.local_print('Failed to read molecule #{} from input file {}'
                                                ''.format(index, each_file),
                                                msg_verbosity=os_util.verbosity_level.error,
                                                current_verbosity=arguments.verbose)
                            raise SystemExit(1)
                        else:
                            os_util.local_print('Failed to read molecule #{} from input file {}. Going on.'
                                                ''.format(index, each_file),
                                                msg_verbosity=os_util.verbosity_level.error,
                                                current_verbosity=arguments.verbose)
                            continue
                    else:
                        each_mol = rdkit.Chem.AddHs(each_mol)
                    new_mol_name = mol_util.verify_molecule_name(each_mol, molecules_dict,
                                                                 new_default_name='Mol_{}'
                                                                                  ''.format(len(molecules_dict) + 1),
                                                                 verbosity=arguments.verbose)
                    molecules_dict[new_mol_name] = each_mol
            elif file_ext in ['.mol2', '.mol']:
                if file_ext == '.mol2':
                    each_mol = rdkit.Chem.MolFromMol2File(each_file, removeHs=False)
                else:
                    each_mol = rdkit.Chem.MolFromMolFile(each_file, removeHs=False)
                if each_mol is not None:
                    each_mol = mol_util.process_dummy_atoms(each_mol, verbosity=arguments.verbose)
                    new_mol_name = mol_util.verify_molecule_name(each_mol, molecules_dict,
                                                                 new_default_name='Mol_{}'
                                                                                  ''.format(len(molecules_dict) + 1),
                                                                 verbosity=arguments.verbose)
                    molecules_dict[new_mol_name] = each_mol
                else:
                    if not arguments.no_checks:
                        os_util.local_print('Failed to read molecule from input file {}'.format(each_file),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=arguments.verbose)
                        raise SystemExit(1)
                    else:
                        os_util.local_print('Failed to read molecule from input file {}. Going on.'.format(each_file),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=arguments.verbose)
                        continue
                progress_data['ligands_data'] = {mol_name: {'molecule': rdkit.Chem.PropertyMol.PropertyMol(rdmol)}
                                                 for mol_name, rdmol in molecules_dict.items()}
                progress_data['ligands_data_{}'.format(time.strftime('%d%m%Y_%H%M%S'))] = progress_data['ligands_data']

            else:
                if not arguments.no_checks:
                    os_util.local_print('Failed to read file {}: format {} not recognized.'
                                        ''.format(each_file, file_ext),
                                        msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)
                    raise SystemExit(1)
                else:
                    os_util.local_print('Failed to read file {}: format {} not recognized.'
                                        ''.format(each_file, file_ext),
                                        msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)

        if len(molecules_dict) == 2:
            if arguments.no_checks:
                os_util.local_print('Only two molecules were read. I need at least 3 to construct a meaningful '
                                    'perturbation graph. Because you are running with no_checks, I will go on.'
                                    ''.format(len(molecules_dict)),
                                    msg_verbosity=os_util.verbosity_level.error,
                                    current_verbosity=arguments.verbose)
            else:
                os_util.local_print('Only two molecules were read. I need at least 3 to construct a meaningful '
                                    'perturbation graph. Should you need to use a single pair, please supply '
                                    'perturbation_map directly to prepare_dual_topology.py. Alternatively, rerunning '
                                    'with no_checks will suppress this error and go on.'
                                    ''.format(len(molecules_dict)),
                                    msg_verbosity=os_util.verbosity_level.error,
                                    current_verbosity=arguments.verbose)
                raise SystemExit(1)

        elif len(molecules_dict) == 1:
            os_util.local_print('A single molecule was read. With a single molecule, I cannot go on.'
                                ''.format(len(molecules_dict)),
                                msg_verbosity=os_util.verbosity_level.error,
                                current_verbosity=arguments.verbose)
            raise SystemExit(1)

        if arguments.verbose >= 1:
            os_util.local_print('These are the molecules read from input files {}: {}'
                                ''.format(', '.join(arguments.input), ', '.join(molecules_dict.keys())),
                                msg_verbosity=os_util.verbosity_level.info,
                                current_verbosity=arguments.verbose)

        full_thermograph = networkx.DiGraph()

    else:
        os_util.local_print('You did not provide input molecules and no molecules could be read from your progress '
                            'file {}. Cannot continue.'
                            ''.format(arguments.progress_file),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(1)

    if arguments.map_type == 'optimal':
        fill_thermograph(full_thermograph, molecules_dict, use_hs=arguments.use_hs, threads=arguments.threads,
                         custom_mcs=custom_user_data, savestate=progress_data, verbosity=arguments.verbose)

        # Prepare ant colony
        ant_colony = all_classes.AntSolver(network_graph=full_thermograph,
                                           alpha=arguments.map_alpha, beta=arguments.map_beta,
                                           path_threshold=arguments.optimal_max_path,
                                           perturbation_multiplier=arguments.optimal_perturbation_multiplier,
                                           perturbation_exponent=arguments.optimal_perturbation_exponent,
                                           length_exponent=arguments.optimal_length_exponent,
                                           degree_target=arguments.optimal_degree_target,
                                           degree_multiplier=arguments.optimal_degree_multiplier,
                                           degree_exponent=arguments.optimal_degree_exponent,
                                           pheromone_intensity=arguments.map_pheromone_intensity,
                                           evaporating_rate=arguments.map_evaporating_rate,
                                           min_edge_desirability=arguments.map_min_desirability,
                                           min_unbound=arguments.optimal_unbound_runs,
                                           permanent_edge_threshold=arguments.optimal_permanent_edge_threshold,
                                           extra_edge_beta=arguments.optimal_extra_edge_beta,
                                           max_pheromone_deposited=arguments.map_max_pheromone_deposited,
                                           min_edges_per_node=arguments.optimal_min_edges_per_node,
                                           algorithm='modified')

        run_workers(ant_colony=ant_colony, n_runs=arguments.map_runs, n_threads=arguments.threads,
                    comm_freq=arguments.map_communication_frequency, elitism=arguments.map_elitism,
                    verbosity=arguments.verbose)

        os_util.local_print('Best map found: Cost: {}; Num edges: {}'
                            ''.format(ant_colony.best_solution.cost,
                                      ant_colony.best_solution.graph.number_of_edges),
                            msg_verbosity=os_util.verbosity_level.default,
                            current_verbosity=arguments.verbose)

        if 'thermograph' not in progress_data:
            progress_data['thermograph'] = {}
        archive = {'runtype': 'optimal', 'bias': arguments.map_bias, 'input_molecules': molecules_dict.copy(),
                   'best_solution': ant_colony.best_solution.graph, 'optimization_data': ant_colony}
        progress_data['thermograph']['run_{}'.format(time.strftime('%d%m%Y_%H%M%S'))] = archive
        progress_data.save_data()

    elif arguments.map_type == 'star':
        center_molecule = test_center_molecule(arguments.map_bias, molecules_dict, arguments.verbose)

        # Fill the graph with edges connecting all molecules to the center
        pairlist = [[center_molecule, each_mol] for each_mol in molecules_dict if each_mol != center_molecule]
        fill_thermograph(full_thermograph, molecules_dict, pairlist=pairlist, use_hs=arguments.use_hs,
                         threads=arguments.threads, savestate=progress_data, custom_mcs=custom_user_data,
                         verbosity=arguments.verbose)

        if 'thermograph' not in progress_data:
            progress_data['thermograph'] = {}
        archive = {'runtype': 'star', 'bias': center_molecule, 'input_molecules': molecules_dict.copy(),
                   'best_solution': full_thermograph.copy()}
        progress_data['thermograph']['run_{}'.format(time.strftime('%d%m%Y_%H%M%S'))] = archive
        progress_data.save_data()

    elif arguments.map_type == 'wheel':
        center_molecule = test_center_molecule(arguments.map_bias, molecules_dict, arguments.verbose)

        wheel_mols = molecules_dict.copy()
        del wheel_mols[center_molecule]

        fill_thermograph(full_thermograph, wheel_mols, use_hs=arguments.use_hs, threads=arguments.threads,
                         savestate=progress_data, custom_mcs=custom_user_data, verbosity=arguments.verbose)

        # Prepare ant colony
        ant_colony = all_classes.AntSolver(network_graph=full_thermograph,
                                           alpha=arguments.map_alpha, beta=arguments.map_beta,
                                           perturbation_multiplier=1.0,
                                           perturbation_exponent=1.0,
                                           length_exponent=0.0,
                                           degree_multiplier=0.0,
                                           pheromone_intensity=arguments.map_pheromone_intensity,
                                           evaporating_rate=arguments.map_evaporating_rate,
                                           min_edge_desirability=arguments.map_min_desirability,
                                           min_unbound=-1,
                                           permanent_edge_threshold=-1,
                                           extra_edge_beta=0.0,
                                           max_pheromone_deposited=arguments.map_max_pheromone_deposited,
                                           min_edges_per_node=2,
                                           algorithm='classic')

        run_workers(ant_colony=ant_colony, n_runs=arguments.map_runs, n_threads=arguments.threads,
                    comm_freq=arguments.map_communication_frequency, elitism=arguments.map_elitism,
                    verbosity=arguments.verbose)

        # Add edges connecting all molecules to the center
        pairlist = [[each_mol, center_molecule] for each_mol in molecules_dict if each_mol != center_molecule]

        full_thermograph = networkx.DiGraph()
        full_thermograph.add_edges_from(ant_colony.best_solution.graph.edges(data=True))
        fill_thermograph(full_thermograph, molecules_dict, pairlist=pairlist, use_hs=arguments.use_hs,
                         threads=arguments.threads, savestate=progress_data, custom_mcs=custom_user_data,
                         verbosity=arguments.verbose)

        if 'thermograph' not in progress_data:
            progress_data['thermograph'] = {}
        archive = {'runtype': 'wheel', 'bias': arguments.map_bias, 'input_molecules': molecules_dict.copy(),
                   'best_solution': full_thermograph.copy(), 'optimization_data': ant_colony}
        progress_data['thermograph']['run_{}'.format(time.strftime('%d%m%Y_%H%M%S'))] = archive
        progress_data.save_data()

    else:
        os_util.local_print('Map type {} not understood. Please, select one of "optimal", "star" or "wheel" (see '
                            'manual)'.format(arguments.map_type),
                            msg_verbosity=os_util.verbosity_level.error,
                            current_verbosity=arguments.verbose)
        raise SystemExit(1)

    # Save the current solution data
    progress_data['thermograph']['last_solution'] = archive
    progress_data.save_data()

    if arguments.plot:
        import matplotlib

        matplotlib.use('svg')
        import matplotlib.pyplot
        import networkx.drawing

        if arguments.map_type in ['optimal', 'wheel']:

            if arguments.map_type == 'optimal':

                node_position = networkx.drawing.spring_layout(ant_colony.best_solution.graph,
                                                               weight='cost', iterations=500)

                if arguments.optimal_permanent_edge_threshold > 0:
                    static_egdes = ant_colony.complete_network.copy()
                    not_static_list = [each_edge if each_edge[2] < arguments.optimal_permanent_edge_threshold else None
                                       for each_edge in static_egdes.edges(data='desirability')]
                    for each_edge in not_static_list:
                        if each_edge is None:
                            continue
                        static_egdes.remove_edge(each_edge[0], each_edge[1])
                    networkx.drawing.draw(static_egdes, with_labels=True, pos=node_position, edge_color='#A0CBE2',
                                          width=4)
                networkx.drawing.draw(ant_colony.best_solution.graph, with_labels=True, pos=node_position)
                labels = networkx.get_edge_attributes(ant_colony.best_solution.graph, 'perturbed_atoms')
                networkx.draw_networkx_edge_labels(ant_colony.best_solution.graph, node_position, edge_labels=labels)
            else:

                # FIXME: fix this layout
                outer_edges = deepcopy(full_thermograph)
                outer_edges.remove_node(center_molecule)
                node_position = networkx.drawing.circular_layout(outer_edges, center=[0.0, 0.0])
                node_position[center_molecule] = [0.0, 0.0]

                networkx.drawing.draw(full_thermograph, with_labels=True, pos=node_position)
                labels = networkx.get_edge_attributes(full_thermograph, 'perturbed_atoms')
                networkx.draw_networkx_edge_labels(full_thermograph, node_position, edge_labels=labels)

            matplotlib.pyplot.savefig('best_graph.svg')
            matplotlib.pyplot.clf()

            color_map = [each_edge[2] for each_edge in ant_colony.complete_network_undirect.edges(data='desirability')]
            if arguments.optimal_permanent_edge_threshold > 0:
                color_map = [each_edge if each_edge <= arguments.optimal_permanent_edge_threshold
                             else arguments.optimal_permanent_edge_threshold for each_edge in color_map]

            networkx.drawing.draw(ant_colony.complete_network, with_labels=True, node_color='#A0CBE2', width=4,
                                  edge_cmap=matplotlib.pyplot.cm.Greys, edge_color=color_map,
                                  pos=networkx.circular_layout(ant_colony.complete_network))
            matplotlib.pyplot.savefig('full_graph.svg')

            subplots_fig, subplots_axs = matplotlib.pyplot.subplots(2, 2, figsize=(10, 10))
            subplots_axs[0, 0].set_title('Score per run (log)')
            subplots_axs[0, 0].semilogy(ant_colony.cost_list, 'b-')
            subplots_axs[0, 1].set_title('Score per run (linear, decomposed)')

            cost_decomposition_matrix = {'total': [], 'length': [], 'perturbation': [], 'degree': []}
            for each_solution in ant_colony.solutions:
                cost_data = ant_colony.calculate_network_cost(each_solution.graph, decompose=True)
                cost_decomposition_matrix['total'].append(cost_data['total'])
                cost_decomposition_matrix['length'].append(cost_data['length'])
                cost_decomposition_matrix['perturbation'].append(cost_data['perturbation'])
                cost_decomposition_matrix['degree'].append(cost_data['degree'])

            subplots_axs[0, 1].plot(cost_decomposition_matrix['total'], label='Total cost', color='#000000')
            subplots_axs[0, 1].plot(cost_decomposition_matrix['length'], label='Length cost', color='#CC6666')
            subplots_axs[0, 1].plot(cost_decomposition_matrix['perturbation'], label='Perturb. cost', color='#66CC66')
            subplots_axs[0, 1].plot(cost_decomposition_matrix['degree'], label='Degree cost', color='#6666CC')
            subplots_axs[0, 1].legend()

            subplots_axs[1, 0].set_title('Pheromone multiplier')
            subplots_axs[1, 0].hist([each_solution.pheromone_multiplier for each_solution in ant_colony.solutions])
            subplots_axs[1, 1].set_title('Pheromone histogram')
            subplots_axs[1, 1].hist(color_map)

            subplots_fig.savefig('result_plot.svg')

        else:
            outer_edges = deepcopy(full_thermograph)
            outer_edges.remove_node(center_molecule)
            node_position = networkx.drawing.circular_layout(outer_edges, center=[0.0, 0.0])
            node_position[center_molecule] = [0.0, 0.0]

            networkx.drawing.draw(full_thermograph, with_labels=True, pos=node_position)
            matplotlib.pyplot.savefig('best_graph.svg')
