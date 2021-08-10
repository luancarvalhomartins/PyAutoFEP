#! /usr/bin/env python3
#
#  merge_topologies.py
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
import os
import re
from copy import deepcopy
import time
from collections.abc import Callable

import rdkit.Chem
import rdkit.Chem.rdFMCS
import rdkit.Chem.rdForceFieldHelpers
import rdkit.ForceField.rdForceField
from rdkit.Chem.AllChem import ConstrainedEmbed, DeleteSubstructs, ShapeProtrudeDist, ShapeTanimotoDist, AlignMol
from rdkit.Chem.AllChem import GetMolFrags

import mol_util
import all_classes
import os_util
import savestate_util


def constrained_embed_forcefield(molecule, core, core_conf_id=-1, atom_map=None, num_conformations=1, randomseed=2342,
                                 restraint_steps=(10.0, 50.0, 100.0), minimization_steps=5,
                                 force_field=rdkit.Chem.AllChem.UFFGetMoleculeForceField, verbosity=0, **kwargs):
    """ Use force field minimization to constrain molecule to core, without Embed code

    :param rdkit.Chem.Mol molecule: mobile molecule to be embed
    :param rdkit.Chem.Mol core: reference molecule
    :param int core_conf_id: use this core conformation
    :param list atom_map: use this atoms to map molecule -> core atoms, default get match automatically
    :param int num_conformations: generate this many conformations
    :param int randomseed: random seed to the conformer generator
    :param list restraint_steps: apply restraints sequentially from this list, default (10.0, 50.0, 100.0)
    :param int minimization_steps: run this much minimization steps
    :param function force_field: get force field function, default UFF
    :param int verbosity: set verbosity
    :rtype: rdkit.Chem.Mol
    """

    default_values = {'randomSeed': randomseed, 'ignoreSmoothingFailures': True, 'enforceChirality': True,
                      'maxAttempts': 50, 'boxSizeMult': 5.0, 'randNegEig': True, 'numZeroFail': 1, 'forceTol': 1.0e-3,
                      'energyTol': 1.0e-4, 'useExpTorsionAnglePrefs': True, 'useBasicKnowledge': True, 'maxIters': 50,
                      'minLen': 0, 'maxLen': 0, 'maxIts': 200}
    [kwargs.setdefault(key, value) for key, value in default_values.items()]
    required_values = {'useRandomCoords': True, 'clearConfs': False}
    [kwargs.__setitem__(key, value) for key, value in required_values.items()]

    if not atom_map:
        adjusted_core = mol_util.adjust_query_properties(rdkit.Chem.RemoveHs(rdkit.Chem.Mol(core)), verbosity=verbosity)
        atom_map = list(enumerate(rdkit.Chem.RemoveHs(molecule).GetSubstructMatch(adjusted_core)))
        if not atom_map:
            os_util.local_print('Failed to match molecule {} to core {} in constrained_embed_forcefield. Cannot '
                                'continue'.format(rdkit.Chem.MolToSmiles(molecule),
                                                  rdkit.Chem.MolToSmiles(adjusted_core)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError("Molecule doesn't match the core")

    for this_conformations in range(num_conformations):
        this_conf_id = rdkit.Chem.AllChem.EmbedMolecule(molecule, randomSeed=randomseed,
                                                        ignoreSmoothingFailures=kwargs['ignoreSmoothingFailures'],
                                                        clearConfs=kwargs['clearConfs'],
                                                        useRandomCoords=kwargs['useRandomCoords'],
                                                        enforceChirality=kwargs['enforceChirality'],
                                                        maxAttempts=kwargs['maxAttempts'],
                                                        boxSizeMult=kwargs['boxSizeMult'],
                                                        randNegEig=kwargs['randNegEig'],
                                                        numZeroFail=kwargs['numZeroFail'],
                                                        forceTol=kwargs['forceTol'],
                                                        useExpTorsionAnglePrefs=kwargs['useExpTorsionAnglePrefs'],
                                                        useBasicKnowledge=kwargs['useBasicKnowledge'])
        if this_conf_id < 0:
            os_util.local_print('Failed to embed molecule {} in constrained_embed_forcefield. Cannot '
                                'continue.'.format(rdkit.Chem.MolToSmiles(molecule)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)

        # align the embedded conformation onto the core:
        rdkit.Chem.AllChem.AlignMol(molecule, core, refCid=core_conf_id, prbCid=this_conf_id, atomMap=atom_map,
                                    maxIters=kwargs['maxIters'])
        sanitize_return = rdkit.Chem.SanitizeMol(molecule, catchErrors=True)
        if sanitize_return != 0:
            os_util.local_print('Could not sanitize molecule {} (SMILES="{}")\nError {} when running '
                                'rdkit.Chem.SanitizeMol.'
                                ''.format(molecule.GetProp("_Name"), rdkit.Chem.MolToSmiles(molecule), sanitize_return),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        ff = force_field(molecule, confId=this_conf_id)
        conf = core.GetConformer(core_conf_id)

        for this_restraint in restraint_steps:
            for dest_atom, core_atom in atom_map:
                p = conf.GetAtomPosition(core_atom)
                p_idx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                ff.AddDistanceConstraint(p_idx, dest_atom, minLen=kwargs['minLen'], maxLen=kwargs['maxLen'],
                                         forceConstant=this_restraint)
            ff.Initialize()
            ff.Minimize(maxIts=kwargs['maxIts'], energyTol=kwargs['energyTol'], forceTol=kwargs['forceTol'])
            for _ in range(minimization_steps):
                if ff.Minimize(maxIts=kwargs['maxIts'], energyTol=kwargs['energyTol'], forceTol=kwargs['forceTol']):
                    break

        # Align molecule again
        rdkit.Chem.AllChem.AlignMol(molecule, core, refCid=core_conf_id, prbCid=this_conf_id, atomMap=atom_map,
                                    maxIters=kwargs['maxIters'])

    return molecule


def constrained_embed_shapeselect(molecule, target, core_conf_id=-1, matching_atoms=None, randomseed=2342,
                                  num_conformers=200, volume_function='tanimoto', rigid_molecule_threshold=1,
                                  num_threads=0, mcs=None, save_state=None, verbosity=0, **kwargs):
    """ Embed a molecule to target used a constrained core and maximizing volume similarity, as measured by
    volume_function. Symmetry will be taken in account.

    :param rdkit.Chem.Mol molecule: molecule to be embed
    :param rdkit.Chem.Mol target: the molecule to use as a source of constraints
    :param int core_conf_id: id of the core conformation to use (default: detect)
    :param dict matching_atoms: dict of atoms in molecule matching target. If None (default) find_mcs will be used to
                                generate a match
    :param int randomseed: seed to EmbedMolecule and EmbedMultipleConfs
    :param int num_conformers: generate this much trial conformers to find a best shape match
    :param [str, function] volume_function: use this function for calculate shape similarity ('protude' or 'tanimoto'
                                            (default)), or a function (so that f(mol1: rdkit.Chem.Mol,
                                            mol2: rdkit.Chem.Mol, conformation1: int, conformation2: int) -> float)
    :param int rigid_molecule_threshold: consider a molecule to be rigid if up to this many heavy atoms are not
        constrained (default 1; -1: molecule is always flexible)
    :param int num_threads: use this many threads during conformer generation (0: max supported)
    :param str mcs: use this SMARTS as common core to merge molecules
    :param savestate_util.SavableState save_state: saved state data
    :param int verbosity: set verbosity level
    :rtype: rdkit.Chem.Mol
    """

    os_util.local_print('Entering constrained_embed_dualmol: pseudomolecule={}, target={}, core_conf_id={}, '
                        'randomseed={}, num_conformers={}, volume_function={}, rigid_molecule_threshold={}, '
                        'num_threads={}, verbosity={}'
                        ''.format(molecule, target, core_conf_id, randomseed, num_conformers,
                                  volume_function, rigid_molecule_threshold, num_threads, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    default_values = {'maxAttempts': 50, 'numConfs': num_conformers, 'randomSeed': randomseed, 'useRandomCoords': True,
                      'clearConfs': True, 'ignoreSmoothingFailures': True, 'useExpTorsionAnglePrefs': True,
                      'enforceChirality': True, 'boxSizeMult': 5.0, 'numThreads': num_threads, 'gridSpacing': 0.5,
                      'vdwScale': 0.8, 'stepSize': 0.25, 'maxLayers': -1, 'ignoreHs': True}
    [kwargs.setdefault(key, value) for key, value in default_values.items()]

    if kwargs['clearConfs']:
        # Clear conformer data
        molecule.RemoveAllConformers()

    if matching_atoms is None:
        if mcs is not None:
            this_mcs = mcs
        else:
            # Use find_mcs
            this_mcs = find_mcs([rdkit.Chem.RemoveHs(molecule), rdkit.Chem.RemoveHs(target)],
                                verbosity=verbosity, savestate=save_state, **kwargs).smartsString

        core_mol = rdkit.Chem.MolFromSmarts(this_mcs)
        if core_mol is None:
            os_util.local_print('Could not detect/convert common core between target mol and {}\nError when running'
                                ' rdkit.Chem.MolFromSmarts\nThis is the Smarts which failed to be converted: {}'
                                ''.format(molecule.GetProp("_Name"), this_mcs),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        if core_mol.GetNumHeavyAtoms() == molecule.GetNumHeavyAtoms():
            try:
                target_name = target.GetProp('_Name')
            except KeyError:
                target_name = str(target)
            os_util.local_print('The detected or supplied core between molecules {} (SMILES="{}") and {} (SMILES="{}") '
                                'has the same number of heavy atoms as molecule {} ({} heavy atoms). Falling back to '
                                'constrained_embed_forcefield with num_conformations=1.'
                                ''.format(molecule.GetProp("_Name"), rdkit.Chem.MolToSmiles(molecule), target_name,
                                          rdkit.Chem.MolToSmiles(target), molecule.GetProp("_Name"),
                                          molecule.GetNumHeavyAtoms()),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

            atom_map = list(zip(get_substruct_matches_fallback(molecule, core_mol)[0],
                                get_substruct_matches_fallback(target, core_mol)[0]))
            constrained_embed_forcefield(molecule, target, core_conf_id=core_conf_id, randomseed=randomseed,
                                         atom_map=atom_map, num_conformations=1, **kwargs)
            return molecule

        sanitize_return = rdkit.Chem.SanitizeMol(core_mol, catchErrors=True)
        if sanitize_return != 0:
            os_util.local_print('Could not sanitize common core between target mol and {}\nError {} when running '
                                'rdkit.Chem.SanitizeMol\nThis is the molecule representing the common core between '
                                'structures (without sanitization): {}'
                                ''.format(molecule.GetProp("_Name"), sanitize_return,
                                          rdkit.Chem.MolToSmiles(core_mol)),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

        temp_core_structure = mol_util.loose_replace_side_chains(target, core_mol, use_chirality=True)
        if temp_core_structure is None:
            try:
                target_name = target.GetProp('_Name')
            except KeyError:
                target_name = target.__str__()
            os_util.local_print('Could not process the core structure to embed while working with the molecule '
                                '{} (SMILES={}). mol_util.loose_replace_side_chains(target, core_mol) failed. '
                                'core_mol: {} (SMARTS={}) target: {} (SMARTS={})'
                                ''.format(molecule.GetProp('_Name'),
                                          rdkit.Chem.MolToSmiles(molecule), core_mol,
                                          rdkit.Chem.MolToSmarts(core_mol),
                                          target_name, rdkit.Chem.MolToSmiles(target)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        else:
            core_mol = temp_core_structure

        # Remove * atoms from common core
        core_mol = DeleteSubstructs(core_mol, rdkit.Chem.MolFromSmiles('*'))
        if core_mol is None:
            os_util.local_print('Could not delete side chains between target mol and {}\n\t'
                                'DeleteSubstructs(core_mol, rdkit.Chem.MolFromSmiles("*")) failed.\n\t'
                                'core_mol: {} (SMARTS={})'
                                ''.format(molecule.GetProp("_Name"),
                                          core_mol, rdkit.Chem.MolToSmarts(core_mol)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        os_util.local_print('This is the Smiles representation of the common core: {}'
                            ''.format(rdkit.Chem.MolToSmiles(core_mol)),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        core_mol = mol_util.adjust_query_properties(core_mol, verbosity=verbosity)
        # Prepare a coordinate map to supply to EmbedMultipleConfs
        matches = get_substruct_matches_fallback(molecule, core_mol, verbosity=verbosity, kwargs=kwargs)

        core_conf = core_mol.GetConformer(core_conf_id)

        for match in matches:
            if len(match) + rigid_molecule_threshold >= molecule.GetNumHeavyAtoms():
                # If there are too few atoms to be sampled, just generate a single conformation. This will be often
                # triggered when the perturbations are small or we are constraining to a reference of the same molecule

                num_conformers = 1

                os_util.local_print('The endpoint {} would have {} constrained atoms and {} not constrained ones. A '
                                    'single conformation will be generated (rigid_molecule_threshold = {})'
                                    ''.format(molecule.GetProp('_Name'),
                                              len(match), molecule.GetNumHeavyAtoms() - len(match),
                                              rigid_molecule_threshold),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
            else:
                num_conformers = kwargs['numConfs']

            coord_map = {endpoint_atom: core_conf.GetAtomPosition(core_atom)
                         for core_atom, endpoint_atom in enumerate(match)}

            num_conformers_before = molecule.GetNumConformers()
            try:
                rdkit.Chem.AllChem.EmbedMultipleConfs(molecule, maxAttempts=kwargs['maxAttempts'],
                                                      numConfs=num_conformers, randomSeed=kwargs['randomSeed'],
                                                      useRandomCoords=kwargs['useRandomCoords'],
                                                      clearConfs=False, coordMap=coord_map,
                                                      ignoreSmoothingFailures=kwargs['ignoreSmoothingFailures'],
                                                      useExpTorsionAnglePrefs=kwargs['useExpTorsionAnglePrefs'],
                                                      enforceChirality=kwargs['enforceChirality'],
                                                      boxSizeMult=kwargs['boxSizeMult'],
                                                      numThreads=kwargs['numThreads'])
            except RuntimeError:
                pass

            if molecule.GetNumConformers() == num_conformers_before:
                os_util.local_print('EmbedMultipleConfs failed to generate conformations. Retrying with force field '
                                    'optimization-based constrained embed.',
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                atom_map = [(endpoint_atom, core_atom) for core_atom, endpoint_atom in enumerate(match)]
                constrained_embed_forcefield(molecule, core_mol, core_conf_id=core_conf_id, randomseed=randomseed,
                                             atom_map=atom_map, **kwargs)
        if molecule.GetNumConformers() == 0:
            os_util.local_print('Failed to generate conformations to molecule {}. Cannot continue.'
                                ''.format(molecule.GetProp("_Name")),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(-1)

    else:
        coord_map = {endpoint_atom: target.GetConformer(core_conf_id).GetAtomPosition(target_atom)
                     for target_atom, endpoint_atom in matching_atoms.items()}

        rdkit.Chem.AllChem.EmbedMultipleConfs(molecule, maxAttempts=kwargs['maxAttempts'], numConfs=kwargs['numConfs'],
                                              randomSeed=kwargs['randomSeed'],
                                              useRandomCoords=kwargs['useRandomCoords'],
                                              clearConfs=kwargs['clearConfs'], coordMap=coord_map,
                                              ignoreSmoothingFailures=kwargs['ignoreSmoothingFailures'],
                                              useExpTorsionAnglePrefs=kwargs['useExpTorsionAnglePrefs'],
                                              enforceChirality=kwargs['enforceChirality'],
                                              boxSizeMult=kwargs['boxSizeMult'], numThreads=kwargs['numThreads'])

    if molecule.GetNumConformers() == 0:
        os_util.local_print('Failed to generate conformations to molecule {}. Cannot continue'
                            ''.format(molecule.GetProp("_Name")),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(-1)

    os_util.local_print('{} conformations were generated to molecule {}'
                        ''.format(molecule.GetNumConformers(),
                                  molecule.GetProp("_Name")),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Use this volume-based comparison function to rank conformation pairs
    if volume_function == 'protude':
        volume_function = ShapeProtrudeDist
    elif volume_function == 'tanimoto':
        volume_function = ShapeTanimotoDist
    elif isinstance(volume_function, Callable):
        # volume_function is already the function, do nothing
        pass
    else:
        os_util.local_print('Shape similarity comparison "{}" not understood. Please, select from "protude" or '
                            '"tanimoto".'.format(volume_function),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    rms_list = [volume_function(molecule, target, conformer.GetId(), core_conf_id, gridSpacing=kwargs['gridSpacing'],
                                vdwScale=kwargs['vdwScale'], stepSize=kwargs['stepSize'], maxLayers=kwargs['maxLayers'],
                                ignoreHs=kwargs['ignoreHs'])
                for conformer in molecule.GetConformers() if conformer.Is3D()]

    # Find the index for best solution
    best_solution = rms_list.index(min(rms_list))

    os_util.local_print('This is the statistics of volume similarity of the conformations generated in'
                        ' constrained_embed_shapeselect, as scored by function {}:\n Number of conformers: {}\n'
                        'Mean similarity: {}\nMost similar: {} (item: {})'
                        ''.format(volume_function, len(rms_list), sum(rms_list) / len(rms_list),
                                  max(rms_list), best_solution),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    [molecule.RemoveConformer(conformer.GetId()) for conformer in molecule.GetConformers()
     if conformer.GetId() != best_solution]

    return molecule


def constrained_embed_dualmol(pseudomolecule, target, core_conf_id=-1, pseudomol_conf_id=-1, randomseed=2342,
                              num_conformers=10, volume_function='tanimoto', rigid_molecule_threshold=1,
                              num_threads=0, mcs=None, savestate=None, verbosity=0):
    """ Generates an embedding of a dual-topology pseudomolecule to target using MCS,

    :param all_classes.MergedTopologies pseudomolecule: pseudomolecule to be embed
    :param rdkit.Chem.Mol target: the molecule to use as a source of constraints
    :param int core_conf_id: id of the core conformation to use (default: detect)
    :param int pseudomol_conf_id: id of the pseudomolecule conformation to save to (default: detect)
    :param int randomseed: seed for rdkit.Chem.EmbedMolecule (only used if num_conformers == 1)
    :param int num_conformers: generate this much trial conformers to find a best shape match
    :param str volume_function: use this function for calculate shape similarity ('protude' or 'tanimoto' (default))
    :param int rigid_molecule_threshold: consider a molecule to be rigid if up to this many heavy atoms are not
                                         constrained (-1: off)
    :param int num_threads: use this many threads during conformer generation (0: max supported)
    :param str mcs: use this SMARTS as common core to merge molecules
    :param savestate_util.SavableState savestate: saved state data
    :param int verbosity: set verbosity level
    :rtype: all_classes.MergedTopologies
    """

    os_util.local_print('Entering constrained_embed_dualmol: pseudomolecule={}, target={}, core_conf_id={}, '
                        'pseudomol_conf_id={}, randomseed={}, num_conformers={}, volume_function={}, '
                        'rigid_molecule_threshold={}, num_threads={}, verbosity={}'
                        ''.format(pseudomolecule, target, core_conf_id, pseudomol_conf_id, randomseed, num_conformers,
                                  volume_function, rigid_molecule_threshold, num_threads, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Iterate over states B and A, use EmbedMultipleConfs to embed num_conformers possible structures to
    # core_structure

    # Prepare a temporary copy of molecules A and B, but remove original conformers to make sure we don't get one
    # of them in the rms_dict
    temp_mol_a = rdkit.Chem.Mol(pseudomolecule.molecule_a)
    temp_mol_b = rdkit.Chem.Mol(pseudomolecule.molecule_b)

    temp_mol_a = constrained_embed_shapeselect(temp_mol_a, target, core_conf_id=core_conf_id, randomseed=randomseed,
                                               num_conformers=num_conformers, volume_function=volume_function,
                                               rigid_molecule_threshold=rigid_molecule_threshold,
                                               num_threads=num_threads, mcs=mcs, save_state=savestate,
                                               verbosity=verbosity)
    temp_mol_b = constrained_embed_shapeselect(temp_mol_b, target, core_conf_id=core_conf_id, randomseed=randomseed,
                                               num_conformers=num_conformers, volume_function=volume_function,
                                               rigid_molecule_threshold=rigid_molecule_threshold,
                                               num_threads=num_threads, mcs=mcs, save_state=savestate,
                                               verbosity=verbosity)

    # Copy coordinates of best conformers to pseudomolecule's molecules A and B
    pseudomolecule.molecule_a.RemoveAllConformers()
    pseudomolecule.molecule_b.RemoveAllConformers()

    pseudomolecule.molecule_a.AddConformer(temp_mol_a.GetConformer())
    pseudomolecule.molecule_b.AddConformer(temp_mol_b.GetConformer())

    # Copy coordinates of best conformers to pseudomolecule.dual_molecule
    for new_each_endpoint_molecule in [temp_mol_a, temp_mol_b]:

        # Unless save_state = None, this MCS will be loaded from it
        if mcs is not None:
            this_mcs = mcs
        else:
            this_mcs = find_mcs([rdkit.Chem.RemoveHs(new_each_endpoint_molecule), rdkit.Chem.RemoveHs(target)],
                                verbosity=verbosity, savestate=savestate).smartsString
        core_structure = rdkit.Chem.MolFromSmarts(this_mcs)

        core_structure = mol_util.adjust_query_properties(core_structure, verbosity=verbosity)
        translation_list = new_each_endpoint_molecule.GetSubstructMatch(core_structure)
        this_conformation = pseudomolecule.dual_molecule.GetConformer(pseudomol_conf_id)
        for targetmol_index, dual_topology_index in enumerate(translation_list):
            this_conformation.SetAtomPosition(dual_topology_index,
                                              new_each_endpoint_molecule.GetConformer().GetAtomPosition(
                                                  targetmol_index))

    return pseudomolecule


def dualmol_to_pdb_block(pseudomolecule, molecule_name=None, confId=-1, verbosity=0):
    """ Returns a PDB block for a pseudomolecule.

    :param MergedTopologies pseudomolecule: pseudomolecule to be embed
    :param molecule_name: use this as molecule name (this will not overwrite data on pseudomolecule, only output PDB
                          will be affected)
    :param int confId: selects which conformation to output (-1 = default)
    :param int verbosity: control verbosity level
    :rtype: str
    """

    os_util.local_print('Entering dualmol_to_pdb_block(pseudomolecule={}, verbosity={}, confId={})'
                        ''.format(pseudomolecule, verbosity, confId), msg_verbosity=os_util.verbosity_level.debug,
                        current_verbosity=verbosity)

    # Prepare PDB, do not print CONECT records
    flavor = (2 | 8)
    molecule_a_pdb = rdkit.Chem.MolToPDBBlock(pseudomolecule.molecule_a, confId=confId, flavor=flavor).split('\n')

    molecule_b_pdb = [each_line
                      for each_line in rdkit.Chem.MolToPDBBlock(pseudomolecule.molecule_b,
                                                                confId=confId, flavor=flavor).split('\n')
                      if each_line.find('HETATM') == 0 or each_line.find('ATOM') == 0]

    # Suppress END record and empty line after molecule A
    # TODO edit COMPND record?
    return_list = molecule_a_pdb[:-2]

    core_structure = mol_util.adjust_query_properties(pseudomolecule.common_core_mol, verbosity=verbosity)

    # Compute atoms present only in topology B
    only_in_b = [each_atom.GetIdx()
                 for each_atom in pseudomolecule.molecule_b.GetAtoms()
                 if each_atom.GetIdx() not in
                 pseudomolecule.molecule_b.GetSubstructMatch(core_structure)]

    os_util.local_print('These are the atoms present only in topology B: {}'.format(only_in_b),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    # Iterate over molecule_b_pdb, adding atoms present only in B to return_list. Use Idx + 1 to match PDB numbering
    for each_atom, new_index in zip((each_line for idx, each_line in enumerate(molecule_b_pdb) if idx in only_in_b),
                                    range(pseudomolecule.molecule_a.GetNumAtoms() + 1,
                                          pseudomolecule.molecule_a.GetNumAtoms() + len(only_in_b) + 1)):
        new_atom_line = each_atom[:6] + '{:>5}'.format(new_index) + each_atom[11:]
        return_list.append(new_atom_line)

    if molecule_name:
        temp_pdb_data = []
        for each_line in return_list:
            if each_line.find('HETATM') == 0 or each_line.find('ATOM') == 0:
                each_line = each_line[:17] + '{:>3}'.format(molecule_name) + each_line[20:]
            temp_pdb_data.append(each_line)
        return_list = temp_pdb_data

    return_list.extend(['TER', '\n'])

    return '\n'.join(return_list)


def merge_topologies(molecule_a, molecule_b, file_topology1, file_topology2, no_checks=False, savestate=None, mcs=None,
                     verbosity=0, **kwargs):
    """ Reads two molecule files and topology files, and merge them into a dual topology structure

    :param rdkit.Chem.rdkit molecule_a: molecule A
    :param rdkit.Chem.rdkit molecule_b: molecule B
    :param list file_topology1: GROMACS-compatible topology of molecule 1
    :param list file_topology2: GROMACS-compatible topology of molecule 2
    :param bool no_checks: ignore all tests and try to go on
    :param savestate_util.SavableState savestate: saved state data
    :param str mcs: use this SMARTS as common core to merge molecules
    :param int verbosity: controls verbosity level
    :rtype: MergedTopologies
    """

    molecule1 = mol_util.process_dummy_atoms(molecule_a, verbosity=verbosity)
    molecule2 = mol_util.process_dummy_atoms(molecule_b, verbosity=verbosity)

    os_util.local_print('Molecule 1 name is {}; molecule 2 name is {}'
                        ''.format(molecule1.GetProp("_Name"), molecule2.GetProp("_Name")),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    topology1 = all_classes.TopologyData(file_topology1, verbosity=verbosity)
    if topology1.num_molecules == 0:
        os_util.local_print('Failed to read topology data from {}. Please, check your input file.'
                            ''.format(file_topology1),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    topology2 = all_classes.TopologyData(file_topology2, verbosity=verbosity)
    if topology2.num_molecules == 0:
        os_util.local_print('Failed to read topology data from {}. Please, check your input file.'
                            ''.format(file_topology2),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    # Do some tests
    if not no_checks:

        if molecule1.GetProp("_Name") == molecule2.GetProp("_Name"):
            os_util.local_print('Molecules A ({}) and B ({}) have the same name {}. Check your input files. Names are '
                                'read from mol2 or, if it fails, read from filaname.'
                                ''.format(rdkit.Chem.MolToSmiles(molecule_a),
                                          rdkit.Chem.MolToSmiles(molecule_b),
                                          molecule1.GetProp("_Name")),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError('molecules names are equal')

        defaults_pattern = re.compile(r'(?:\[\s+)defaults(?:\s+]).*')

        for (each_molecule, each_topology, each_top_file) in \
                [[molecule1, topology1, file_topology1],
                 [molecule2, topology2, file_topology2]]:
            if any(filter(defaults_pattern.match, map(str, each_topology.output_sequence))):
                os_util.local_print('{} contains a [ defaults ] directive!'.format(each_top_file),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

            try:
                non_matching_names = [(mol2_name.GetProp('_TriposAtomName'), top_name.atom_name)
                                      for mol2_name, top_name
                                      in zip(each_molecule.GetAtoms(), each_topology.molecules[0].atoms_dict.values())
                                      if mol2_name.GetProp('_TriposAtomName') != top_name.atom_name]
            except KeyError as error:
                if error.args[0] == '_TriposAtomName':
                    os_util.local_print('Failed to read atom names from {}. I will not check topology atom names '
                                        'against the structure atom names. '.format(each_top_file),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                break

            if len(non_matching_names):
                os_util.local_print('Not matching atom names between files {} and {}. Atom names from {} will be '
                                    'used!\n' 
                                    ''.format(each_top_file, each_molecule.GetProp('_Name'), each_top_file)
                                    + '=' * 50
                                    + '\nThese are the non-matching names:\n{:<25}{:<25}'
                                      ''.format('{}'.format(each_molecule.GetProp('_Name')),
                                                '{}'.format(each_top_file)),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                for mol2_name, top_name in non_matching_names:
                    os_util.local_print('{:<25}{:<25}'.format(mol2_name, top_name),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                os_util.local_print('=' * 50, msg_verbosity=os_util.verbosity_level.warning,
                                    current_verbosity=verbosity)

            if each_topology.num_molecules > 1:
                os_util.local_print('Topology file {} contains {} molecules, but I can only understand one '
                                    'moleculetype per file.'
                                    ''.format(each_molecule.GetProp('_Name'), each_topology.num_molecules),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise ValueError('only one molecule accepted, but {} found'.format(each_topology.num_molecules))
            if each_topology.molecules[0].num_atoms != each_molecule.GetNumAtoms():
                os_util.local_print('Topology file {} contains {} atoms, but molecule file {} contain {} atoms'
                                    ''.format(each_top_file, each_topology.molecules[0].num_atoms,
                                              each_molecule.GetProp('_Name'), each_molecule.GetNumAtoms()),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise ValueError('number of atoms mismatch')

    moleculetype_a = topology1.molecules[0]
    moleculetype_b = topology2.molecules[0]

    if not mcs:
        # completeRingsOnly and matchValences are required to prepare a dual topology
        commom_core_smiles = find_mcs([molecule1, molecule2], matchValences=True, ringMatchesRingOnly=True,
                                      completeRingsOnly=True, verbosity=verbosity, savestate=savestate).smartsString
        os_util.local_print('MCS between molecules {} and {} was obtained by find_mcs and is {}'
                            ''.format(molecule1.GetProp('_Name'), molecule2.GetProp('_Name'), commom_core_smiles),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
    else:
        # User supplied a MCS, do not save it
        commom_core_smiles = mcs
        if savestate:
            os_util.local_print('You supplied both a MCS and a save_state object. Input MCS will be not saved, as it '
                                'was not computed here', msg_verbosity=os_util.verbosity_level.warning,
                                current_verbosity=verbosity)
        savestate = None

    # Prepare a pseudomolecule representing the common core between molecules 1 and 2
    core_structure = rdkit.Chem.MolFromSmarts(commom_core_smiles)
    if core_structure is None:
        os_util.local_print('Could not detect/convert common core between topologies\n\tError when running '
                            'rdkit.Chem.MolFromSmarts(commom_core_smiles)\n\tcommom_core_smiles = {}'
                            ''.format(commom_core_smiles),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    # Reconstruct a molecule based on common core and coordinates from molecule1
    temp_core_structure = mol_util.loose_replace_side_chains(molecule1, core_structure, use_chirality=True)
    if temp_core_structure is None:
        os_util.local_print('Could not process the core structure to embed while working with the molecule '
                            '{} (SMILES={}). mol_util.loose_replace_side_chains(target, core_mol) failed. '
                            'core_query: {} (SMARTS={})'
                            ''.format(molecule1, rdkit.Chem.MolToSmiles(molecule1), core_structure,
                                      rdkit.Chem.MolToSmarts(core_structure)),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    else:
        core_structure = temp_core_structure

    # Remove * atoms from common core. Note: if there are more than 10 substitution points, this will fail.
    core_structure = DeleteSubstructs(core_structure, rdkit.Chem.MolFromSmarts('[1,2,3,4,5,6,7,8,9#0]'))
    if core_structure == '':
        os_util.local_print('Could not detect/convert common core between topologies',
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        os_util.local_print('Error when running core_structure = DeleteSubstructs(core_structure, '
                            'rdkit.Chem.MolFromSmiles("*"))',
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        os_util.local_print('This is the Smiles representation of core_structure: {}'
                            ''.format(rdkit.Chem.MolToSmiles(core_structure)),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        raise SystemExit(1)
    os_util.local_print('This is the common core structure: {}'.format(core_structure),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    # Get the list of matching atoms from common_atoms -> molecule1 and common_atoms -> molecule2, then map to
    # into molecule1 -> molecule2
    core_structure = mol_util.adjust_query_properties(core_structure, verbosity=verbosity)
    common_atoms = list(zip(molecule1.GetSubstructMatch(core_structure), molecule2.GetSubstructMatch(core_structure)))

    if len(common_atoms) < 3:
        os_util.local_print('Less than 3 common atoms were found between molecule {} (SMILES={}) and {} (SMILES={})'
                            ''.format(molecule1.GetProp("_Name"), molecule1, molecule2.GetProp("_Name"),
                                      ', '.join(map(str, common_atoms))),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    os_util.local_print('These are the common atoms between {} and {}: {}'
                        ''.format(molecule1.GetProp("_Name"), molecule2.GetProp("_Name"),
                                  ', '.join(map(str, common_atoms))),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Atoms present only in state B. Id + 1 to match Gromacs atom numbering
    only_in_b = [each_atom.GetIdx() + 1 for each_atom in molecule2.GetAtoms()
                 if each_atom.GetIdx() not in map(lambda x: x[1], common_atoms)]

    # Atoms present only in state A. Id + 1 to match Gromacs atom numbering
    only_in_a = [each_atom.GetIdx() + 1 for each_atom in molecule1.GetAtoms()
                 if each_atom.GetIdx() not in map(lambda x: x[0], common_atoms)]

    os_util.local_print('These are the atoms present only in {}: {}'
                        ''.format(molecule1.GetProp("_Name"), ', '.join(map(str, only_in_a))),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
    os_util.local_print('These are the atoms present only in {}: {}'
                        ''.format(molecule2.GetProp("_Name"), ', '.join(map(str, only_in_b))),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Use topology A as base to prepare a dual topology
    dual_topology = all_classes.DualTopologyData(file_topology1)

    # Rename atoms to identify common core and topology A
    for each_atom in dual_topology.molecules[0].atoms_dict.values():
        if each_atom.atom_index in [a + 1 for a, b in common_atoms]:
            # Atom present in A and B, I will only perturb the charges
            newname = '{}_const_idx{}'.format(each_atom.atom_name, each_atom.atom_index)
            b_idx = [b + 1 for a, b in common_atoms if a + 1 == each_atom.atom_index][0]
            atom_b_charge = topology2.molecules[0].atoms_dict[b_idx].q_e

            dual_topology.add_dual_atom_add_atomtype(newname, each_atom,
                                                     topology1.atomtype_dict[each_atom.atom_type], mol_region='const',
                                                     q_a=each_atom.q_e, q_b=atom_b_charge, vdw_v_a=None, vdw_w_a=None,
                                                     vdw_v_b=None, vdw_w_b=None, verbosity=verbosity)
        else:
            # Atom only in A, charge and VdW at B = 0.0 (dummy atom)
            newname = '{}_topA_idx{}'.format(each_atom.atom_name, each_atom.atom_index)
            dual_topology.add_dual_atom_add_atomtype(newname, each_atom, topology1.atomtype_dict[each_atom.atom_type],
                                                     mol_region='A', q_a=None, vdw_v_a=None, vdw_w_a=None, q_b=0.0,
                                                     vdw_v_b=0.0, vdw_w_b=0.0, verbosity=verbosity)

    os_util.local_print('These are the atoms present in topology B only:\n{}'
                        ''.format('\n'.join(['\t{}: {}'.format(each_atom, data.atom_type)
                                             for each_atom, data in topology1.molecules[0].atoms_dict.items()
                                             if each_atom in only_in_b])),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Prepares a dict connecting the old atom index from Topology B to new atom to be added to dual topology
    new_b_indices = dict(zip(only_in_b, range(moleculetype_a.num_atoms + 1,
                                              moleculetype_a.num_atoms + len(only_in_b) + 1)))
    atom_translation_dict = dict([(b + 1, a + 1) for a, b in common_atoms])
    atom_translation_dict.update(new_b_indices)

    # Selects terms from B which contains atoms in the B region of dual topology (by searching in only in B)
    new_molecule = dual_topology.molecules[0]
    for each_atom, new_index in new_b_indices.items():
        # Creates a new atom from each_atom in topology B and updates its name
        new_atom = deepcopy(moleculetype_b.atoms_dict[each_atom])
        newname = '{}_topB_idx{}'.format(new_atom.atom_name, new_index)
        new_atom.atom_index = new_index
        new_atom.charge_group_number = new_index
        dual_topology.add_dual_atom_add_atomtype(newname, new_atom, topology2.atomtype_dict[new_atom.atom_type],
                                                 mol_region='B', q_a=0.0, vdw_v_a=0.0, vdw_w_a=0.0, q_b=None,
                                                 vdw_v_b=None, vdw_w_b=None, verbosity=verbosity)

        # Also add atom to atoms_dict and output sequence
        last_atom = new_molecule.output_sequence.index(new_molecule.atoms_dict[new_index - 1])
        new_molecule.output_sequence.insert(last_atom + 1, new_atom)
        new_molecule.atoms_dict[new_index] = new_atom

    for key, each_atomtype in dual_topology.atomtype_dict.items():
        if each_atomtype.name not in [each_atom.atom_type for each_atom in new_molecule.atoms_dict.values()]:
            suppress_line = '; {} Suppressed\n'.format(new_molecule._format_inline(each_atomtype))
            dual_topology.output_sequence[dual_topology.output_sequence.index(each_atomtype)] = suppress_line
            dual_topology.atomtype_dict[key] = suppress_line

    # Iterates over possible terms
    for term_group_name in ['bonds_dict', 'pairs_dict', 'pairsnb_dict', 'exclusions_dict', 'angles_dict', 'dihe_dict',
                            'constraints_dict', 'vsites2_dict', 'vsites3_dict', 'vsites4_dict']:
        term_group = getattr(moleculetype_b, term_group_name)
        new_term_group = getattr(new_molecule, term_group_name)

        # Iterates over terms of this type containing the atom being modified
        for each_bonded_term in term_group.search_all_with_index(only_in_b):

            new_bonded_term = deepcopy(each_bonded_term)
            new_bonded_term.comments = 'Added by dual top; {}'.format(new_bonded_term.comments)

            # Updates the atoms indices with indices for dual topology
            if term_group.n_fields is not None:
                [setattr(new_bonded_term, field_name, atom_translation_dict[this_each_atom])
                 for this_each_atom, field_name in zip(new_bonded_term[:term_group.n_fields], new_bonded_term._fields)]
            else:
                if term_group_name != 'exclusions_dict':
                    # FIXME: support N-body virtual sites here (also need to add virtual_sitesn parser to topology
                    #  reader
                    os_util.local_print('Cannot parse term {} in molecule {}. N-body virtual site is not currently'
                                        ' supported by dual-topology code.'
                                        ''.format(term_group_name, new_molecule.GetProp('_Name')),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(1)

                for this_each_atom, field_name in zip(new_bonded_term[:-1], new_bonded_term._fields):
                    setattr(new_bonded_term, field_name, atom_translation_dict[this_each_atom])

            # Then adds the term to the dual topology object at the end of the directive
            last_position = new_molecule.output_sequence.index(new_term_group[-1])
            new_molecule.output_sequence.insert(last_position + 1, new_bonded_term)
            new_term_group.append(new_bonded_term)

            os_util.local_print('New bonded term: {}'.format(new_bonded_term),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Add exclusions between topologies A and B
    if len(new_molecule.exclusions_dict) == 0:
        # There are no exclusions in new_molecule.output_sequence, add a [exclusions] directive to
        # new_molecule.output_sequence and set last_directive_index to it
        last_atom_element = new_molecule.atoms_dict[next(reversed(new_molecule.atoms_dict))]
        last_directive_index = new_molecule.output_sequence.index(last_atom_element) + 1
        new_molecule.output_sequence.insert(last_directive_index, '[ exclusions ]     ; Added by dual_topology')
    else:
        # Set last_directive_index to the end of exclusions_dict (ie: last exclusion line in topology)
        last_directive_index = new_molecule.output_sequence.index(new_molecule.exclusions_dict[-1])

    # Add an exclusion between each atom in A part and all atoms in B part
    for idx, each_atom_a in enumerate(only_in_a):
        fields_list = ['atom_ref'] + ['atom_{}'.format(i) for i in range(len(new_b_indices))] + ['comments']
        this_exclusion_data = all_classes.namedlist('ExclusionsData', fields_list)
        exclusion_atoms_list = [each_atom_a] + list(new_b_indices.values()) + ['Added by dual topology']
        this_exclusion = this_exclusion_data(*exclusion_atoms_list)
        new_molecule.exclusions_dict.append(this_exclusion)
        new_molecule.output_sequence.insert(last_directive_index + idx + 1, this_exclusion)
        os_util.local_print('Added exclusion between atom {} and atoms {}'
                            ''.format(this_exclusion.atom_ref, list(new_b_indices.values())),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    os_util.local_print('These are the dual atoms:\n\t{}\nThese are the dual atomtypes:\n\t{}'
                        ''.format('\n\t'.join(map(str, dual_topology.dualatom_data_dict.values())),
                                  '\n\t'.join(map(str, dual_topology.dualatomtype_data_dict.values()))),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Construct dual topology molecule, first iterate over atoms only in B and add them to dual topology molecule
    dual_molecule = rdkit.Chem.RWMol(molecule1)
    dual_molecule.SetProp("_Name", "{}~{}".format(moleculetype_a.name, moleculetype_b.name))
    new_atom_list = [[each_atom.GetIdx(), dual_molecule.AddAtom(rdkit.Chem.Atom(each_atom.GetAtomicNum()))]
                     for each_atom in molecule2.GetAtoms() if each_atom.GetIdx() + 1 in only_in_b]

    # Then iterate over bonds in topology B containing atoms from topB only region, then translate the indexes and add
    # the bond to dual_molecule
    for each_bond in molecule2.GetBonds():
        if not {each_bond.GetBeginAtomIdx() + 1, each_bond.GetEndAtomIdx() + 1}.isdisjoint(set(only_in_b)):
            being_atom = getattr(new_atom_list, str(each_bond.GetBeginAtomIdx() + 1),
                                 atom_translation_dict[each_bond.GetBeginAtomIdx() + 1])
            end_atom = getattr(new_atom_list, str(each_bond.GetEndAtomIdx() + 1),
                               atom_translation_dict[each_bond.GetEndAtomIdx() + 1])

            # TODO: verify stereochemistry (can it be a problem anyway?)
            # Idx-1 to convert back from gromacs numbering to rdkit numbering
            dual_molecule.AddBond(being_atom - 1, end_atom - 1, each_bond.GetBondType())

            os_util.local_print('Added bond between atoms {} and {}'.format(being_atom - 1, end_atom - 1),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    dual_molecule = dual_molecule.GetMol()

    # If we got this far, save mcs
    if savestate:
        savestate.save_data()

    # Embed a copy of molecule B to the common core
    molecule2_embed = rdkit.Chem.Mol(molecule2)

    # Adjust query
    core_structure = mol_util.adjust_query_properties(core_structure, verbosity=verbosity)

    if rdkit.Chem.SanitizeMol(molecule2_embed, catchErrors=True) != 0:
        os_util.local_print('Failed to process molecule {}. Error during sanitization, prior to virtual site '
                            'processing. Cannot continue, please check the input molecules.'
                            ''.format(molecule2_embed))
        raise SystemExit(1)

    try:
        ConstrainedEmbed(molecule2_embed, core_structure, enforceChirality=True)
    except ValueError as error:
        if error.__str__() == "molecule doesn't match the core":
            os_util.local_print('Embeding of molecule {} (SMILES = {}) to common core {} with molecule {} '
                                '(SMILES = {}) has failed. Molecule does not match the core.'
                                ''.format(molecule2.GetProp("_Name"), rdkit.Chem.MolToSmiles(molecule2),
                                          rdkit.Chem.MolToSmiles(core_structure),
                                          molecule1.GetProp("_Name"), rdkit.Chem.MolToSmiles(molecule1)),
                                msg_verbosity=os_util.verbosity_level.error,
                                current_verbosity=verbosity)
            raise SystemExit(1)

        os_util.local_print('First attempt to embed {} to common core {} with molecule {} has failed. Retrying with '
                            'ignoreSmoothingFailures=True, randomseed=randint(9999999), maxAttempts=50.'
                            ''.format(molecule2.GetProp("_Name"), rdkit.Chem.MolToSmiles(core_structure),
                                      molecule1.GetProp("_Name")),
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

        from numpy.random import randint
        constrained_embed_forcefield(molecule2_embed, core_structure, ignoreSmoothingFailures=True,
                                     randomseed=randint(9999999), useRandomCoords=True, maxAttempts=50,
                                     enforceChirality=True, **kwargs)
    finally:
        # If the molecule has a virtual site, UFF will lack a term for it
        if not rdkit.Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule2_embed) \
                and any((moleculetype_b.vsites2_dict, moleculetype_b.vsites3_dict, moleculetype_b.vsites4_dict)):
            os_util.local_print('UFF does not implement a virtual site term. Using distance constraints to position '
                                'the dummy atoms. Please, check your final geometry.',
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

            dist_spread = 0.1

            uff_ff_mol = rdkit.Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(molecule2_embed)

            for each_term in moleculetype_b.vsites2_dict:
                distance = molecule2_embed.GetAtomPosition(each_term.atom_i).Distance(
                    molecule2_embed.GetAtomPosition(each_term.atom_j))
                max_dist = (1.0 + dist_spread) * distance
                min_dist = (1.0 - dist_spread) * distance
                uff_ff_mol.UFFAddDistanceConstraint(each_term.atom_i, each_term.atom_j, relative=True, minLen=min_dist,
                                                    maxLen=max_dist, forceConstant=100)
                os_util.local_print('Added constraint between atoms {} and {} to a distance of between {} and {}'
                                    ''.format(each_term.atom_i, each_term.atom_j, min_dist, max_dist),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

            [uff_ff_mol.UFFAddPositionConstraint(atom_i.GetIdx(), maxDispl=dist_spread, forceConstant=100)
             for atom_i in molecule2_embed.GetAtoms() if atom_i.GetAtomicNum() != 0]

            uff_ff_mol.Initialize()
            uff_ff_mol.Minimize()

            core_structure = mol_util.adjust_query_properties(core_structure, verbosity=verbosity)
            atom_map = [(j, i) for i, j in enumerate(molecule2_embed.GetSubstructMatch(core_structure))]
            try:
                AlignMol(molecule2_embed, core_structure, atomMap=atom_map)
            except ValueError:
                os_util.local_print('Embending of molecule {} (SMILES = {}) to common core {} with molecule {} '
                                    '(SMILES = {}) has failed. Molecule does not match the core.'
                                    ''.format(molecule2.GetProp("_Name"), rdkit.Chem.MolToSmiles(molecule2),
                                              rdkit.Chem.MolToSmiles(core_structure),
                                              molecule1.GetProp("_Name"), rdkit.Chem.MolToSmiles(molecule1)),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)

    # mergedtopologies_class ('MergedTopologies', ['dual_topology', 'dual_molecule', 'mcs', 'common_core_mol',
    #                         'molecule_a', 'topology_a', 'molecule_b', 'topology_b', 'dual_molecule_name'])
    return all_classes.mergedtopologies_class(dual_topology, dual_molecule, commom_core_smiles, core_structure,
                                              molecule1, topology1, molecule2_embed, topology2,
                                              '{}_{}'.format(molecule1.GetProp("_Name"), molecule2.GetProp("_Name")))


## HERE - implement MCS selection
def find_mcs(mol_list, savestate=None, verbosity=0, **kwargs):
    """ Find the MCS between molecules in mol_list

    :param list mol_list: list of rdkit.Chem.Mol molecules
    :param [savestate_util.SavableState, Nonetype] savestate: savestate data, None: do not load or save anything
    :param int verbosity: verbosity level
    :param kwargs: kwargs to be passed to FindMCS
    :rtype: all_classes.MCSResult
    """

    os_util.local_print('Entering find_mcs: mol_list={} (SMILES: {}), verbosity={}, kwargs={})'
                        ''.format(mol_list, [rdkit.Chem.MolToSmiles(each_mol) for each_mol in mol_list],
                                  verbosity, kwargs),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    default_values = {'verbose': verbosity > 2, 'completeRingsOnly': True, 'ringMatchesRingOnly': True,
                      'matchChiralTag': False, 'threshold': 1.0, 'timeout': 3600, 'seedSmarts': '', 'uniquify': True,
                      'useChirality': False, 'useQueryQueryMatches': False, 'maxMatches': 1000, 'plot': False}
    [kwargs.setdefault(key, value) for key, value in default_values.items()]
    # FIXME: work around for bug #2801 in rdkit, verbosity must be set to False, otherwise we get a segfault.
    #  Revert this to `'verbosity': verbosity > 2` when developers fix the bug upstream
    kwargs['verbose'] = False

    if savestate:
        mols_frozenset = frozenset([rdkit.Chem.MolToSmiles(each_mol) for each_mol in mol_list])
        if 'mcs_dict' not in savestate:
            savestate['mcs_dict'] = {}
        elif mols_frozenset in savestate['mcs_dict']:
            os_util.local_print('MCS between molecules {} is loaded from save_state {}. MCS is {}'
                                ''.format(list(mols_frozenset), savestate, savestate['mcs_dict'][mols_frozenset]),
                                os_util.verbosity_level.debug, current_verbosity=verbosity)
            return savestate['mcs_dict'][mols_frozenset]

    # First, find MCS of atoms without hydrogens. This speeds up execution
    altered_mol_list = [rdkit.Chem.RemoveHs(rdkit.Chem.Mol(each_mol)) for each_mol in mol_list]

    # hash atoms using hybridization, atomic number and ring/not in ring
    hash_function = lambda atom: 1000 * rdkit.Chem.QueryAtom.IsInRing(atom) \
                                 + 100 * int(atom.GetHybridization()) \
                                 + atom.GetAtomicNum()
    [each_atom.SetIsotope(hash_function(each_atom)) for each_mol in altered_mol_list for each_atom in
     each_mol.GetAtoms()]

    # Compare isotopes to use hashed information
    os_util.local_print('Running find_mcs for: ["{}"]'.format('", "'.join([rdkit.Chem.MolToSmiles(each_mol)
                                                                           for each_mol in altered_mol_list])),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
    mcs_result = rdkit.Chem.rdFMCS.FindMCS(altered_mol_list, completeRingsOnly=kwargs['completeRingsOnly'],
                                           ringMatchesRingOnly=kwargs['ringMatchesRingOnly'],
                                           matchChiralTag=kwargs['matchChiralTag'],
                                           atomCompare=rdkit.Chem.rdFMCS.AtomCompare.CompareIsotopes,
                                           threshold=kwargs['threshold'], timeout=kwargs['timeout'],
                                           seedSmarts=kwargs['seedSmarts'], verbose=kwargs['verbose'])
    if mcs_result.canceled or mcs_result.numAtoms == 0:
        os_util.local_print('Failed to calculate MCS between molecules {}. Retry with a longer timeout.'
                            ''.format([rdkit.Chem.MolToSmiles(each_mol) for each_mol in mol_list]),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    # FIXME: horrible work around for bug #2801 in rdkit. Replace offending bonds with & after nothing. Revert this
    #  after bug is fixed
    altered_smarts = re.sub(r'(]|\)|^)&!@(\[|\(|$)', r'\1~&!@\2', mcs_result.smartsString)

    os_util.local_print('This is the first MCS (raw, atoms hashed): {}'.format(mcs_result.smartsString),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
    tempmol = rdkit.Chem.MolFromSmarts(altered_smarts)
    if tempmol is None:
        os_util.local_print('Internal error: FindMCS returned an invalid SMARTS (raw SMARTS): {}'
                            ''.format(mcs_result.smartsString),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    [atom.SetAtomicNum(atom.GetIsotope() % 100) for atom in tempmol.GetAtoms()]
    [atom.SetIsotope(0) for atom in tempmol.GetAtoms()]
    os_util.local_print('This is first MCS:\n\tSMARTS: {}\n\tTranslated SMILES: {}'
                        ''.format(altered_smarts, rdkit.Chem.MolToSmiles(tempmol)),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    core_mol = rdkit.Chem.MolFromSmarts(altered_smarts)
    matches = get_substruct_matches_fallback(altered_mol_list[0], core_mol, verbosity=verbosity, kwargs=kwargs)

    if len(matches) > 1:
        # FIXME: allow user to supply an atom map so this case is covered
        os_util.local_print('There is more than one possible match between the first MCS and the first molecule.',
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        match = matches[0]
    else:
        match = matches[0]

    template_mol = rdkit.Chem.RemoveHs(rdkit.Chem.Mol(mol_list[0]))
    common_struct_smiles = rdkit.Chem.MolFragmentToSmiles(template_mol, atomsToUse=match, isomericSmiles=True,
                                                          canonical=False)

    # Construct a mol from SMILES
    common_struct_mol = rdkit.Chem.MolFromSmiles(common_struct_smiles, sanitize=False)
    for each_atom in common_struct_mol.GetAtoms():
        each_atom.SetFormalCharge(0)
        each_atom.SetNumRadicalElectrons(0)
        each_atom.SetNoImplicit(False)
    common_struct_mol.UpdatePropertyCache()

    os_util.local_print('This is the translated SMILES representation of the first MCS: {}'
                        ''.format(rdkit.Chem.MolToSmiles(common_struct_mol)),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    try:
        rdkit.Chem.SanitizeMol(common_struct_mol)
    except ValueError:
        # FIXME: test for a newer version of rdkit which exposes the sanitizaion error
        # Verify if error was caused by atoms from rings that were included in MCS, but are not in a complete ring
        while True:
            try:
                rdkit.Chem.SanitizeMol(common_struct_mol)
            except ValueError as error:
                match_data = re.search(r'non-ring atom (\d+) marked aromatic', error.args[0])
                if match_data is not None:
                    os_util.local_print('Trying to repair MCS'
                                        ''.format(rdkit.Chem.MolToSmiles(common_struct_mol)),
                                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

                    # Suppress the offending atom from molecule
                    problematic_atom = common_struct_mol.GetAtomWithIdx(int(match_data.group(1)))
                    common_struct_mol = rdkit.Chem.RWMol(common_struct_mol)
                    problematic_bonds = [each_bond for each_bond in
                                         common_struct_mol.GetBonds()
                                         if problematic_atom.GetIdx() in [each_bond.GetBeginAtom().GetIdx(),
                                                                          each_bond.GetEndAtom().GetIdx()]]
                    if len(problematic_bonds) != 1:
                        os_util.local_print('The aromatic non-ring atom {} has {} bonds. I cannot process this '
                                            'molecule.'.format(problematic_atom.GetIdx(), len(problematic_bonds)),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise SystemExit(1)
                    else:
                        problematic_bond = problematic_bonds[0]
                        common_struct_mol.RemoveBond(problematic_bond.GetBeginAtomIdx(),
                                                     problematic_bond.GetEndAtomIdx())
                        common_struct_mol.RemoveAtom(problematic_atom.GetIdx())
                        common_struct_mol = rdkit.Chem.Mol(common_struct_mol)
                else:
                    # Sanitization error is not due a non-ring aromatic atom, re-raise
                    raise ValueError(error)

            else:
                # No error in rdkit.Chem.SanitizeMol, go on
                break

    os_util.local_print('This is the SMILES representation of common core: {}'
                        ''.format(rdkit.Chem.MolToSmiles(common_struct_mol)),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Check if the user asked for MCS between molecules having no explicit Hs, if so, return first MCS
    if all([each_mol.GetNumAtoms() == each_mol.GetNumHeavyAtoms() for each_mol in mol_list]):
        os_util.local_print('No molecules from mol_list {} bear hydrogen atoms. Returning first MCS as final MCS.'
                            .format([rdkit.Chem.MolToSmiles(each_mol) for each_mol in mol_list]),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

        mcs_result = all_classes.MCSResult(rdkit.Chem.MolToSmarts(common_struct_mol))
        if savestate:
            mols_frozenset = frozenset([rdkit.Chem.MolToSmiles(each_mol) for each_mol in mol_list])
            savestate['mcs_dict'][mols_frozenset] = mcs_result
            savestate.save_data()

        return mcs_result

    # Use info from first MCS to rehash all atoms
    altered_mol_list = [rdkit.Chem.Mol(each_mol) for each_mol in mol_list]

    match_atoms_dict = {each_mol: {atom.GetIdx(): [] for atom in each_mol.GetAtoms()} for each_mol in altered_mol_list}

    for each_mol in altered_mol_list:
        common_struct_mol = mol_util.adjust_query_properties(common_struct_mol, verbosity=verbosity)
        matches_list = each_mol.GetSubstructMatches(common_struct_mol, uniquify=False,
                                                    useChirality=kwargs['useChirality'],
                                                    useQueryQueryMatches=kwargs['useQueryQueryMatches'],
                                                    maxMatches=kwargs['maxMatches'])

        if not matches_list:
            os_util.local_print('Failed to get a substructure match between molecules {} and {}'
                                ''.format(rdkit.Chem.MolToSmiles(each_mol), rdkit.Chem.MolToSmiles(common_struct_mol)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        # Mapping of i: the idx of the atom in common_struct_mol -> j: the idx in eachmol
        [match_atoms_dict[each_mol][j].append(i) for each_match in matches_list for i, j in enumerate(each_match)]

        os_util.local_print('{} and {} -> matches_list: {}'
                            ''.format(rdkit.Chem.MolToSmiles(each_mol),
                                      rdkit.Chem.MolToSmiles(common_struct_mol), matches_list),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Prepares a list of isotopes to be used in the final FindMCS. This groups equivalent atoms together
    isotopes_list = []
    for rdmol, each_mol in match_atoms_dict.items():
        for key, each_atom_list in each_mol.items():
            if not each_atom_list:
                continue
            each_mol[key] = set(each_atom_list)
            current_iso = os_util.inner_search(set(each_atom_list), isotopes_list, die_on_error=False)
            if current_iso is False:
                isotopes_list.append(set(each_atom_list))

    unmachable_atom_isotope = 1000

    for each_mol in altered_mol_list:
        for idx, each_atom in enumerate(each_mol.GetAtoms()):
            this_atom_map = match_atoms_dict[each_mol]
            if each_atom.GetAtomicNum() == 1:
                each_atom.SetIsotope(1)
                os_util.local_print('Atom {} is a hydrogen, isotope = 1'.format(idx),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            elif this_atom_map[each_atom.GetIdx()]:
                # This atom matches an atom from common core, set its isotope
                new_isotope = os_util.inner_search(this_atom_map[each_atom.GetIdx()], isotopes_list) + 100
                each_atom.SetIsotope(new_isotope)
                os_util.local_print('Atom {}{} matches common core, isotope = {}'
                                    ''.format(idx, each_atom.GetSymbol(), new_isotope),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            else:
                # This atom matches no atom from common core
                each_atom.SetIsotope(unmachable_atom_isotope)
                os_util.local_print('Atom {}{} does not match common core, isotope = {}'
                                    ''.format(idx, each_atom.GetSymbol(), unmachable_atom_isotope),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                unmachable_atom_isotope += 1

        os_util.local_print('These are the atoms of molecule {} to build the final MCS: \n{}'
                            ''.format(each_mol, '\n'.join(['\tmatch_idx: {} GetIdx: {} Isotope: {}'
                                                           ''.format(idx, each_atom.GetIdx(), each_atom.GetIsotope())
                                                           for idx, each_atom in enumerate(each_mol.GetAtoms())])),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if kwargs['plot']:
        from rdkit.Chem import Draw
        ms = [rdkit.Chem.Mol(x) for x in mol_list]
        [rdkit.Chem.AllChem.Compute2DCoords(m) for m in ms]
        for draw_mol, mol in zip(ms, altered_mol_list):
            [draw_mol.GetAtomWithIdx(each_atom.GetIdx()).SetProp('molAtomMapNumber', str(each_atom.GetIsotope()))
             for each_atom in mol.GetAtoms()]
        core_draw = rdkit.Chem.Mol(core_mol)
        rdkit.Chem.AllChem.Compute2DCoords(core_draw)
        img = Draw.MolsToGridImage([*ms, core_draw], subImgSize=(300, 300),
                                   legends=[m.GetProp('_Name') for m in ms] + ["MCS"],
                                   useSVG=True)
        with open('mcs_plot_{}_{}.svg'.format('_'.join([os.path.basename(m.GetProp('_Name')) for m in ms]),
                                              time.strftime('%H%M%S_%d%m%Y')), 'w') as fh:
            fh.write(img)

    # Find the actual, final MCS
    mcs_result = rdkit.Chem.rdFMCS.FindMCS(altered_mol_list, completeRingsOnly=kwargs['completeRingsOnly'],
                                           ringMatchesRingOnly=kwargs['ringMatchesRingOnly'],
                                           matchChiralTag=kwargs['matchChiralTag'],
                                           atomCompare=rdkit.Chem.rdFMCS.AtomCompare.CompareIsotopes,
                                           threshold=kwargs['threshold'], timeout=kwargs['timeout'],
                                           seedSmarts=kwargs['seedSmarts'], verbose=kwargs['verbose'])

    if mcs_result.canceled or mcs_result.numAtoms == 0:
        os_util.local_print('Failed to calculate MCS between molecules {}. Retry with a longer timeout.'
                            ''.format([rdkit.Chem.MolToSmiles(each_mol) for each_mol in mol_list]),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        os_util.local_print('These are the hashed SMILES of the molecules: {}'
                            ''.format([rdkit.Chem.MolToSmiles(each_mol) for each_mol in altered_mol_list]),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        raise SystemExit(1)

    os_util.local_print('This is the final MCS (unmodified, atoms hashed): {}'.format(mcs_result.smartsString),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # FIXME: horrible work around for bug #2801 in rdkit. Replace of offending bonds with & after nothing. Revert this
    #  after bug is fixed
    altered_smarts = re.sub(r'(]|\)|^)&!@(\[|\(|$)', r'\1~&!@\2', mcs_result.smartsString)

    core_mol = mol_util.adjust_query_properties(rdkit.Chem.MolFromSmarts(altered_smarts), generic_atoms=False,
                                                ignore_isotope=False)
    match = altered_mol_list[0].GetSubstructMatch(core_mol, useChirality=kwargs['useChirality'],
                                                  useQueryQueryMatches=kwargs['useQueryQueryMatches'])

    common_struct_smiles = rdkit.Chem.MolFragmentToSmiles(mol_list[0], atomsToUse=match, isomericSmiles=True,
                                                          canonical=True)

    os_util.local_print('This is the final MCS Smiles: {}'.format(common_struct_smiles),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    mcs_result = all_classes.MCSResult(common_struct_smiles, num_atoms=mcs_result.numAtoms,
                                       num_bonds=mcs_result.numBonds, canceled=False)

    if savestate:
        mols_frozenset = frozenset([rdkit.Chem.MolToSmiles(each_mol) for each_mol in mol_list])
        savestate['mcs_dict'][mols_frozenset] = mcs_result
        savestate.save_data()

    return mcs_result


def find_mcs_3d(molecule_a, molecule_b, tolerance=0.3, num_conformers=50, randomseed=2342, num_threads=0,
                savestate=None, verbosity=0, **kwargs):
    """ Find the MCS between molecules molecule_a and molecule_b subject to the constrain that the MCS atoms are close
        in a 3D superimposition of the molecules. This is useful to find MCS between chiral molecules where
        stereocenters varies. Note that this function does not ensure complete rings in the MCS.

    :param rdkit.Chem.Mol molecule_a: first molecule
    :param rdkit.Chem.Mol molecule_b: reference molecule (this molecule's conformation will be use as a reference to the
                                      3D alignment and conformational search.
    :param float tolerance: atoms this close in 3D will be considered as matching
    :param int num_conformers: number of conformers to be generate in each iteration, use a higher value if you molecule
                               is very flexible or if the first chiral MCS is too small.
    :param [savestate_util.SavableState, Nonetype] savestate: savestate data, None: do not load or save anything
    :param int randomseed: seed to rdkit.Chem.EmbedMultipleConfs
    :param int num_threads: use this many threads during conformer generation (0: max supported)
    :param int verbosity: verbosity level
    :rtype: all_classes.MCSResult
    """

    os_util.local_print('Entering find_mcs_3d: molecule_a={} (SMILES={}), molecule_b={} (SMILES={}), tolerance={}, '
                        'num_conformers={}, randomseed={}, num_threads={}, savestate={}, verbosity={}'
                        ''.format(molecule_a, rdkit.Chem.MolToSmiles(molecule_a), molecule_b,
                                  rdkit.Chem.MolToSmiles(molecule_b), tolerance, num_conformers, randomseed,
                                  num_threads, savestate, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if savestate:
        mols_frozenset = frozenset([rdkit.Chem.MolToSmiles(molecule_a), rdkit.Chem.MolToSmiles(molecule_b)])
        if '3dmcs_dict' not in savestate:
            savestate['3dmcs_dict'] = {}
        elif mols_frozenset in savestate['3dmcs_dict']:
            os_util.local_print('3D MCS between molecules {} loaded from save_state {}'
                                ''.format(list(mols_frozenset), savestate), os_util.verbosity_level.debug,
                                current_verbosity=verbosity)
            return savestate['3dmcs_dict'][mols_frozenset]

    # Work with a copy of molecule_a
    molecule_a = rdkit.Chem.Mol(molecule_a)

    # Prepare a MCS ignoring chirality. This will guide the selection of atoms viable to be included in the final MCS
    kwargs.update(matchChiralTag=False)
    achiral_mcs = find_mcs([molecule_a, molecule_b], savestate=savestate, **kwargs)
    os_util.local_print('Achiral MCS: {}'.format(achiral_mcs.smartsString),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    achiral_mcs_mol = rdkit.Chem.MolFromSmarts(achiral_mcs.smartsString)

    achiral_match_a = molecule_a.GetSubstructMatch(achiral_mcs_mol)
    achiral_match_b = molecule_a.GetSubstructMatch(achiral_mcs_mol)

    mcs = find_mcs([molecule_a, molecule_b], savestate=savestate, matchChiralTag=True)
    os_util.local_print('First chiral MCS: {}'.format(mcs.smartsString),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    mcs_history = [rdkit.Chem.MolFromSmarts(mcs.smartsString)]
    while True:
        m1_match = molecule_a.GetSubstructMatch(mcs_history[-1])
        m2_match = molecule_b.GetSubstructMatch(mcs_history[-1])

        match = {a: b for a, b in sorted(zip(m1_match, m2_match))}

        coord_map = {endpoint_atom: molecule_b.GetConformer().GetAtomPosition(target_atom)
                     for target_atom, endpoint_atom in match.items()}

        rdkit.Chem.AllChem.EmbedMultipleConfs(molecule_a, maxAttempts=50, numConfs=num_conformers,
                                              randomSeed=randomseed, useRandomCoords=True, clearConfs=False,
                                              coordMap=coord_map, ignoreSmoothingFailures=True, enforceChirality=True,
                                              numThreads=num_threads)

        mcs_candidate = None
        for each_conf in molecule_a.GetConformers():
            this_core_mol = rdkit.Chem.RWMol(achiral_mcs_mol)
            include_atoms = []
            for core_atom, (idx1, idx2) in enumerate(zip(achiral_match_a, achiral_match_b)):
                delt = each_conf.GetAtomPosition(idx1) - molecule_b.GetConformer().GetAtomPosition(idx2)
                if delt.Length() <= tolerance:
                    include_atoms.append(core_atom)

            if not include_atoms:
                continue

            [this_core_mol.RemoveAtom(atom_idx) for atom_idx in reversed(range(this_core_mol.GetNumAtoms()))
             if atom_idx not in include_atoms]

            core_fragments = GetMolFrags(this_core_mol, asMols=True)
            largest_fragment = max(core_fragments, default=core_fragments, key=lambda m: m.GetNumAtoms())
            if mcs_candidate is None:
                mcs_candidate = largest_fragment
            elif mcs_candidate.GetNumAtoms() < largest_fragment.GetNumAtoms():
                mcs_candidate = largest_fragment

        mcs_history.append(mcs_candidate)
        os_util.local_print('Iteration {} MCS: {} ({} atoms)'
                            ''.format(len(mcs_history), rdkit.Chem.MolToSmiles(mcs_candidate),
                                      mcs_candidate.GetNumAtoms()),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        if len(mcs_history) >= 2 and mcs_history[-1].GetNumAtoms() == mcs_history[-2].GetNumAtoms():
            os_util.local_print('Solution found. MCS atoms per iteration: {}'
                                ''.format(', '.join(['{}: {}'.format(i, m.GetNumAtoms())
                                                     for i, m in enumerate(mcs_history)])),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            break

    rdkit.Chem.MolToPDBFile(molecule_a, 'mola.pdb')
    rdkit.Chem.MolToPDBFile(molecule_b, 'molb.pdb')

    mcs_result = all_classes.MCSResult(rdkit.Chem.MolToSmiles(mcs_history[-1]), num_atoms=mcs_history[-1].GetNumAtoms(),
                                       num_bonds=mcs_history[-1].GetNumBonds(), canceled=False)

    if savestate:
        savestate['3dmcs_dict'][mols_frozenset] = mcs_result
        savestate.save_data()

    return mcs_result


def get_substruct_matches_fallback(reference_mol, core_mol, die_on_error=True, verbosity=0, **kwargs):
    """ Use rdkit GetStructMatched to get matches between reference_mol and core_mol, falling back to progressively
    more loose criteria to match the structures. kwargs will be passed to GetStructMatched

    :param rdkit.Chem.Mol reference_mol: reference molecule
    :param rdkit.Chem.Mol core_mol: query molecule
    :param bool die_on_error: if no match is found, raise
    :param int verbosity: set verbosity level
    :return:
    """

    default_values = {'uniquify': True, 'useChirality': False, 'useQueryQueryMatches': False, 'maxMatches': 1000}
    [kwargs.setdefault(key, value) for key, value in default_values.items()]

    # Prepare a mol representing the MCS and matches it to one of the mols in altered_mol_list (the first one)
    matches = reference_mol.GetSubstructMatches(core_mol, uniquify=kwargs['uniquify'],
                                                useChirality=kwargs['useChirality'],
                                                useQueryQueryMatches=kwargs['useQueryQueryMatches'],
                                                maxMatches=kwargs['maxMatches'])
    if len(matches) == 0:
        os_util.local_print('Could not find a match between molecule {} and MCS SMARTS {}. Retrying with relaxed '
                            'matching logic.'
                            ''.format(rdkit.Chem.MolToSmiles(reference_mol),
                                      rdkit.Chem.MolToSmarts(core_mol)),
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        adj_core_mol = mol_util.adjust_query_properties(core_mol, verbosity=verbosity)
        matches = reference_mol.GetSubstructMatches(adj_core_mol, uniquify=kwargs['uniquify'],
                                                    useChirality=kwargs['useChirality'],
                                                    useQueryQueryMatches=kwargs['useQueryQueryMatches'],
                                                    maxMatches=kwargs['maxMatches'])
        if len(matches) == 0:
            os_util.local_print('Could not find a match between molecule {} and MCS SMARTS {} using relaxed matching '
                                'logic. Retrying with extra-relaxed, generic atoms matching.'
                                ''.format(rdkit.Chem.MolToSmiles(reference_mol),
                                          rdkit.Chem.MolToSmarts(core_mol)),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
            adj_core_mol = mol_util.adjust_query_properties(core_mol, generic_atoms=True, verbosity=verbosity)
            matches = reference_mol.GetSubstructMatches(adj_core_mol, uniquify=kwargs['uniquify'],
                                                        useChirality=kwargs['useChirality'],
                                                        useQueryQueryMatches=kwargs['useQueryQueryMatches'],
                                                        maxMatches=kwargs['maxMatches'])
            if len(matches) == 0:
                if die_on_error:
                    os_util.local_print('Could not find a match between molecule {} and MCS SMARTS {} using '
                                        'extra-relaxed matching. Cannot continue. Check your input files.'
                                        ''.format(rdkit.Chem.MolToSmiles(reference_mol),
                                                  rdkit.Chem.MolToSmarts(core_mol)),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise ValueError("molecule does not match the core")
                else:
                    return False

    return matches
