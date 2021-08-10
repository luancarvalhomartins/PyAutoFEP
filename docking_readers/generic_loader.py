#! /usr/bin/env python3
#
#  generic_loader.py
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

import rdkit
from os.path import splitext
import all_classes
import os_util
import mol_util


def extract_docking_poses(ligands_dict, no_checks=False, verbosity=0):
    """
    :param dict ligands_dict: dict containing docking poses
    :param bool no_checks: ignore checks and tries to go on
    :param int verbosity: be verbosity
    :rtype: dict
    """

    os_util.local_print('Entering extract_docking_poses(poses_data={}, verbosity={})'
                        ''.format(ligands_dict, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    os_util.local_print('{:=^50}\n{:<15} {:<20}'.format(' Poses read ', 'Name', 'File'),
                        msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)

    docking_mol_local = {}
    for each_name, each_mol in ligands_dict.items():

        if isinstance(each_mol, str):
            ligand_format = splitext(each_mol)[1].lower()
            docking_mol_rd = generic_mol_read(ligand_format, each_mol, verbosity=verbosity)
        elif isinstance(each_mol, all_classes.Namespace):
            docking_mol_rd = generic_mol_read(each_mol.format, each_mol.data, verbosity=verbosity)
        elif isinstance(each_mol, dict):
            if isinstance(each_mol['molecule'], rdkit.Chem.Mol):
                docking_mol_rd = each_mol['molecule']
            else:
                ligand_format = each_mol.setdefault('format', os.path.splitext(each_mol['molecule'])[1])
                docking_mol_rd = generic_mol_read(ligand_format, each_mol['molecule'], verbosity=verbosity)
        elif isinstance(each_mol, rdkit.Chem.Mol):
            docking_mol_rd = each_mol
        else:
            os_util.local_print("Could not understand type {} (repr: {}) for your ligand {}"
                                "".format(type(each_mol), repr(each_mol), each_name),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
            raise TypeError('Ligand must be str or all_classes.Namespace')

        if docking_mol_rd is not None:
            os_util.local_print("Read molecule {} from {}"
                                "".format(each_name, each_mol),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.info)
            docking_mol_rd = mol_util.process_dummy_atoms(docking_mol_rd, verbosity=verbosity)

            # docking_mol_local[each_name] = mol_util.rwmol_to_obmol(docking_mol_rd, verbosity=verbosity)
            docking_mol_local[each_name] = docking_mol_rd

            os_util.local_print('{:<15} {:<18}'.format(each_name, str(each_mol)),
                                msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)
            os_util.local_print('Read molecule {} (SMILES: {}) from file {}'
                                ''.format(each_name, rdkit.Chem.MolToSmiles(docking_mol_rd), each_mol),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

        elif no_checks:
            os_util.local_print('Could not read data in {} using rdkit. Falling back to openbabel. It is strongly '
                                'advised you to check your file and convert it to a valid mol2.'
                                ''.format(str(each_mol)),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
            import pybel

            if verbosity < os_util.verbosity_level.debug:
                pybel.ob.obErrorLog.SetOutputLevel(pybel.ob.obError)
            try:
                if type(each_mol) == str:
                    ligand_format = splitext(each_mol)[1].lstrip('.').lower()
                    docking_mol_ob = pybel.readfile(ligand_format, each_mol).__next__()
                elif type(each_mol) == all_classes.Namespace:
                    docking_mol_ob = pybel.readstring(each_mol.format, each_mol.data)
                else:
                    os_util.local_print("Could not understand type {} (repr: {}) for your ligand {}"
                                        "".format(type(each_mol), repr(each_mol), each_name))
                    raise TypeError('Ligand must be str or all_classes.Namespace')
            except (OSError, StopIteration) as error_data:
                os_util.local_print('Could not read your ligand {} from {} using rdkit nor openbabel. Please '
                                    'check/convert your ligand file. Openbabel error was: {}'
                                    ''.format(each_name, str(each_mol), error_data),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                if not no_checks:
                    raise SystemExit(1)
            else:
                # Convert and convert back to apply mol_util.process_dummy_atoms
                docking_mol_rd = mol_util.process_dummy_atoms(mol_util.obmol_to_rwmol(docking_mol_ob))
                docking_mol_local[each_name] = docking_mol_rd

                os_util.local_print('{:<15} {:<18}'
                                    ''.format(each_name,
                                              each_mol['comment'] if isinstance(each_mol, dict)
                                              else each_mol),
                                    msg_verbosity=os_util.verbosity_level.default, current_verbosity=verbosity)
                os_util.local_print('Extracted molecule {} (SMILES: {}) using openbabel fallback from {}.'
                                    ''.format(each_name, rdkit.Chem.MolToSmiles(docking_mol_rd),
                                              str(each_mol)),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        else:
            os_util.local_print('Could not read data in {} using rdkit. Please, check your file and convert it to a '
                                'valid mol2. (You can also use "no_checks" to enable reading using pybel)'
                                ''.format(str(each_mol)),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(-1)

    return docking_mol_local


def read_reference_structure(reference_structure, verbosity=0):
    """ Reads a structure file

    :param str reference_structure: receptor file
    :param int verbosity: be verbosity
    :rtype: pybel.OBMol
    """

    import pybel

    os_util.local_print('Entering extract read_reference_structure(reference_structure={}, verbosity={})'
                        ''.format(reference_structure, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if isinstance(reference_structure, pybel.Molecule):
        return reference_structure

    receptor_format = splitext(reference_structure)[1].lstrip('.')
    if receptor_format == 'pdbqt':
        receptor_format = 'pdb'

    os_util.local_print('Reading receptor data from {} as a {} file'.format(reference_structure, receptor_format),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    try:
        receptor_mol_local = pybel.readfile(receptor_format, reference_structure).__next__()
    except (ValueError, StopIteration, IOError) as error_data:
        os_util.local_print('Could not read file {}. Format {} was guessed from extension). Error message was "{}"'
                            ''.format(reference_structure, receptor_format, error_data),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    else:
        return receptor_mol_local


def generic_mol_read(ligand_format, ligand_data, verbosity=0):
    """ Tries to read a ligand detecting formats and types

    :param str ligand_format: data format or extension
    :param [str, rdkit.Chem.Mol] ligand_data: data to be read
    :param int verbosity: set verbosity
    :rtype: rdkit.Chem.Mol
    """

    if isinstance(ligand_data, rdkit.Chem.Mol):
        return ligand_data

    if ligand_format in ['mol2', '.mol2']:
        docking_mol_rd = rdkit.Chem.MolFromMol2Block(ligand_data, removeHs=False)
        if docking_mol_rd is None:
            docking_mol_rd = rdkit.Chem.MolFromMol2File(ligand_data, removeHs=False)
    elif ligand_format in ['mol', '.mol']:
        docking_mol_rd = rdkit.Chem.MolFromMolBlock(ligand_data, removeHs=False)
        if docking_mol_rd is None:
            docking_mol_rd = rdkit.Chem.MolFromMolFile(ligand_data, removeHs=False)
    elif ligand_format in ['pdbqt', '.pdbqt', 'pdb', '.pdb']:
        os_util.local_print('You are reading a pdb or pdbqt file ({}), which requires openbabel. Should this fail, you '
                            'may try converting it to a mol2 before hand. This may be unsafe.'.format(ligand_data),
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        import pybel
        try:
            ob_molecule = pybel.readstring('pdb', ligand_data)
        except OSError:
            ob_molecule = pybel.readfile('pdb', ligand_data).__next__()

        docking_mol_rd = mol_util.obmol_to_rwmol(ob_molecule)

    else:
        os_util.local_print('Failed to read pose data from {} with type {}'
                            ''.format(ligand_data, ligand_format),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    return docking_mol_rd
