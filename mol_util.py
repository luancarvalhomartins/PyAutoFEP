#! /usr/bin/env python3
#
#  mol_util.py
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
import shutil

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import numpy
import os_util


def rwmol_to_obmol(rdkit_rwmol, verbosity=0):
    """ Converts a rdkit.RWMol to openbabel.OBMol

    :param rdkit.Chem.rdchem.Mol rdkit_rwmol: the ROMol to be converted
    :param int verbosity: be verbosity
    :rtype: pybel.ob.OBMol
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

    if isinstance(rdkit_rwmol, pybel.ob.OBMol):
        os_util.local_print('Molecule {} (SMILES={}) is already a pybel.ob.OBMol'
                            ''.format(rdkit_rwmol, pybel.Molecule(rdkit_rwmol).write('smi')),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.warning)
        return rdkit_rwmol
    if isinstance(rdkit_rwmol, pybel.Molecule):
        os_util.local_print('Molecule {} (SMILES={}) is already a a pybel.Molecule, converting to pybel.ob.OBMol only'
                            ''.format(rdkit_rwmol, rdkit.Chem.MolToSmiles(rdkit_rwmol)),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.warning)
        return rdkit_rwmol.OBMol

    # Set some lookups
    _bondorders = {rdkit.Chem.BondType.SINGLE: 1,
                   rdkit.Chem.rdchem.BondType.UNSPECIFIED: 1,
                   rdkit.Chem.BondType.DOUBLE: 2,
                   rdkit.Chem.BondType.TRIPLE: 3,
                   rdkit.Chem.BondType.AROMATIC: 5}
    _bondstereo = {rdkit.Chem.rdchem.BondStereo.STEREONONE: 0,
                   rdkit.Chem.rdchem.BondStereo.STEREOE: 1,
                   rdkit.Chem.rdchem.BondStereo.STEREOZ: 2}

    new_obmol = pybel.ob.OBMol()
    new_obmol.BeginModify()

    # Assigning atoms
    for index, each_atom in enumerate(rdkit_rwmol.GetAtoms()):
        new_atom = new_obmol.NewAtom()
        new_atom.SetAtomicNum(each_atom.GetAtomicNum())
        new_atom.SetFormalCharge(each_atom.GetFormalCharge())
        try:
            # Update to OpenBabel 3.X, SetImplicitValence was removed
            new_atom.SetImplicitHCount(each_atom.GetNumImplicitHs())
        except AttributeError:
            # This is OpenBabel 2.4
            new_atom.SetImplicitValence(each_atom.GetImplicitValence())
        if each_atom.GetIsAromatic():
            new_atom.SetAromatic()
        new_atom.SetVector(rdkit_rwmol.GetConformer().GetAtomPosition(index).x,
                           rdkit_rwmol.GetConformer().GetAtomPosition(index).y,
                           rdkit_rwmol.GetConformer().GetAtomPosition(index).z)

    # Assigning bonds
    for each_bond in rdkit_rwmol.GetBonds():
        new_obmol.AddBond(each_bond.GetBeginAtomIdx() + 1, each_bond.GetEndAtomIdx() + 1,
                          _bondorders[each_bond.GetBondType()])
        if each_bond.GetIsAromatic():
            new_obmol.GetBond(each_bond.GetBeginAtomIdx() + 1, each_bond.GetEndAtomIdx() + 1).SetAromatic()

    # Copy molecule data
    for k, v in rdkit_rwmol.GetPropsAsDict().items():
        new_data = pybel.ob.OBPairData()
        try:
            new_data.SetAttribute(k)
            new_data.SetValue(v)
        except TypeError:
            # Key or value maybe a boost-derived type which cannot be used in SetValue/SetAttribute, ignore and move on
            os_util.local_print('Molecule data {}: {} cannot be converted to a RDKit compatible type'.format(k, v),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        else:
            new_obmol.CloneData(new_data)

    # FIXME: assign stereochemistry
    # si = rdkit.Chem.FindPotentialStereo(mol)
    # for element in si:
    #     print(f'  Type: {element.type}, Which: {element.centeredOn}, Specified: {element.specified},
    #     Descriptor: {element.descriptor} ')
    #
    # facade = pybel.ob.OBStereoFacade(m)
    # for atom in pybel.ob.OBMolAtomIter(m):
    #     mid = atom.GetId()
    #     if facade.HasTetrahedralStereo(mid):
    #         tetra = facade.GetTetrahedralStereo(mid)
    #         if tetra.IsSpecified():
    #             num_tetra += 1
    #
    # for bond in pybel.ob.OBMolBondIter(m):
    #     mid = bond.GetId()
    #     if facade.HasCisTransStereo(mid):
    #         cistrans = facade.GetCisTransStereo(mid)
    #         if cistrans.IsSpecified():
    #             num_cistrans += 1

    new_obmol.EndModify()

    os_util.local_print('Converted rdkit molecule SMILES {} to an openbabel molecule SMILES: {}'
                        ''.format(rdkit.Chem.MolToSmiles(rdkit_rwmol), pybel.Molecule(new_obmol).write('smi')),
                        current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.info)

    return new_obmol


def obmol_to_rwmol(openbabel_obmol, verbosity=0):
    """Converts an openbabel.OBMol to rdkit.RWMol

    Parameters
    ----------
    openbabel_obmol : pybel.ob.OBMol
        The OBMol to be converted
    verbosity : int
        Sets verbosity level

    Returns
    -------
    rdkit.Chem.Mol
        Converted molecule
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

    if isinstance(openbabel_obmol, rdkit.Chem.Mol):
        os_util.local_print('Entering obmol_to_rwmol. Molecule {} (Props: {}) is already a rdkit.Chem.Mol object!'
                            ''.format(openbabel_obmol, openbabel_obmol.GetPropsAsDict()),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.warning)
        return openbabel_obmol
    elif isinstance(openbabel_obmol, pybel.Molecule):
        openbabel_obmol = openbabel_obmol.OBMol
    elif not isinstance(openbabel_obmol, pybel.ob.OBMol):
        os_util.local_print('Entering obmol_to_rwmol. Molecule {} is a {}, but pybel.Molecule or pybel.ob.OBMol '
                            'required.'
                            ''.format(openbabel_obmol, type(openbabel_obmol)),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise ValueError('pybel.Molecule or pybel.ob.OBMol expected, got {} instead'.format(type(openbabel_obmol)))

    # Set some lookups
    _bondtypes = {0: rdkit.Chem.BondType.UNSPECIFIED,
                  1: rdkit.Chem.BondType.SINGLE,
                  2: rdkit.Chem.BondType.DOUBLE,
                  3: rdkit.Chem.BondType.TRIPLE,
                  5: rdkit.Chem.BondType.AROMATIC}
    _bondstereo = {0: rdkit.Chem.rdchem.BondStereo.STEREONONE,
                   1: rdkit.Chem.rdchem.BondStereo.STEREOE,
                   2: rdkit.Chem.rdchem.BondStereo.STEREOZ}

    _bondtypes_names = {0: "UNSPECIFIED", 1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 5: "AROMATIC"}

    rdmol = rdkit.Chem.Mol()
    rdedmol = rdkit.Chem.RWMol(rdmol)

    # Use pybel write to trigger residue data evaluation, otherwise we get and StopIteration error
    pybel.Molecule(openbabel_obmol).write('pdb')
    try:
        residue_iter = pybel.ob.OBResidueIter(openbabel_obmol).__next__()
    except StopIteration:
        os_util.local_print('Could not read atom names from molecule "{}" (Smiles: {})'
                            ''.format(openbabel_obmol.GetTitle(), pybel.Molecule(openbabel_obmol).write('smi')),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.warning)
        residue_iter = None

    # Assigning atoms
    dummy_atoms = set()
    for index, each_atom in enumerate(pybel.ob.OBMolAtomIter(openbabel_obmol)):
        if residue_iter is not None and residue_iter.GetAtomID(each_atom)[0:2].upper() in ['LP', 'XX'] \
                and each_atom.GetAtomicMass() == 0:
            dummy_atoms.add(index)
            rdatom = rdkit.Chem.MolFromSmarts('*').GetAtomWithIdx(0)
            os_util.local_print('Atom {} was detected as a lone pair because of its name {} and its mass {}'
                                ''.format(index, residue_iter.GetAtomID(each_atom), each_atom.GetAtomicMass()),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.info)

        elif residue_iter is None and each_atom.GetAtomicMass() == 0:
            dummy_atoms.add(index)
            rdatom = rdkit.Chem.MolFromSmarts('*').GetAtomWithIdx(0)
            os_util.local_print('Atom {} was detected as a lone pair because of its mass {} (Note: it was not possible '
                                'to read atom name)'
                                ''.format(index, residue_iter.GetAtomID(each_atom), each_atom.GetAtomicMass()),
                                current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.info)

        else:
            rdatom = rdkit.Chem.Atom(each_atom.GetAtomicNum())

        new_atom = rdedmol.AddAtom(rdatom)
        rdedmol.GetAtomWithIdx(new_atom).SetFormalCharge(each_atom.GetFormalCharge())
        if residue_iter is not None:
            rdedmol.SetProp('_TriposAtomName', residue_iter.GetAtomID(each_atom))
        if each_atom.IsAromatic():
            rdedmol.GetAtomWithIdx(new_atom).SetIsAromatic(True)

    if dummy_atoms:
        os_util.local_print('These are the dummy atoms detected: dummy_atoms={}'.format(dummy_atoms),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.debug)
    else:
        os_util.local_print('No dummy atoms detected.'.format(dummy_atoms),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.debug)

    # Assigning bonds
    for each_bond in pybel.ob.OBMolBondIter(openbabel_obmol):
        rdedmol.AddBond(each_bond.GetBeginAtomIdx() - 1, each_bond.GetEndAtomIdx() - 1,
                        _bondtypes[each_bond.GetBondOrder()])
        if each_bond.IsAromatic():
            rdedmol.GetBondBetweenAtoms(each_bond.GetBeginAtomIdx() - 1,
                                        each_bond.GetEndAtomIdx() - 1).SetIsAromatic(True)
            rdedmol.GetBondBetweenAtoms(each_bond.GetBeginAtomIdx() - 1,
                                        each_bond.GetEndAtomIdx() - 1).SetBondType(_bondtypes[5])

        # This bond contains a dummy atom, converting bond to a UNSPECIFIED
        if dummy_atoms.intersection({each_bond.GetBeginAtomIdx() - 1, each_bond.GetEndAtomIdx() - 1}):
            rdedmol.GetBondBetweenAtoms(each_bond.GetBeginAtomIdx() - 1,
                                        each_bond.GetEndAtomIdx() - 1).SetBondType(_bondtypes[0])
        os_util.local_print('Bond between atoms {} and {} converted to {} type'
                            ''.format(each_bond.GetBeginAtomIdx() - 1, each_bond.GetEndAtomIdx() - 1,
                                      _bondtypes_names[each_bond.GetBondOrder()]),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.debug)

    # FIXME: assign stereochemistry

    rdmol = rdedmol.GetMol()
    try:
        rdmol.UpdatePropertyCache()
    except ValueError as error:
        os_util.local_print('Failed to convert molecule {} to RWMol. Cannot go on. Please, check your input. Error '
                            'was {}'.format(openbabel_obmol.GetTitle(), error),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise error

    # Copy coordinates, first generate at least one conformer
    rdkit.Chem.AllChem.EmbedMolecule(rdmol, useRandomCoords=True, maxAttempts=1000, enforceChirality=True,
                                     ignoreSmoothingFailures=True)
    if rdmol.GetNumConformers() != 1:
        os_util.local_print('Failed to generate coordinates to molecule',
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise ValueError

    for atom_rdkit, atom_obmol in zip(rdmol.GetAtoms(), pybel.ob.OBMolAtomIter(openbabel_obmol)):
        this_position = rdkit.Geometry.rdGeometry.Point3D()
        this_position.x = atom_obmol.x()
        this_position.y = atom_obmol.y()
        this_position.z = atom_obmol.z()
        rdmol.GetConformer().SetAtomPosition(atom_rdkit.GetIdx(), this_position)

    # Copy data
    [rdmol.SetProp(k, v) for k, v in pybel.MoleculeData(openbabel_obmol).items()]
    rdmol.SetProp('_Name', openbabel_obmol.GetTitle())

    for each_atom in rdmol.GetAtoms():
        if each_atom.GetBonds() != ():
            continue
        import numpy
        dist_list = numpy.argsort(numpy.array(rdkit.Chem.AllChem.Get3DDistanceMatrix(rdmol)[each_atom.GetIdx()]))
        closer_atom = int(dist_list[1])
        rdedmol = rdkit.Chem.RWMol(rdmol)
        rdedmol.AddBond(each_atom.GetIdx(), closer_atom)
        rdmol = rdedmol.GetMol()
        rdkit.Chem.SanitizeMol(rdmol)

        os_util.local_print('Atom id: {} is not explicitly bonded to any atom in molecule, connecting it to the closer '
                            'atom id: {}'.format(each_atom.GetIdx(), closer_atom),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.warning)

    rdkit.Chem.SanitizeMol(rdmol)

    os_util.local_print("obmol_to_rwmol converted molecule {} (name: {}). Pybel SMILES: {} to rdkit SMILES: {}"
                        "".format(openbabel_obmol, openbabel_obmol.GetTitle(),
                                  pybel.Molecule(openbabel_obmol).write('smi'), rdkit.Chem.MolToSmiles(rdedmol)),
                        current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.debug)

    return rdmol


def rdmol_com(input_mol, conformer=-1):
    """ Calculates COM of a rdkit.Chem.Mol

    :param rdkit.Chem.Mol input_mol: molecule
    :param conformer: COM of this conformer (-1: auto)
    :rtype: numpy.array
    """

    # TODO: rewrite this for speed using Get
    return numpy.reshape(numpy.array([each_atom_pos * each_atom.GetMass()
                                      for each_atom in input_mol.GetAtoms()
                                      for each_atom_pos in
                                      input_mol.GetConformer(conformer).GetAtomPosition(each_atom.GetIdx())]),
                         [-1, 3]).mean(0)


def initialise_neutralization_reactions():
    """ Prepare SMARTS patterns to neutralize_molecule. Borrowed from rdkit cookbook
    (http://www.rdkit.org/docs/Cookbook.html#neutralizing-charged-molecules)

    :rtype: list
    """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(rdkit.Chem.MolFromSmarts(x), rdkit.Chem.MolFromSmiles(y, False)) for x, y in patts]


def neutralize_molecule(mol, reactions=None):
    """ Neutralizes a molecule. Borrowed from rdkit cookbook
    (http://www.rdkit.org/docs/Cookbook.html#neutralizing-charged-molecules)

    :param rdkit.Chem.Mol mol: molecule to be neutralized
    :param list reactions: use these reactions instead of default ones
    :rtype: set
    """

    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions = initialise_neutralization_reactions()
        reactions = _reactions
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = rdkit.Chem.AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    return mol, replaced


_reactions = None


def verify_molecule_name(molecule, moldict, new_default_name=None, verbosity=0):
    """ Verify the a molecule name exists and is unique and return a valid name the molecule

    :param [rdkit.Chem.Mol, str] molecule: molecule to be verified
    :param dict moldict: dict of read molecules
    :param str new_default_name: if molecule lacks a name, use this name instead (Default: generate a random name)
    :param int verbosity: controls the verbosity level
    :rtype: str
    """

    if isinstance(molecule, rdkit.Chem.Mol):
        try:
            this_mol_name = molecule.GetProp('_Name')
        except KeyError:
            this_mol_name = None
    else:
        if not molecule:
            this_mol_name = None
        else:
            this_mol_name = molecule

    if this_mol_name is None:
        if new_default_name:
            this_mol_name = new_default_name
        else:
            this_mol_name = '(mol_{})'.format(numpy.random.randint(1, 999999999))
            while this_mol_name in moldict:
                this_mol_name = '(mol_{})'.format(numpy.random.randint(1, 999999999))
            if isinstance(molecule, rdkit.Chem.Mol):
                os_util.local_print('Molecule {} have no name. Molecule name is used to save molecule data '
                                    'and serves as an index. I will generate a random name for it, namely: {}'
                                    ''.format(rdkit.Chem.MolToSmiles(molecule), this_mol_name),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
            else:
                os_util.local_print('A molecule have no name. Molecule name is used to save molecule data '
                                    'and serves as an index. I will generate a random name for it, namely: {}'
                                    ''.format(this_mol_name),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    if this_mol_name in moldict:
        colliding_name = this_mol_name
        this_mol_name = '{}_1'.format(this_mol_name)
        while this_mol_name in moldict:
            this_mol_name = this_mol_name[:-1] + str(int(this_mol_name[-1]) + 1)

        if isinstance(molecule, rdkit.Chem.Mol):
            os_util.local_print('Two molecules (Smiles: {} and {}) have the same name {}. Molecule name is used to '
                                'save molecule data and serves as an index. I will rename molecule {} to {}'
                                ''.format(rdkit.Chem.MolToSmiles(moldict[colliding_name]),
                                          rdkit.Chem.MolToSmiles(molecule), colliding_name,
                                          rdkit.Chem.MolToSmiles(molecule), this_mol_name),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        else:
            os_util.local_print('Two molecules have the same name {}. Molecule name is used to '
                                'save molecule data and serves as an index. I will rename the last molecule {}'
                                ''.format(colliding_name, this_mol_name),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    if isinstance(molecule, rdkit.Chem.Mol):
        molecule.SetProp('_Name', this_mol_name)

    return this_mol_name


def process_dummy_atoms(molecule, verbosity=0):
    """ Sanitizes dummy atoms in a rdkit.Chem.Mol

    :param rdkit.Chem.Mol molecule: molecule to be verified
    :param int verbosity: controls the verbosity level
    :rtype: rdkit.Chem.Mol
    """

    os_util.local_print('Entering process_dummy_atoms(molecule=({}; SMILES={}), verbosity={})'
                        ''.format(molecule, rdkit.Chem.MolToSmiles(molecule), verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Iterates over a copy of molecule ahd convert query atoms to dummy atoms, adding bonds if necessary
    temp_mol = rdkit.Chem.Mol(molecule)
    for atom_idx, each_atom in enumerate(temp_mol.GetAtoms()):
        if isinstance(each_atom, rdkit.Chem.rdchem.QueryAtom):
            newdummy = rdkit.Chem.Atom(0)
            rdedmol = rdkit.Chem.RWMol(molecule)
            rdedmol.ReplaceAtom(atom_idx, newdummy, preserveProps=True)
            molecule = rdedmol.GetMol()

            if each_atom.GetProp('_TriposAtomName')[:2] == 'LP':
                os_util.local_print('Lone pair found. Atom with id {} was assumed a lone pair by its name ({}) and '
                                    'its type ({}). If this is wrong, please change the atom name.'
                                    ''.format(atom_idx, each_atom.GetProp('_TriposAtomName'),
                                              each_atom.GetProp('_TriposAtomType')),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
                if each_atom.GetBonds() == ():
                    if temp_mol.GetNumConformers() == 0:
                        os_util.local_print('Disconnected lone pair atom found in a molecule with no 3D coordinates. '
                                            '3D coordinates are used to guess the LP host, but are absent in molecule '
                                            '{}. I cannot continue.'.format(temp_mol),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise SystemExit(1)
                    rdkit.Chem.rdMolTransforms.CanonicalizeConformer(temp_mol.GetConformer(0))
                    # LP is not bonded to any other atom. Connect it to the closer one
                    import numpy
                    temp_mol.GetConformer(0)
                    dist_list = numpy.argsort(numpy.array(rdkit.Chem.Get3DDistanceMatrix(temp_mol)[atom_idx]))
                    closer_atom = int(dist_list[1])
                    rdedmol = rdkit.Chem.RWMol(molecule)
                    rdedmol.AddBond(atom_idx, closer_atom)
                    molecule = rdedmol.GetMol()
                    rdkit.Chem.SanitizeMol(molecule)

                    os_util.local_print('Lonepair {} (id: {}) is not explicitly bonded to any atom in molecule, '
                                        'connecting it to the closer atom {} (id: {}). Please, check the output'
                                        ''.format(molecule.GetAtomWithIdx(atom_idx).GetProp('_TriposAtomName'),
                                                  atom_idx,
                                                  molecule.GetAtomWithIdx(closer_atom).GetProp('_TriposAtomName'),
                                                  closer_atom),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

            else:
                # FIXME: support other dummy atoms (eg: in linear molecules)
                os_util.local_print('The molecule {} contains dummy atoms which are not lonepairs. This is not '
                                    'supported.'.format(molecule),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)

    return molecule


@os_util.trace
def adjust_query_properties(query_molecule, generic_atoms=False, ignore_charge=True, ignore_isotope=True, verbosity=0):
    """ Adjust query settings removing all charges, isotope, aromaticity and valence info from core_structure SMARTS

    :param rdkit.Chem.Mol query_molecule: query molecule
    :param bool generic_atoms: make atoms generic
    :param bool ignore_charge: set all atomic charges to 0
    :param bool ignore_isotope: ignore atomic isotopes
    :param int verbosity: controls the verbosity level
    :rtype: rdkit.Chem.Mol
    """

    new_query_molecule = rdkit.Chem.Mol(query_molecule)

    # Parameters to GetSubstructMatch
    query_m = rdkit.Chem.rdmolops.AdjustQueryParameters()
    query_m.makeBondsGeneric = True
    query_m.makeDummiesQueries = True
    query_m.adjustDegree = False

    if generic_atoms:
        query_m.makeAtomsGeneric = True
    else:
        if ignore_isotope:
            [a0.SetQuery(rdkit.Chem.MolFromSmarts('[#{}]'.format(a0.GetAtomicNum())).GetAtomWithIdx(0))
             for a0 in new_query_molecule.GetAtoms() if isinstance(a0, rdkit.Chem.QueryAtom)]
        if ignore_charge:
            [a0.SetFormalCharge(0) for a0 in new_query_molecule.GetAtoms()]

    new_query_molecule = rdkit.Chem.AdjustQueryProperties(new_query_molecule, query_m)

    os_util.local_print('The molecule {} (SMARTS={}) was altered by adjust_query_properties to {} (SMARTS={})'
                        ''.format(query_molecule, rdkit.Chem.MolToSmarts(query_molecule),
                                  new_query_molecule, rdkit.Chem.MolToSmarts(new_query_molecule)),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    return new_query_molecule


def num_explict_hydrogens(mol):
    """ Return the number of explicit hydrogens in molecular graph

    :param  rdkit.Chem.Mol mol: the input molecule
    :rtype: int
    """

    return sum([1 for i in mol.GetAtoms() if i.GetAtomicNum() == 1])


def num_implicit_hydrogens(mol):
    """Return the number of implicit hydrogens in the molecular graph

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule

    Returns
    -------
    int
    """

    return sum([each_atom.GetNumImplicitHs() for each_atom in mol.GetAtoms()])


def loose_replace_side_chains(mol, core_query, use_chirality=False, verbosity=True):
    """ Reconstruct a molecule based on common core. First, try to use the regular query. If fails, fallback to
        generalized bonds then generalized atoms.

    :param rdkit.Chem.Mol mol: the molecule to be modified
    :param rdkit.Chem.Mol core_query: the molecule to be used as a substructure query for recognizing the core
    :param bool use_chirality: match the substructure query using chirality
    :param int verbosity: set verbosity level
    :rtype: rdkit.Chem.Mol
    """

    temp_core_structure = rdkit.Chem.Mol(core_query)
    if num_explict_hydrogens(core_query) > 0 and num_explict_hydrogens(mol) == 0:
        os_util.local_print('loose_replace_side_chains was called with a mol without explict hydrogens and a '
                            'core_query with {} explict hydrogens. Removing core_query explict Hs.'
                            ''.format(num_explict_hydrogens(core_query)),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        editable_core = rdkit.Chem.EditableMol(core_query)
        hydrogen_atoms = [each_atom.GetIdx() for each_atom in core_query.GetAtoms() if each_atom.GetAtomicNum() == 1]
        for idx in sorted(hydrogen_atoms, reverse=True):
            editable_core.RemoveAtom(idx)
        temp_core_structure = editable_core.GetMol()
        rdkit.Chem.SanitizeMol(temp_core_structure, catchErrors=True)

    result_core_structure = rdkit.Chem.ReplaceSidechains(mol, temp_core_structure, useChirality=use_chirality)
    if result_core_structure is None:
        os_util.local_print('rdkit.Chem.ReplaceSidechains failed with mol={} (SMILES="{}") and coreQuery={} '
                            '(SMARTS="{}"). Retrying with adjust_query_properties.'
                            ''.format(mol, rdkit.Chem.MolToSmiles(mol), temp_core_structure,
                                      rdkit.Chem.MolToSmarts(temp_core_structure)),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
        temp_core_mol = adjust_query_properties(temp_core_structure, verbosity=verbosity)
        result_core_structure = rdkit.Chem.ReplaceSidechains(mol, temp_core_mol, useChirality=use_chirality)
        if result_core_structure is None:
            os_util.local_print('rdkit.Chem.ReplaceSidechains failed with mol={} (SMILES="{}") and coreQuery={} '
                                '(SMARTS="{}"). Retrying with adjust_query_properties setting generic_atoms=True.'
                                ''.format(mol, rdkit.Chem.MolToSmiles(mol), temp_core_structure,
                                          rdkit.Chem.MolToSmarts(temp_core_structure)),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            temp_core_mol = adjust_query_properties(temp_core_structure, generic_atoms=True, verbosity=verbosity)
            result_core_structure = rdkit.Chem.ReplaceSidechains(mol, temp_core_mol, useChirality=use_chirality)

    return result_core_structure


def has_3d(temp_mol, conf_id=-1, tolerance=1e-5, verbosity=0):
    """Check if molecules has 3D coordinates. Based upon OpenBabel OBMol::Has3D()

    Parameters
    ----------
    temp_mol : rdkit.Chem.Mol
        Molecule to be checked.
    conf_id : int
        Test 3D for this conformation.
    tolerance : float
        Tolerance, in angstroms, for a value to be taken as not null.
    verbosity : int
        Sets the verbosity level.

    Returns
    -------
    bool
        True if molecule has 3D coordinates, False otherwise.
    """

    positions_array = temp_mol.GetConformer(conf_id).GetPositions()
    return not numpy.allclose(positions_array, 0.0, atol=tolerance, rtol=0.0)


def translation_to_4by4_mat(translation, verbosity=0):
    """ Convert a translation to a 4-by-4 matrix format

    Parameters
    ----------
    translation : numpy.array
        Operation to be converted
    verbosity : int
        Sets the verbosity level

    Returns
    -------
    numpy.array
        4-by-4 translation matrix
    """

    if translation.shape == (3,):
        ret_mat = numpy.identity(4)
        ret_mat[:3, 3] = translation[:3]
    elif translation.shape == (4, 4):
        if numpy.identity(4)[: 4, :3] != translation[: 4, :3]:
            os_util.local_print('{} is not a valid translation matrix!',
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

        ret_mat = translation
    else:
        os_util.local_print('Cannot understand translation {}. It must be either a 3-element vector or a 4-by-4 '
                            'matrix'.format(translation), msg_verbosity=os_util.verbosity_level.error,
                            current_verbosity=verbosity)
        raise TypeError

    return ret_mat


def rotation_to_4by4_mat(rotation, verbosity=0):
    """ Convert a rotation to a 4-by-4 matrix format

    Parameters
    ----------
    rotation : numpy.array
        Operation to be converted
    verbosity : int
        Sets the verbosity level

    Returns
    -------
    numpy.array
        4-by-4 rotation matrix
    """

    if rotation.shape == (3, 3):
        ret_mat = numpy.identity(4)
        ret_mat[:3, :3] = rotation
    elif rotation.shape == (4, 4):
        if numpy.allclose(numpy.identity(4)[3, :], rotation[:, 3]) \
                and numpy.allclose(numpy.identity(4)[3, :], rotation[3, :]):
            os_util.local_print('{} is not a valid rotation matrix!',
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        ret_mat = rotation
    else:
        os_util.local_print('Cannot understand rotation {}. It must be either a 3-by-3 or a 4-by-4 matrix'
                            ''.format(rotation),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise TypeError

    return ret_mat


def parameterize_small_molecule(input_molecule, param_type='acpype', executable=None, charge_method=None,
                                atom_type=None, output_dir=None, verbosity=0, **kwargs):
    """ Parameterize small molecules using AcPYPE.

    Parameters
    ----------
    input_molecule : rdkit.Chem.Mol
        Molecule to be parameterized
    param_type : str
        Which parameterization software to use. Currently, only "acpype" is supported
    executable : str
        Use this executable to parameterize
    charge_method : str
        Selects the charge method. Choices are "bcc", "user", and "gas" for AcPYPE
    atom_type : str
        Which atom type to use. Only relevant for AcPYPE. Choices are 'gaff', 'amber', 'gaff2', 'amber2'
    output_dir : str
        Save generated topology files to this directory
    verbosity : int
        Sets the verbosity level

    Returns
    -------
    list
        List containing the generated topology files for this ligand
    """

    if not output_dir:
        output_dir = os.getcwd()
    top_files = []

    timeout = kwargs.get('timeout', None)

    if param_type == 'acpype':
        # Check whether ligand parameters are already there
        new_files = [os.path.join(output_dir, input_molecule.GetProp('_Name') + each_ext)
                     for each_ext in ['.itp', '.top']]
        if all([os.path.isfile(f) for f in new_files]):
            # Parameter files exists in output_dir. Return then
            return new_files

        # TODO: use AcPYPE API instead of subprocess. But it will require acpype to be installed in the same env as
        #  PyAutoFEP, which may or may not be the best option for the user. Maybe using the API, then falling back to
        #  subprocess would be ideal.
        from subprocess import run, CalledProcessError, TimeoutExpired
        from tempfile import TemporaryDirectory

        ligand_file = '{}.mdl'.format(input_molecule.GetProp('_Name'))

        # Assemble AcPYPE command line using:
        # acpype -i _file_ -c _string_ -n _int_ -a _string_
        if not executable:
            executable = 'acpype'

        # Total charge is pre-calculated using rdkit. In case acpype guesses it wrong, topology would be wrong.
        cmd_line = [executable, '-i', ligand_file, '-n', str(rdkit.Chem.GetFormalCharge(input_molecule)), '-o', 'gmx',
                    '-b', input_molecule.GetProp('_Name')]

        # Parse charge method
        if charge_method in ['gas', 'bcc', 'user']:
            cmd_line += ['-c', charge_method]
        elif charge_method:
            # User supplied a charge_method, but acpype won't support it
            os_util.local_print('AcPYPE allowed charge methods are "gas","bcc", and "user". Charge method {} cannot be'
                                'used.'.format(charge_method),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError('Incompatible charge method {}.'.format(charge_method))

        # Parse atom types
        if atom_type in ['gaff', 'amber', 'gaff2', 'amber2']:
            cmd_line += ['-a', atom_type]
        elif atom_type:
            os_util.local_print('AcPYPE allowed atom types are "gaff", "amber", "gaff2", "amber2". Atom typing {} '
                                'cannot be used.'.format(atom_type),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError('Incompatible atom type {}.'.format(atom_type))

        for each_option in ['qprog', 'max_time']:
            if each_option in kwargs:
                cmd_line.extend(['--{}'.format(each_option), str(kwargs[each_option])])

        os_util.local_print("Parameterizing small molecule {} using {} and the following options: executable={}, "
                            "charge_method={}, atom_type={}, output_dir={}, verbosity={}. Command line is: \"{}\""
                            "".format(input_molecule.GetProp('_Name'), param_type, executable, charge_method,
                                      atom_type, output_dir, verbosity, ' '.join(cmd_line)),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

        # Run acpype in a temporary directory
        with TemporaryDirectory() as this_tmp_dir:
            rdkit.Chem.MolToMolFile(input_molecule, os.path.join(this_tmp_dir, ligand_file))
            try:
                acpype_run = run(cmd_line, capture_output=True, text=True, cwd=this_tmp_dir, timeout=timeout,
                                 env=os.environ)
            except TimeoutExpired as error:
                os_util.local_print('AcPYPE run failed due to timeout (timeout={}). The complete run command was {}. '
                                    ''.format(timeout, ' '.join(cmd_line)),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise error
            os_util.local_print('This was the acpype run data: {}'.format(acpype_run),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

            if acpype_run.returncode != 0:
                os_util.local_print('AcPYPE run failed. The complete run command was {}. Return code was {} and '
                                    'output was:\n{}\n{}'
                                    ''.format(' '.join(cmd_line), acpype_run.returncode, acpype_run.stdout,
                                              acpype_run.stderr),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise CalledProcessError(returncode=acpype_run.returncode, cmd=executable, output=acpype_run.stdout,
                                         stderr=acpype_run.stderr)
            # Create the destination directory, if needed
            result_dir = os.path.join(this_tmp_dir, input_molecule.GetProp('_Name') + '.acpype')
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass

            for each_ext in ['.itp', '.top']:
                original_file = os.path.join(result_dir, input_molecule.GetProp('_Name') + '_GMX' + each_ext)
                new_file = os.path.join(output_dir, input_molecule.GetProp('_Name') + each_ext)
                try:
                    shutil.copy2(original_file, new_file)
                except FileNotFoundError as error:
                    os_util.local_print('Ligand parameter file {} not found after running AcPYPE. Parameterization '
                                        'failed. AcPYPE return code was {} and output:\n{}\n{}'
                                        ''.format(original_file, acpype_run.returncode, acpype_run.stdout,
                                                  acpype_run.stderr),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise error

                os_util.local_print('Copying {} to {}'.format(original_file, new_file),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                top_files.append(new_file)

    # After writing the initial code below, I found out that, as of 2022.06.02, HTMD parameterize is no longer publicly
    # available (https://github.com/Acellera/htmd/issues/1029). I am keeping this code commented here in the hope that
    # parameterize maybe be once again made public, and we can support it.
    # elif param_type == 'htmd':
    #     from subprocess import run
    #
    #     ligand_file = '{}.mol2'.format(input_molecule.GetProp('_Name'))
    #     if not executable:
    #         executable = 'parameterize'
    #
    #     # By default, only a single CPU and local queue will be used. That's because this function maybe parallelized,
    #     # so that using several CPU could require more cores than available. Local queue is required because if a job
    #     # is submitted to a queue, process would return and script would go on, regardless of the job itself to be
    #     # finished. Total charge is pre-calculated using rdkit. In case parameterize guesses it wrong, topology would
    #     # be wrong.
    #     cmd_line = [executable, ligand_file, '--charge', str(rdkit.Chem.GetFormalCharge(input_molecule)),
    #                 '--queue', 'local', '--ncpus', '1', '--outdir', output_dir]
    #
    #     # Parse charge method
    #     if charge_method in ['gas', 'bcc', 'esp']:
    #         # Unify the charge method names, so that the user don't (--charge-type option in Parameterize accepts
    #         # Gasteiger, AM1-BCC, ESP, so using gas, bcc and esp, respectively.
    #         if charge_method == 'gas':
    #             charge_method = 'Gasteiger'
    #         if charge_method == 'bcc':
    #             charge_method = 'AM1-BCC'
    #         if charge_method == 'esp':
    #             charge_method = 'ESP'
    #         cmd_line += ['--charge-type', charge_method]
    #     elif charge_method:
    #         # User supplied a charge_method, but parameterize won't support it
    #         os_util.local_print('HTMD parameterize allowed charge methods (--charge-type) are "gas" (Gasteiger), "bcc" '
    #                             '(AM1-BCC), and "esp" (ESP). Charge method {} cannot be used.'.format(charge_method),
    #                             msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
    #         raise ValueError('Incompatible charge method {}.'.format(charge_method))
    #
    #     for each_option in ['dihed-fit-type', 'dihed-num-iterations', 'seed', 'dihed-num-iterations', 'scan-type',
    #                         'no-dihed', 'min-type', 'environment', 'basis', 'theory', 'code', 'forcefield', 'dihedral']:
    #         if each_option in kwargs:
    #             cmd_line.extend(['--{}'.format(each_option), str(kwargs[each_option])])
    #
    #     os_util.local_print("Parameterizing small molecule {} using {} and the following options: executable={}, "
    #                         "charge_method={}, output_dir={}, verbosity={}. Command line is: \"{}\""
    #                         "".format(input_molecule.GetProp('_Name'), param_type, executable, charge_method,
    #                                   output_dir, verbosity, ' '.join(cmd_line)),
    #                         msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
    #
    #     # convert/copy parameters
    #     htmd_run = run(cmd_line, capture_output=True, text=True, env=os.environ)

    else:
        os_util.local_print('Unknown parameterization software/library {}. Please, choose between "acpype" or "htmd".'
                            ''.format(param_type),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise ValueError('Unknown parameterization software/library {}.'.format(param_type))

    return top_files


def read_small_molecule_from_pdbqt(ligand_file, charge_error_tol=0.5, no_checks=False, verbosity=0):
    """ Reads PDBQT for a small ligand (ie, it will fail for macromolecules) using meeko, rdkit and openbabel.

    Parameters
    ----------
    ligand_file : str
        Input PDBQT file
    charge_error_tol : float
        Maximum error allowed between the sum of PDBQT partial charges and total charge of the read molecule. Because
        total charge is an integer, a large value here should be safe.
    no_checks : bool
        Ignore checks and keep going
    verbosity : int
        Sets the verbosity level

    Returns
    -------
    rdkit.Chem.Mol
    """
    # I was unable to correctly read PDBQT it using pybel. See issue openbabel issue:
    # https://github.com/openbabel/openbabel/issues/2470.
    try:
        # Try reading with meeko. It can only interpret pdbqt
        from meeko import PDBQTMolecule
        temp_mol = PDBQTMolecule.from_file(ligand_file, skip_typing=True).export_rdkit_mol()
        return temp_mol
    except (ImportError, RuntimeError, KeyError):
        # Either meeko is not installed or the input was not created with meeko, so that meeko
        # cannot read it. Solution: convert PDBQT to PDB using openbabel, then read it as PDB.
        from all_classes import PDBFile

        # If a SMILES REMARK is present, we can read the PDB using RDKit and avoid openbabel altogether
        mol_text = os_util.read_file_to_buffer(ligand_file, return_as_list=True)
        smiles_line = os_util.inner_search(needle='SMILES', haystack=mol_text,
                                           apply_filter=lambda s: not s.startswith('REMARK'))
        if smiles_line is not False:
            temp_pdb = PDBFile(ligand_file)
            for each_atom in temp_pdb.atoms:
                each_atom.element = ''
            # Prune lines, so that the remaining data is PDB-compatible
            if len(temp_pdb.models) > 1:
                mol_names = {i.split('=')[1].strip() for j in temp_pdb.models for i in j.__str__()
                             if i.startswith('REMARK') and i.find('Name') != -1}
                if len(mol_names) == 0:
                    # No "Name =" was found in the pdbqt REMARK records, which suggests that this is not a
                    # Vina/QVina2 output.
                    if no_checks:
                        os_util.local_print('No "Name = " found in the REMARK records in {}. Because you are running '
                                            'with no_checks, I will assume {} contains a single molecule and use it as '
                                            'is.'.format(ligand_file, ligand_file),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    else:
                        os_util.local_print('No "Name = " found in the REMARK records in {}. Please, check your '
                                            'inputs and try again. Alternatively, rerunning with no_check will turn '
                                            'of this checking.'.format(ligand_file),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise SystemExit(-1)
                elif len(mol_names) > 1:
                    if no_checks:
                        os_util.local_print('More than one molecule ({}) read from PDBQT file {}. Currently, this is '
                                            'not supported. Because your are running with no_checks, I WILL IGNORE ALL '
                                            'MOLECULES BUT THE FIRST and go on.'.format(mol_names, ligand_file),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    else:
                        os_util.local_print('More than one molecule ({}) read from PDBQT file {}. Currently, this is '
                                            'not supported. Please, check your your input molecules and split, if '
                                            'needed. Alternatively, rerunning with no_check will turn of this '
                                            'checking.'.format(mol_names, ligand_file),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise SystemExit(-1)

                temp_pdb_data = temp_pdb.models[0].__str__()
            else:
                temp_pdb_data = temp_pdb.to_file()

            temp_pdb_list = [re.sub('\n+', '\n', i) for i in temp_pdb_data if ('{:<6}'.format(i))[:6] in
                             ['REMARK', 'HETATM', 'ATOM  ', 'MODEL ', 'CONECT', 'COMPND', 'ENDMDL', 'END   ']]
            read_mol = read_small_molecule_from_pdb(''.join(temp_pdb_list), verbosity=verbosity)

            return read_mol

        else:
            # There is no SMILES REMARK, use openbabel to guess bond orders
            try:
                from openbabel import pybel
            except ImportError:
                import pybel

            if verbosity < os_util.verbosity_level.extra_debug:
                pybel.ob.obErrorLog.SetOutputLevel(pybel.ob.obError)
            else:
                os_util.local_print('OpenBabel warning messages are on, expect a lot of output.',
                                    msg_verbosity=os_util.verbosity_level.extra_debug, current_verbosity=verbosity)

            temp_mol = [i for i in pybel.readfile('pdbqt', ligand_file) if i is not None]

            # Read the ligand_name in "Name = ligand_name" from each of the MODEL in the pdbqt. Make
            # sure it all matches (by converting to a set then checking the len).
            mol_names = {i.split('=')[1].strip() for j in temp_mol for i in j.data['REMARK'].split('\n')
                         if i.find('Name') != -1}
            if len(mol_names) == 0:
                # No "Name =" was found in the pdbqt REMARK records, which suggests that this is not a
                # Vina/QVina2 output.
                if no_checks:
                    os_util.local_print('No "Name = " found in the REMARK records in {}. Because you are running with '
                                        'no_checks, I will assume {} contains a single molecule and use it as is.'
                                        ''.format(ligand_file, ligand_file),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                else:
                    os_util.local_print('No "Name = " found in the REMARK records in {}. Please, check your inputs and '
                                        'try again. Alternatively, rerunning with no_check will turn of this checking.'
                                        ''.format(ligand_file),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(-1)

            if len(mol_names) > 1:
                if no_checks:
                    os_util.local_print('More than one molecule ({}) read from PDBQT file {}. Currently, this is not '
                                        'supported. Because your are running with no_checks, I WILL IGNORE ALL '
                                        'MOLECULES BUT THE FIRST and go on.'.format(mol_names, ligand_file),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                else:
                    os_util.local_print('More than one molecule ({}) read from PDBQT file {}. Currently, this is not '
                                        'supported. Please, check your your input molecules and split, if needed. '
                                        'Alternatively, rerunning with no_check will turn of this checking.'
                                        ''.format(mol_names, ligand_file),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(-1)

            this_mol_name = temp_mol[0].title
            this_tot_charge = sum([i.partialcharge for i in temp_mol[0].atoms])
            temp_mol = pybel.readstring('pdb', temp_mol[0].write('pdb', opt={'n': True}))
            # Only non-polar Hs are added because polar Hs should be present in the input PDBQT and
            # adding all hydrogens could change the ligand protonation state.
            temp_mol.OBMol.AddNonPolarHydrogens()

            temp_pdb = temp_mol.write('pdb', opt={'n': True}).splitlines(keepends=True)
            temp_mol = PDBFile(temp_pdb)
            for each_atom in temp_mol.atoms:
                each_atom.element = ''
            temp_mol = rdkit.Chem.MolFromPDBBlock(''.join(temp_mol.to_file()), removeHs=False)

            if abs(this_tot_charge - rdkit.Chem.GetFormalCharge(temp_mol)) > charge_error_tol:
                # The sum of partial charges differs from the formal charge of the parsed molecule, therefore
                #  reading/conversion failed.
                if no_checks:
                    os_util.local_print('Incorrect total charge calculated when reading the input molecule {}. This '
                                        'means that there was an error when converting and parsing from the PDBQT '
                                        'format. Because you are running with no_checks, I will go on. Be aware that '
                                        'this will definitely lead to problems!'.format(ligand_file),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                else:
                    os_util.local_print('Incorrect total charge calculated when reading the input molecule {}. This '
                                        'means that there was an error when converting and parsing from the PDBQT '
                                        'format. Please, use another input format. Alternatively, using no_checks '
                                        'will supress this checking. The molecule total charge is {}, while the sum '
                                        'of PDBQT partial charges is {} (tolerance = {}). The molecule smiles ie {}.'
                                        ''.format(ligand_file, rdkit.Chem.GetFormalCharge(temp_mol), this_tot_charge,
                                                  charge_error_tol, rdkit.Chem.MolToSmiles(temp_mol)),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(-1)
            temp_mol.SetProp('_Name', this_mol_name)
            return temp_mol


def read_small_molecule_from_pdb(ligand_data, verbosity=0):
    """ Reads a molecule from a PDB file or block using RDKit, falling back to OpenBabel

    Parameters
    ----------
    ligand_data : str
        PDB file or PDB block
    verbosity : int
        Sets verbosity level

    Returns
    -------
    rdkit.Chem.Mol
    """

    try:
        read_mol = rdkit.Chem.MolFromPDBFile(ligand_data, removeHs=True)
    except OSError:
        read_mol = rdkit.Chem.MolFromPDBBlock(ligand_data, removeHs=True)
        mol_text = ligand_data.split('\n')
    else:
        mol_text = os_util.read_file_to_buffer(ligand_data, return_as_list=True, verbosity=verbosity)

    if read_mol is not None:
        smiles_line = os_util.inner_search(needle='SMILES', haystack=mol_text,
                                           apply_filter=lambda s: not s.startswith('REMARK'))
        if smiles_line is not False:
            smiles = mol_text[smiles_line].split()[-1]
            if num_explict_hydrogens(read_mol) != 0:
                os_util.local_print('There are explicit hydrogens in the molecule {}. Removing.'
                                    ''.format(ligand_data if len(ligand_data) < 30
                                              else '<{}-chr str>'.format(len(ligand_data))),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                hs_params = rdkit.Chem.RemoveHsParameters()
                hs_params.removeDegreeZero = True
                read_mol = rdkit.Chem.RemoveHs(read_mol, hs_params)
            ref_mol = rdkit.Chem.MolFromSmiles(smiles)
            if ref_mol is None:
                os_util.local_print('Failed to convert SMILES string in molecule {}. SMILES is {}.'
                                    ''.format(ligand_data if len(ligand_data) < 30
                                              else '<{}-chr str>'.format(len(ligand_data)), smiles),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise ValueError

            try:
                read_mol = rdkit.Chem.AllChem.AssignBondOrdersFromTemplate(ref_mol, read_mol)
            except ValueError as error:
                os_util.local_print('Failed to match SMILES {} to molecule {}.'
                                    ''.format(smiles,
                                              ligand_data if len(ligand_data) < 30
                                              else '<{}-chr str>'.format(len(ligand_data))),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise error

            read_mol = rdkit.Chem.AddHs(read_mol, addCoords=True, addResidueInfo=True)
            os_util.local_print('Bond orders in {} set to match the smiles {}.'
                                ''.format(ligand_data if len(ligand_data) < 30
                                          else '<{}-chr str>'.format(len(ligand_data)), smiles),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
            rdkit.Chem.AllChem.AssignStereochemistry(read_mol)
            rdkit.Chem.AllChem.AssignStereochemistryFrom3D(read_mol, replaceExistingTags=False)
        else:
            # There was no SMILES line in the PDB file, cannot use AssignBondOrdersFromTemplate. Fallback to openbabel.
            read_mol = None

    if read_mol is None:
        os_util.local_print('Reading PDB {} using RDKit failed, falling back to openbabel.'
                            ''.format(ligand_data if len(ligand_data) < 30
                                      else '<{}-chr str>'.format(len(ligand_data))),
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        try:
            from openbabel import pybel
        except ImportError:
            import pybel

        if verbosity < os_util.verbosity_level.extra_debug:
            pybel.ob.obErrorLog.SetOutputLevel(pybel.ob.obError)
        else:
            os_util.local_print('OpenBabel warning messages are on, expect a lot of output.',
                                msg_verbosity=os_util.verbosity_level.extra_debug, current_verbosity=verbosity)

        try:
            ob_molecule = pybel.readstring('pdb', ligand_data)
        except OSError:
            ob_molecule = pybel.readfile('pdb', ligand_data).__next__()

        ob_molecule.OBMol.AddNonPolarHydrogens()
        try:
            read_mol = obmol_to_rwmol(ob_molecule)
        except ValueError:
            ob_molecule.OBMol.AddHydrogens()
            read_mol = obmol_to_rwmol(ob_molecule)

    return read_mol


def generic_mol_read(ligand_data, ligand_format=None, no_checks=False, verbosity=0):
    """ Tries to read a ligand detecting formats and types

    Parameters
    ----------
    ligand_data : str
        Small molecule file name or small molecule data
    ligand_format : str
        Data format or file extension. Default: guess from file name
    no_checks : bool
        Ignore checks and go on.
    verbosity : int
        Sets verbosity level

    Returns
    -------
    rdkit.Chem.Mol
    """

    if not ligand_format:
        ligand_format = os.path.splitext(ligand_data)[1]

    if isinstance(ligand_data, rdkit.Chem.Mol):
        return ligand_data

    if ligand_format in ['mol2', '.mol2']:
        read_mol_data = rdkit.Chem.MolFromMol2Block(ligand_data, removeHs=False)
        if read_mol_data is None:
            read_mol_data = rdkit.Chem.MolFromMol2File(ligand_data, removeHs=False)
    elif ligand_format in ['mol', '.mol']:
        read_mol_data = rdkit.Chem.MolFromMolBlock(ligand_data, removeHs=False)
        if read_mol_data is None:
            read_mol_data = rdkit.Chem.MolFromMolFile(ligand_data, removeHs=False)
    elif ligand_format in ['pdb', '.pdb']:
        read_mol_data = read_small_molecule_from_pdb(ligand_data, verbosity=verbosity)
    else:
        read_mol_data = None

    if read_mol_data is None:
        if not no_checks:
            os_util.local_print('Failed to read pose data from {} with type {}'
                                ''.format(ligand_data, ligand_format),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError('Failed to read {}'.format(ligand_data))
        else:
            os_util.local_print('Failed to read pose data from {} with type {}. Because you are running with '
                                'no_checks, I will ignore and move on.'
                                ''.format(ligand_data, ligand_format),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            return None

    return read_mol_data
