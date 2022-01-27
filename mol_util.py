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

    import pybel

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

    # Assign atoms
    for index, each_atom in enumerate(rdkit_rwmol.GetAtoms()):
        new_atom = new_obmol.NewAtom()
        new_atom.SetAtomicNum(each_atom.GetAtomicNum())
        new_atom.SetFormalCharge(each_atom.GetFormalCharge())
        new_atom.SetImplicitValence(each_atom.GetImplicitValence())
        if each_atom.GetIsAromatic():
            new_atom.SetAromatic()
        new_atom.SetVector(rdkit_rwmol.GetConformer().GetAtomPosition(index).x,
                           rdkit_rwmol.GetConformer().GetAtomPosition(index).y,
                           rdkit_rwmol.GetConformer().GetAtomPosition(index).z)

    # Assing bonds
    for each_bond in rdkit_rwmol.GetBonds():
        new_obmol.AddBond(each_bond.GetBeginAtomIdx() + 1, each_bond.GetEndAtomIdx() + 1,
                          _bondorders[each_bond.GetBondType()])
        if each_bond.GetIsAromatic():
            new_obmol.GetBond(each_bond.GetBeginAtomIdx() + 1, each_bond.GetEndAtomIdx() + 1).SetAromatic()

    # FIXME: assign stereochemistry

    new_obmol.EndModify()

    os_util.local_print('Converted rdkit molecule SMILES {} to an openbabel molecule SMILES: {}'
                        ''.format(rdkit.Chem.MolToSmiles(rdkit_rwmol), pybel.Molecule(new_obmol).write('smi')),
                        current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.info)

    return new_obmol


def obmol_to_rwmol(openbabel_obmol, verbosity=0):
    """Converts a openbabel.OBMol to rdkit.RWMol

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

    import pybel

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

    # Assign atoms
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
        rdedmol.SetProp('_TriposAtomName', residue_iter.GetAtomID(each_atom))
        if each_atom.IsAromatic():
            rdedmol.GetAtomWithIdx(new_atom).SetIsAromatic(True)

        os_util.local_print('[DEBUG] These are the dummy atoms detected: dummy_atoms={}'.format(dummy_atoms),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.debug)

    # Assing bonds
    for each_bond in pybel.ob.OBMolBondIter(openbabel_obmol):
        rdedmol.AddBond(each_bond.GetBeginAtomIdx()-1, each_bond.GetEndAtomIdx()-1,
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
        os_util.local_print('Bond between atoms {} and {} converted to an UNSPECIFIED type'
                            ''.format(each_bond.GetBeginAtomIdx()-1, each_bond.GetEndAtomIdx()-1),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.debug)

    # FIXME: assign stereochemistry

    rdmol = rdedmol.GetMol()

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


def initialise_neutralisation_reactions():
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
            _reactions = initialise_neutralisation_reactions()
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


def adjust_query_properties(query_molecule, generic_atoms=False, ignore_charge=True, ignore_isotope=True, verbosity=0):
    """ Adjust query settings removing all charges, isotope, aromaticity and valence info from core_structure SMARTS

    :param rdkit.Chem.Mol query_molecule: query molecule
    :param bool generic_atoms: make atoms generic
    :param bool ignore_charge: set all atomic charges to 0
    :param bool ignore_isotope: ignore atomic isotopes
    :param int verbosity: controls the verbosity level
    :rtype: rdkit.Chem.Mol
    """

    os_util.local_print('Entering adjust_query_properties(query_molecule={} (SMILES={}), generic_atoms={}, '
                        'verbosity={})'
                        ''.format(query_molecule, rdkit.Chem.MolToSmiles(query_molecule), generic_atoms, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

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
                        ''.format(query_molecule, rdkit.Chem.MolToSmiles(query_molecule),
                                  new_query_molecule, rdkit.Chem.MolToSmiles(new_query_molecule)),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    return new_query_molecule


def num_explict_hydrogens(mol):
    """ Return the number of explicit hydrogens in molecular graph

    :param  rdkit.Chem.Mol mol: the input molecule
    :rtype: int
    """

    return sum([1 for i in mol.GetAtoms() if i.GetAtomicNum() == 1])


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
