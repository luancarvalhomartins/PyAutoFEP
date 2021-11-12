#! /usr/bin/env python3
#
#  align_utils.py
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

import numpy
import os_util
from mol_util import obmol_to_rwmol
from importlib import import_module


def align_sequences_match_residues(mobile_seq, target_seq, seq_align_mat='BLOSUM80', gap_penalty=-1.0, verbosity=0):
    """ Align two aminoacid sequences using Bio.pairwise2.globalds and substution matrix seq_align_mat, return a tuple
    with two list of residues to be used in the 3D alignment (mobile, refence)

    :param str mobile_seq: sequence of mobile protein
    :param str target_seq: sequence of target protein
    :param str seq_align_mat: use this substution matrix from Bio.SubsMat.MatrixInfo
    :param float gap_penalty: gap penalty to the alignment; avoid values too low in module
    :param int verbosity: sets the verbosity level
    :rtype: tuple
    """
    try:
        from Bio.pairwise2 import align
        from Bio.Align import substitution_matrices
        seq_align_mat = substitution_matrices.load(seq_align_mat)
    except ImportError as error:
        os_util.local_print('Failed to import Biopython with error: {}\nBiopython is necessary to sequence'
                            'alignment. Sequences to be aligned:\nReference: {}\nMobile: {}'
                            ''.format(error, target_seq, mobile_seq),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise ImportError(error)
    except FileNotFoundError as error:
        available_matrices = substitution_matrices.load()
        os_util.local_print('Failed to import substitution matrix {} with error: {}\nSubstitution matrix must be one '
                            'of: {})'
                            ''.format(seq_align_mat, error, available_matrices),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise FileNotFoundError(error)
    else:
        align_result = align.globalds(target_seq, mobile_seq, seq_align_mat, gap_penalty,
                                      gap_penalty)[0]
        os_util.local_print('This is the alignment result to be used in protein alignment:\n{}'
                            ''.format(align_result),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        ref_align_str = [True if res_j != '-' else False
                         for res_i, res_j in zip(align_result[0], align_result[1])
                         if res_i != '-']
        mob_align_str = [True if res_i != '-' else False
                         for res_i, res_j in zip(align_result[0], align_result[1])
                         if res_j != '-']

        return mob_align_str, ref_align_str


def get_position_matrix(each_mol, each_mol_str=None, atom_selection=None, verbosity=0):
    """

    :param pybel.Molecule each_mol: molecule to get positions from
    :param list each_mol_str: a alignment string from where residues to be used will be read
    :param list atom_selection: use atoms matching this name (default: CA)
    :param int verbosity: sets the verbosity level
    :rtype: list
    """

    if atom_selection is None:
        atom_selection = ['CA']
    if each_mol_str is None:
        each_mol_str = [True for _ in range(len(each_mol.residues))]

    added_atoms = []
    return_list = []
    for each_residue, residue_alignment in zip(each_mol.residues, each_mol_str):
        for each_atom in each_residue.atoms:
            if each_atom.OBAtom.GetResidue().GetAtomID(each_atom.OBAtom).lstrip().rstrip() in atom_selection:
                atom_str = '{}{}{}'.format(each_atom.OBAtom.GetResidue().GetAtomID(each_atom.OBAtom),
                                           each_residue.name, each_residue.idx)
                if residue_alignment:
                    if atom_str not in added_atoms:
                        return_list.append(each_atom.OBAtom.GetVector())
                        added_atoms.append(atom_str)
                    else:
                        os_util.local_print('Atom {} found twice in your protein {}. Cannot handle multiple '
                                            'occupancies.'.format(atom_str, each_mol.title),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        raise SystemExit(1)

    return return_list


def align_protein(mobile_mol, reference_mol, align_method='openbabel', seq_align_mat='BLOSUM80',
                  gap_penalty=-1, verbosity=0):
    """ Align mobile_mol to reference_mol using method defined in align_method. Defaults to openbabel.OBAlign, which is
    fastest. rdkit's GetAlignmentTransform is much slower and may not work on larger systems.

    :param [rdkit.RWMol, pybel.Molecule] reference_mol: molecule to be used as alignment reference
    :param [rdkit.RWMol, pybel.Molecule] mobile_mol: rdkit.RWMol molecule to be aligned
    :param str align_method: method to be used, options are 'openbabel', 'rdkit'
    :param str seq_align_mat: use this matrix to sequence alignment, only used if sequences differ. Any value from
                                    Bio.SubsMat.MatrixInfo
    :param float gap_penalty: use this gap penalty to sequence alignment, only used if sequences differ.
    :param int verbosity: be verbosity
    :rtype: dict
    """

    os_util.local_print('Entering align_protein(mobile_mol={}, reference_mol={}, align_method={}, verbosity={})'
                        ''.format(mobile_mol.title, reference_mol.title, align_method, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    if align_method == 'rdkit':
        # Uses rdkit.Chem.rdMolAlign.GetAlignmentTransform to align mobile_mol to reference_mol
        import rdkit.Chem.rdMolAlign
        reference_mol_rwmol = obmol_to_rwmol(reference_mol)
        if reference_mol_rwmol is None:
            os_util.local_print('Could not internally convert reference_mol',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            if verbosity >= os_util.verbosity_level.info:
                os_util.local_print('Dumping data to receptor_mol_error.pdb',
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
                reference_mol.write('mol', 'receptor_mol_error.pdb')
            raise SystemExit(1)

        mobile_mol_rwmol = obmol_to_rwmol(mobile_mol)
        if mobile_mol_rwmol is None:
            os_util.local_print('Could not internally convert OpenBabel mobile_mol to a RDKit.Chem.Mol object.',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        os_utils.local_print('Done reading and converting reference_mol {} and mobile_mol {}'
                             ''.format(reference_mol_rwmol.GetProp('_Name'), mobile_mol_rwmol.GetProp('_Name')),
                             msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

        # FIXME: implement this
        transformation_mat = rdkit.Chem.rdMolAlign.GetAlignmentTransform(reference_mol_rwmol, mobile_mol_rwmol)
        raise NotImplementedError('rdkit aligment method not implemented')

    elif align_method == 'openbabel':
        # FIXME: implement a Biopython-only method
        from openbabel import OBAlign
        import pybel

        reference_mol_seq = reference_mol.write('fasta').split('\n', 1)[1].replace('\n', '')
        mobile_mol_seq = mobile_mol.write('fasta').split('\n', 1)[1].replace('\n', '')

        if reference_mol_seq != mobile_mol_seq:
            os_util.local_print('Aminoacid sequences of {} and {} differs:\nReference: {}\nMobile: {}'
                                ''.format(reference_mol.title, mobile_mol.title, reference_mol_seq, mobile_mol_seq),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
            mob_align_str, ref_align_str = align_sequences_match_residues(mobile_mol_seq, reference_mol_seq,
                                                                          seq_align_mat=seq_align_mat,
                                                                          gap_penalty=gap_penalty,
                                                                          verbosity=verbosity)

        else:
            ref_align_str = None
            mob_align_str = None

        # Creates a new molecule containing only the selected atoms of both proteins
        ref_atom_vec = get_position_matrix(reference_mol, ref_align_str)
        reference_mol_vec = pybel.ob.vectorVector3(ref_atom_vec)
        mob_atom_vec = get_position_matrix(mobile_mol, mob_align_str)
        mobile_mol_vec = pybel.ob.vectorVector3(mob_atom_vec)
        os_util.local_print('Done extracting Ca from {} and {}'.format(reference_mol.title, mobile_mol.title),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

        # Align mobile to reference using the Ca coordinates
        align_obj = OBAlign(reference_mol_vec, mobile_mol_vec)
        if not align_obj.Align():
            os_util.local_print('Failed to align mobile_mol {} to reference_mol {}'
                                ''.format(mobile_mol.title, reference_mol.title),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        os_util.local_print('Alignment RMSD is {}'.format(align_obj.GetRMSD()),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        # Prepare translation and rotation matrices
        reference_mol_center = numpy.array([[a.GetX(), a.GetY(), a.GetZ()] for a in reference_mol_vec]).mean(0)
        mobile_mol_center = numpy.array([[a.GetX(), a.GetY(), a.GetZ()] for a in mobile_mol_vec]).mean(0)
        translation_vector = pybel.ob.vector3(*reference_mol_center.tolist())
        centering_vector = pybel.ob.vector3(*(-mobile_mol_center).tolist())
        rot_matrix = align_obj.GetRotMatrix()
        rot_vector_1d = [rot_matrix.Get(i, j) for i in range(3) for j in range(3)]

        os_util.local_print('Alignment data:\n\tReference: {}\n\tMobile: {}\n\tCentering: {}\n\tTranslation: {}'
                            '\n\tRotation matrix:\n\t\t{}, {}, {}\n\t\t{}, {}, {}\n\t\t{}, {}, {}'
                            ''.format(reference_mol_center, mobile_mol_center, centering_vector, translation_vector,
                                      *rot_vector_1d),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.debug)

        return {'centering_vector': centering_vector, 'translation_vector': translation_vector,
                'rotation_matrix': rot_vector_1d}

    else:
        # TODO implement a internal alignment method
        os_util.local_print('Unknown alignment method {}. Currently, only "openbabel" is allowed.'.format(align_method),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise ValueError('Unknown alignment method {}.'.format(align_method))
