#! /usr/bin/env python3
#
#  savestate_util.py
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

import pickle
from all_classes import Namespace
import os
import os_util
from time import strftime
from shutil import copy2, copytree, rmtree


class SavableState(Namespace):
    """ A class which also behaves like a dict and can load and save to pickle

    Data structure = {
        mcs_dict: {
            frozenset([smiles_a, smiles_b]): mcs,
        },
        ligands_data: {
            molecule_name: {
                'molecule': rdkit.Chem.Mol,
                'topology': [top_file_a, top_file_b],          # GROMACS-compatible topology files
                'image': {
                    '2d_hs': str                               # SVG data of 2D depiction with Hs
                    '2d_nohs': str                             # SVG data of 2D depiction without Hs
                    'perturbations': {
                        molecule_b_name: {
                            common_core_smarts: {
                                '2d_hs': svg_data_hs,          # 2D SVG, with Hs, aligned to core, atoms highlighted
                                '2d_nohs': svg_data_nohs       # 2D SVG, no Hs, aligned to core, atoms highlighted
                            }
                        }
                    }
                }
            },
        },
        'superimpose_data': {
            'reference_pose_path': path_to_ref_pose_file,
            'ligand_dict': {
                molecule_name: rdkit.Chem.Mol,                 # Molecule aligned to path_to_ref_pose_file
            }
        },
        'thermograph': {
            'run_%d%m%Y_%H%M%S': {
                'runtype': runtype,                            # One of: ['optimal', 'star', 'wheel']
                'bias': molecule_name                          # Map was biased towards a molecule, if any
                'input_molecules': {
                    molecule_name: { 'molecule': rdkit.Chem.Mol }
                }
                'best_solution': networkx.Graph                # Graph of the best solution found
                'optimization_data': all_classes.AntSolver     # Containing data of every optimization step
            }
        }
    }
    """

    def __init__(self, input_file='', verbosity=0):
        """ Init SavableState by reading a pickle file, or doing nothing. If no input_file is given, a default name will
         be generated

        :param str input_file: save result to this file
        :param int verbosity: control verbosity level
        :rtype dict
        """

        super().__init__()

        if input_file:
            saved_data = self._read_data(input_file)
            for key, value in saved_data.items():
                setattr(self, key, value)
            try:
                if self.data_file != input_file and verbosity >= 0:
                    # User moved/renamed progress file or something wired happened
                    os_util.local_print('Progress file {} claims to be generated as file {}'
                                        ''.format(input_file, self.data_file), current_verbosity=verbosity,
                                        msg_verbosity=os_util.verbosity_level.warning)
                self.data_file = input_file
            except AttributeError:
                os_util.local_print('Progress file {} does not contain data_file data. Is it a progress file?'
                                    ''.format(input_file, self.data_file),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise ValueError('Invalid progress file')

        else:
            # User did not supplied a name, generate one
            from time import strftime
            self.data_file = 'savedata_{}.pkl'.format(strftime('%d%m%Y_%H%M%S'))

    def _read_data(self, input_file):
        """ Reads a pickle file, returns its object or None on fail

        :param str input_file: save result to this file
        :rtype: dict
        """

        try:
            with open(input_file, 'rb') as file_handler:
                raw_data = pickle.load(file_handler)
        except FileNotFoundError:
            # input_file does not exist, create file and save data to it
            self.data_file = input_file
            with open(input_file, 'wb') as file_handler:
                pickle.dump(self.__dict__, file_handler)
            return self.__dict__
        else:
            return raw_data

    def save_data(self, output_file='', verbosity=0):
        """ Save state to a pickle file

        :param str output_file: save result to this file
        :param int verbosity: controls verbosity level
        :rtype: bool
        """

        if output_file != '':
            self.data_file = output_file

        try:
            with open('.temp_pickle_test.pkl', 'wb') as file_handler:
                pickle.dump(self.__dict__, file_handler)
                os.fsync(file_handler.fileno())
        except (IOError, FileNotFoundError):
            os_util.local_print('Could not save data to {}'.format(self.data_file), current_verbosity=verbosity,
                                msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(1)
        else:
            try:
                os.replace('.temp_pickle_test.pkl', self.data_file)
            except FileNotFoundError as error:
                os_util.local_print('Failed to save progress data to file {}'.format(self.data_file),
                                    current_verbosity=verbosity,
                                    msg_verbosity=os_util.verbosity_level.error)
                raise FileNotFoundError(error)
            os_util.local_print('Saved data to {}'.format(self.data_file), current_verbosity=verbosity,
                                msg_verbosity=os_util.verbosity_level.debug)
            return True

    @property
    def __dict__(self):
        return dict(self)

    def update_mol_image(self, mol_name, save=False, verbosity=0):
        """ Generate mol images, if needed

        :param str mol_name: name of the molecule to be updated
        :param bool save: automatically save data
        :param int verbosity: controls verbosity level
        """

        import rdkit.Chem
        from rdkit.Chem.Draw import MolDraw2DSVG
        from rdkit.Chem.AllChem import Compute2DCoords

        this_mol = self.ligands_data[mol_name]['molecule']
        if 'images' not in self.ligands_data[mol_name]:
            draw_2d_svg = MolDraw2DSVG(300, 300)
            temp_mol = rdkit.Chem.Mol(this_mol)
            Compute2DCoords(temp_mol)
            draw_2d_svg.DrawMolecule(temp_mol)
            draw_2d_svg.FinishDrawing()
            svg_data_hs = draw_2d_svg.GetDrawingText()

            temp_mol = rdkit.Chem.RemoveHs(temp_mol)
            Compute2DCoords(temp_mol)
            draw_2d_svg = MolDraw2DSVG(300, 300)
            draw_2d_svg.DrawMolecule(temp_mol)
            draw_2d_svg.FinishDrawing()
            svg_data_no_hs = draw_2d_svg.GetDrawingText()

            self.ligands_data[mol_name]['images'] = {'2d_hs': svg_data_hs, '2d_nohs': svg_data_no_hs}
        if not save:
            self.save_data()


class UserStorageDirectory:

    def __init__(self, path='', verbosity=0):
        """ Constructs a UserStorageDirectory object

        :param str path: use this dir, default: $XDG_CONFIG_HOME/fep_automate
        :param int verbosity: verbosity level
        """

        if not path:
            try:
                self.path = os.environ['XDG_CONFIG_HOME']
            except KeyError:
                try:
                    self.path = os.path.join(os.environ['HOME'], '.config')
                except KeyError:
                    # Is this unix?
                    self.path = os.path.join(os.curdir, '.config')
                    os_util.local_print('You seem to be running on a non-UNIX system (or there are issues in your '
                                        'environment). Trying to go on, but you may experience errors.',
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        else:
            self.path = path

        self.path = os.path.join(self.path, 'fep_automate')

        try:
            os.mkdir(self.path)
        except FileExistsError:
            pass

    def create_file(self, file_name, contents, verbosity=0):
        """ Create file in storage dir

        :param str file_name: name of file to be created
        :param [str, bytes] contents: save this data to file
        :param int verbosity: verbosity level
        :rtype: bool
        """

        if isinstance(contents, str):
            mode = 'w'
        elif isinstance(contents, bytes):
            mode = 'wb'
        else:
            raise TypeError('str or bytes expected, got {} instead'.format(type(contents)))

        with open(os.path.join(self.path, file_name), mode=mode) as fh:
            fh.write(contents)

        # Save a backup of the file with and timestamp
        backup_name = os.path.join(self.path, '{}_{}{}'.format(os.path.basename(file_name), strftime('%H%M%S_%d%m%Y'),
                                                               os.path.splitext(file_name)))
        copy2(os.path.join(self.path, file_name), backup_name)

        return True

    def store_file(self, source, dest_file='', verbosity=0):
        """ Copy file or dir to storage dir

        :param str source: file to be copied
        :param str dest_file: new file name, default: use source_file name
        :param int verbosity: verbosity level
        :rtype: bool
        """

        if not dest_file:
            dest_file = os.path.basename(source)
            if dest_file == '':
                # source is a directory
                dest_file = source.split(os.sep)[-1]
            if dest_file in ['', '.']:
                os_util.local_print('Could not get a name from {} and dest_file was not supplied. Cannot go continue.'
                                    ''.format(source),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise ValueError('invalid source name')

        backup_name = os.path.join(self.path, '{}_{}{}'.format(os.path.basename(dest_file), strftime('%H%M%S_%d%m%Y'),
                                                               os.path.splitext(dest_file)))
        try:
            copy2(source, os.path.join(self.path, dest_file))
            copy2(source, backup_name)
        except IsADirectoryError:
            try:
                copytree(source, os.path.join(self.path, dest_file))
            except FileExistsError:
                rmtree(os.path.join(self.path, dest_file))
                copytree(source, os.path.join(self.path, dest_file))
            finally:
                copytree(source, os.path.join(self.path, backup_name))

        return True
