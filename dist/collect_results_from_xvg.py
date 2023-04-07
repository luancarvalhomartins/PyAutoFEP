#! /usr/bin/env python3
#
#  collect_results_from_xvg.py
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

import subprocess
import os
from collections import OrderedDict
import re
import argparse
import pickle


def process_xvg_to_dict(input_files, temperature=298.15):
    """ Process a list of xvg/edr files into a ready to produce a u_nk pandas.Dataframe

    :param list input_files: files to be processed
    :param float temperature: absolute temperature of the sampling
    :rtype: dict
    """

    # Reads XVG data into a dict
    potential_table = OrderedDict()
    pv_table = OrderedDict()
    for index, each_file in enumerate(input_files):

        this_title, this_data = read_xvg(each_file, temperature)
        try:
            name_parse = re.search(r'rerun_struct(.+)_coord(.+).xvg', each_file).groups()
        except AttributeError:
            # XVG is from the first run, we have extracted pV
            name_parse = re.search(r'lambda(\d+)', each_file).groups()
            lambda_struct = int(name_parse[0])
            if lambda_struct not in pv_table:
                pv_table[lambda_struct] = OrderedDict()
            for each_time, each_value in zip(this_title, this_data):
                pv_table[lambda_struct][each_time] = each_value
        else:
            # XVG is from a rerun, we have extracted Potential
            lambda_struct = int(name_parse[0])
            lambda_coord = int(name_parse[1])
            if lambda_struct not in potential_table:
                potential_table[lambda_struct] = OrderedDict()
            if lambda_coord not in potential_table[lambda_struct]:
                potential_table[lambda_struct][lambda_coord] = OrderedDict()
            for each_time, each_value in zip(this_title, this_data):
                potential_table[lambda_struct][lambda_coord][each_time] = each_value

    time_count = 0
    for i in range(len(potential_table)):
        time_count = max(time_count, len(potential_table[0][i]))

    struct_count = max([len(i) for i in potential_table.values()])
    converted_table = [[float('inf')] * (len(potential_table) + 2) for _ in range(struct_count * time_count)]

    # Reorganize dictionary in list of lists
    for index_struct, each_struct in potential_table.items():
        for index_coord, each_coord in each_struct.items():
            for index_time, time_potential in enumerate(each_coord.items()):
                try:
                    converted_table[index_coord * time_count + index_time][0] = time_potential[0]
                    converted_table[index_coord * time_count + index_time][1] = index_coord
                    converted_table[index_coord * time_count + index_time][index_struct + 2] = \
                        time_potential[1] + pv_table[index_struct][time_potential[0]]
                except (IndexError, KeyError) as error:
                    print('[ERROR] index_coord: {}; time_count: {}; index_time: {}; index: {}'
                          ''.format(index_coord, time_count, index_time, str(index_coord * time_count + index_time)))
                    print('[ERROR] Error message is: {}'.format(error))
                    print('[ERROR] Make sure all your FEP run to completion and that the time steps are consistent.')

    # Prepare indexes and column names
    indexes = ['time', 'fep-lambda']
    column_names = indexes.copy()
    [column_names.append(each_name) for each_name in sorted(potential_table.keys())]

    return {'column_names': column_names, 'indexes': indexes, 'converted_table': converted_table,
            'input_files': input_files, 'temperature': temperature}


def read_xvg(file_name, temperature=298.15):
    """ Parses an .xvg file

    :param str file_name: xvg file to be read
    :param float temperature: absolute T (default: 298.15)
    :rtype: tuple
    """

    beta = 1 / (8.3144621E-3 * temperature)

    try:
        with open(file_name) as file_handler:
            file_data = file_handler.readlines()
    except IOError:
        print('[ERROR] Could not read ' + file_name)
        raise SystemExit(1)

    title_list = []
    data_matrix = []
    for each_line in file_data:
        if (len(each_line) > 0) and ((each_line[0] != '#') and (each_line[0] != '@')):
            temp_data = list(map(float, each_line.split()))
            # Note: workaround for Gromacs floating-point error while summing time. Using a nstcalcenergy/nstenergy <
            #  10 will produce incorrect results
            title_list.append(round(temp_data[0], 2))
            #  1/kT*(U + pV)
            data_matrix.append(beta * (temp_data[1]))

    return title_list, data_matrix


def generate_xvg(input_files, gmx_path='gmx', verbosity=0):
    """ Generates XVG from EDR files using $gmx_path energy

    :param list input_files: list of edr files
    :param str gmx_path: path to Gromacs binary (Default: gmx)
    :param int verbosity: verbosity level
    :rtype: list
    """

    filelist = []
    # Call gmx energy for each edr file, rename file accordingly, move the xvg file to pwd
    for index, each_file in enumerate(input_files):
        if os.path.splitext(each_file)[1] == '.xvg':
            filelist.append(each_file)
            continue

        edr_filename = os.path.basename(each_file)
        if re.match(r'rerun_struct(.+)_coord(.+)\.edr', edr_filename) is not None:
            quantity = 'Potential'
            xvg_name = os.path.join(os.getcwd(), re.sub('\.edr', '.xvg', edr_filename))
        elif re.search(r'lambda(\d+)', each_file) is not None:
            quantity = 'pV'
            xvg_name = os.path.join(os.getcwd(), '{}.xvg'.format(re.search(r'lambda(\d+)', each_file).group(0)))
        else:
            print('Could not understand edr filename {}'.format(edr_filename))
            raise SystemExit(1)

        if os.path.exists(xvg_name):
            filelist.append(xvg_name)
        else:

            gmx_cmd = [gmx_path, "energy", '--quiet', "-f", each_file, '-o', xvg_name]
            try:
                gmx_process = subprocess.Popen(gmx_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)
                output, stderrdata = gmx_process.communicate(quantity)
            except (FileNotFoundError, OSError) as error:
                print('Failed to execute Gromacs with command line {}'.format(' '.join(gmx_cmd)))
                raise SystemExit(1)

            filelist.append(xvg_name)
    return filelist


if __name__ == '__main__':
    Parser = argparse.ArgumentParser(description='Read xvg data and saves an pickle file')
    Parser.add_argument('-i', '--input', required=True, type=str, nargs='+',
                        help='Energy files (.edr) or data files (.xvg)')
    Parser.add_argument('--gmx_bin', default='gmx', type=str,
                        help='Use this Gromacs executable. (Default: gmx)')
    Parser.add_argument('--output', default='unk_matrix.pkl', type=str,
                        help='Output data file (pkl). (Default: unk_matrix.pkl)')
    Parser.add_argument('--temperature', default=298.15, type=float,
                        help='Absolute temperature of the sampling. (Default: 298.15 K)')
    Arguments = Parser.parse_args()

    data = process_xvg_to_dict(generate_xvg(Arguments.input, Arguments.gmx_bin), temperature=Arguments.temperature)

    savedata = {'u_nk_data': data, 'read_files': Arguments.input, 'temperature': Arguments.temperature}

    with open(Arguments.output, 'wb') as fh:
        pickle.dump(data, fh)
