#! /usr/bin/env python3
#
#  process_user_input.py
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

import configparser
import argparse
import all_classes
import os_util
import os


def add_argparse_global_args(argparse_handler, verbosity=0):
    """ Adds default arguments to an argparse.ArgumentParser

    :param argparse.ArgumentParser argparse_handler: argument handler to be edited
    :param int verbosity: sets the verbosity level
    """

    general = argparse_handler.add_argument_group('General options')
    general.add_argument('--threads', type=int, default=None,
                         help='Use this many threads (default: 0: max supported by the system)')
    general.add_argument('--progress_file', type=str, default=None,
                         help='Use this progress file to save and load data between runs (Default: progress.pkl)')
    general.add_argument('--mcs_type', choices=['graph', '3d'], default=None,
                         help='Select MCS algorithm. graph (default) is suitable for molecules where no sterocenters '
                              'are inverted on perturbations. 3d is spatial-guided MCS, which works on perturbed '
                              'stereocenters but may break rings. See documentation for further reference.')
    general.add_argument('--config_file', type=str, default=None, help='Read this configuration file.')
    general.add_argument('--plot', action='store_const', default=None, const=True,
                         help='Where possible, plot data using matplotlib.')
    general.add_argument('--no_checks', action='store_const', default=None, const=True,
                         help='Ignore all checks and try to go on (Default: off)')
    general.add_argument('-v', '--verbose', action='count',
                         help='Controls the verbosity level. -v: turns on some useful warnings; -vv: warnings + some '
                              'furhter info; -vvv: warnings + info + LOTS of debug messages + rdkit messages; -vvvv: '
                              'all the previous + (tons of) openbabel messages)')
    general.add_argument('--quiet', default=None, help='Be quiet', action='store_const', const=True)


def read_options(argument_parser, unpack_section='', user_config_file=None, default_internal_file=None,
                 verbosity=0):
    """ Process configuration files and command line arguments. Resolution order is arguments > user_config_file >
        default_config_file.

    :param argparse.ArgumentParser argument_parser: command line arguments to be processed
    :param str unpack_section: unpack all variables from this section from user_config_file (if present) and
                               default_config_file
    :param str user_config_file: read this configuration file, takes precedence over default_config_file
    :param str default_internal_file: read internal paths and vars from this file, this will not be superseeded by user
                                      input
    :param verbosity: set the verbosity level
    :rtype: all_classes.Namedlist
    """

    os_util.local_print('Entering read_options(argument_parser={}, unpack_section={}, user_config_file={}, '
                        'default_config_file={}, verbosity={})'
                        ''.format(argument_parser, unpack_section, user_config_file, default_internal_file, verbosity),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    internals = configparser.ConfigParser()
    if not default_internal_file:
        read_data = internals.read(os.path.join(os.path.dirname(__file__), 'config', 'internal.ini'))
    else:
        read_data = internals.read(default_internal_file)

    if not read_data:
        os_util.local_print('Failed to read internal data file. Cannot continue. Check your install, this should not '
                            'happen'.format(default_internal_file if default_internal_file else 'config/internal.ini'),
                            current_verbosity=verbosity, msg_verbosity=os_util.verbosity_level.error)
        raise SystemExit(-1)

    # Reads command line parameters
    arguments = argument_parser.parse_args()

    # Reads defaults from default_config_file or from default location
    default_config_file = os.path.join(os.path.dirname(__file__), internals['default']['default_config'])

    if arguments.config_file is not None and user_config_file is None:
        user_config_file = arguments.config_file

    result_data = configparser.ConfigParser()

    try:
        read_files = result_data.read(default_config_file)
    except IOError:
        os_util.local_print('Failed to read the configuration file {}. I cannot continue without it.'
                            ''.format(default_config_file),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    else:
        if not read_files:
            os_util.local_print('Failed to read the configuration file {}. I cannot continue without it.'
                                ''.format(default_config_file),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

    result_data = {key: dict(result_data.items(key)) for key in result_data.sections()}
    if user_config_file:
        user_file = configparser.ConfigParser()

        if not user_file.read(user_config_file):
            os_util.local_print('Failed to read the configuration file {}. I cannot continue without it.'
                                ''.format(user_config_file),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)

        result_data = os_util.recursive_update(result_data,
                                               {key: dict(user_file.items(key)) for key in user_file.sections()})

    if unpack_section:
        # Copy all info in section unpack_section to top level
        result_data = os_util.recursive_update(dict(result_data['globals']), dict(result_data[unpack_section]))

    # Overwrites values in result_data (ie: read from config files) with those from command line
    result_data.update(dict(filter(lambda x: x[1] is not None, vars(arguments).items())))

    # If values were not provided in config files, load them from argparse defaults (None for most cases)
    result_data.update({k: v for k, v in vars(arguments).items() if k not in result_data})

    # Detect all types in result_data
    result_data = all_classes.Namespace(os_util.recursive_map(os_util.detect_type, dict(result_data)))

    # Programmatically set some global variables
    if result_data.verbose is None:
        result_data.verbose = 0

    if result_data.quiet:
        if result_data.verbose > 0:
            os_util.local_print('I cannot be quiet and verbose at once. Please, select only one of them.',
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        else:
            result_data.verbose = -1

    if result_data.verbose <= 2:
        from rdkit.rdBase import DisableLog
        DisableLog('rdApp.error')

    if result_data.threads == 0 or result_data.threads is None:
        try:
            from os import sched_getaffinity
        except ImportError:
            from os import cpu_count
            result_data.threads = cpu_count()
        else:
            result_data.threads = len(sched_getaffinity(0))
    if type(result_data.threads) != int or result_data.threads < -1 or result_data.threads == 0:
        os_util.local_print('Invalid number of threads supplied or detected. Falling back to threads = 1',
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        result_data.threads = 1

    if result_data.no_checks is None:
        result_data.no_checks = False

    if result_data.progress_file is None:
        result_data.progress_file = 'progress.pkl'

    result_data['internal'] = all_classes.Namespace({key: dict(internals.items(key)) for key in internals.sections()})

    return result_data
