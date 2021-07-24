#! /usr/bin/env python3
#
#  os_util.py
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
import itertools
import re
import os
import shutil
import subprocess
from io import TextIOBase
from collections import namedtuple
import traceback
from ast import literal_eval

verbosity_level = namedtuple("VerbosityLevel", "error default warning info debug")(-1, 0, 1, 2, 3)


def makedir(dir_name, error_if_exists=False, parents=False, verbosity=0):
    """ Safely create a directory

    :param str dir_name: name of the directory to be created
    :param bool error_if_exists: throw an error if dir_name exists
    :param bool parents: create parent dirs as needed
    :param int verbosity: sets th verbosity level
    """

    local_print('Entering makedir(dir_name={}, error_if_exists={})'.format(dir_name, error_if_exists),
                msg_verbosity=verbosity_level.debug, current_verbosity=verbosity)

    if not parents:
        try:
            os.mkdir(dir_name)
        except OSError as error:
            if error.errno == 17:
                if error_if_exists:
                    local_print('Directory {} exists (and makedir was called with error_if_exists=True).'.format(dir_name),
                                msg_verbosity=verbosity_level.error, current_verbosity=verbosity)
                    raise SystemExit(1)
            else:
                local_print('Could not create directory {}. Error was {}'.format(dir_name, error),
                            msg_verbosity=verbosity_level.error, current_verbosity=verbosity)
                raise OSError(error)
    else:
        try:
            os.makedirs(dir_name, exist_ok=(not error_if_exists))
        except FileExistsError as error:
            local_print('Could not create directory {}. Error was {}'.format(dir_name, error),
                        msg_verbosity=verbosity_level.error, current_verbosity=verbosity)
            raise FileExistsError(error)


def read_file_to_buffer(filename, die_on_error=False, return_as_list=False, error_message=None, verbosity=0):
    """ Read and return file contents

    :param str filename: file to be read
    :param bool die_on_error: throw an error if fails to read
    :param bool return_as_list: return a list instead of str
    :param str error_message: if die_on_error=True, print this message error instead of default one
    :param int verbosity: sets th verbosity level
    :rtype: [str,list,bool]
    """

    local_print('Entering read_file_to_buffer(filename={}, die_on_error={}, return_as_list={})'
                ''.format(filename, die_on_error, return_as_list),
                msg_verbosity=verbosity_level.debug, current_verbosity=verbosity)

    try:
        with open(filename, 'r') as input_file:
            if return_as_list:
                data_buffer = input_file.readlines()
            else:
                data_buffer = input_file.read()
    except (IOError, TypeError) as error:
        if die_on_error:
            if error_message is None:
                error_message = 'Could not read file {} (and read_file_to_buffer was called with ' \
                                'die_on_error=False). Error was: {}'.format(filename, error)
            else:
                error_message += '\nCould not read file {} (and read_file_to_buffer wall called with ' \
                                 'die_on_error=False). Error was: {}'.format(filename, error)
            local_print(error_message, msg_verbosity=verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
        else:
            return False
    else:
        return data_buffer


def run_gmx(gmx_bin, arg_list, input_data='', output_file=None, return_output=False, alt_environment=None,
            cwd=None, die_on_error=True, verbosity=0):
    """ Run gmx_bin with arg_list

    :param str gmx_bin: path to Gromacs binary
    :param list arg_list: pass these args to gmx
    :param str input_data: data to be send to gmx, empty str (default) to send nothing
    :param str output_file: save output (stdout + stderr) to this file (default: None = don't save)
    :param bool return_output: if True, return stdout
    :param dict alt_environment: environment to be passed (on top of current) to Gromacs
    :param str cwd: run in this directory
    :param bool die_on_error: raise error if command returns a error code
    :param int verbosity: verbose level
    """

    this_env = os.environ.copy()
    if alt_environment is not None:
        this_env.update(alt_environment)

    if isinstance(gmx_bin, list):
        final_arg_list = gmx_bin[:]
    elif isinstance(gmx_bin, str):
        final_arg_list = [gmx_bin]
    else:
        local_print('Could not understand gmx bin input to run_gmx. gmx_bin = {} (type = {}). Invalid type'
                    ''.format(gmx_bin, type(gmx_bin)),
                    msg_verbosity=verbosity_level.error, current_verbosity=verbosity)
        raise TypeError("expected str or list, not {}".format(type(gmx_bin)))
    final_arg_list.extend(arg_list)

    gmx_handler = subprocess.Popen(final_arg_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   stdin=subprocess.PIPE, universal_newlines=True, env=this_env, cwd=cwd)
    if input_data != '':
        stdout, stderr = gmx_handler.communicate(input_data)
    else:
        stdout, stderr = gmx_handler.communicate()

    if die_on_error and gmx_handler.returncode != 0:
        local_print('Failed to run {} {}. Error code {}.\nCommand line was: {}\n\nstdout:\n{}\n\nstderr:\n{}'
                    ''.format(gmx_bin, arg_list[0], gmx_handler.returncode, [gmx_bin] + arg_list, stdout, stderr),
                    msg_verbosity=verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    else:
        if output_file is not None:
            with open(output_file, 'w') as fh:
                fh.write(stdout)
        if return_output:
            return stdout, stderr


def assemble_shell_command(gmx_bin, arg_list, input_data='', output_file=None, cwd=None, die_on_error=True,
                           verbosity=0):
    """ Return a gmx_bin command with arg_list

    :param [str, list] gmx_bin: path to Gromacs binary
    :param list arg_list: pass these args to gmx
    :param str input_data: data to be send to gmx, empty str (default) to send nothing
    :param str output_file: pipe stdout + stderr to this file
    :param str cwd: run in this directory
    :param bool die_on_error: test for return code
    :param int verbosity: verbose level
    """

    local_print('Entering assemble_shell_command(gmx_bin={}, arg_list={}, input_data={}, output_file={}, verbosity={})'
                ''.format(gmx_bin, arg_list, input_data, output_file, verbosity),
                msg_verbosity=verbosity_level.debug, current_verbosity=verbosity)

    return_output = '( cd {} && '.format(cwd) if cwd else ''
    for old, new in {'\n': r'\n', '"': r'\"'}.items():
        input_data = input_data.replace(old, new)
    return_output += 'printf "{}" | '.format(input_data) if input_data else ''

    if isinstance(gmx_bin, list):
        return_output += ' '.join(gmx_bin + arg_list)
    elif type(gmx_bin) == str:
        return_output += ' '.join([gmx_bin] + arg_list)
    else:
        local_print('Could not understand gmx bin input to run_gmx. gmx_bin = {} (type = {}). Invalid type'
                    ''.format(gmx_bin, type(gmx_bin)),
                    msg_verbosity=verbosity_level.error, current_verbosity=verbosity)
        raise TypeError("expected str or list, not {}".format(type(gmx_bin)))

    return_output += ' > {} 2>&1'.format(output_file) if output_file else ''
    return_output += r' )'.format(cwd) if cwd else ''
    return_output += ' || {{ echo "Failed to run command {} at line ${{LINENO}}" && exit; }}' \
                     ''.format(input_data) if die_on_error else ''

    return return_output


def detect_type(value, test_for_boolean=True, test_for_dict=False, test_for_list=False, list_max_split=0, verbosity=0):
    """ Detect and converts

    :param Any value: the value to be converted
    :param bool test_for_boolean: further tests for boolean values
    :param bool test_for_dict: further tests for flexible-formatted dicts
    :param bool test_for_list: further tests for flexible-formatted lists
    :param int list_max_split: if a list is detected and list_max_split is nonzero, at most list_max_split splits occur,
                               and the remainder of the string is returned as the final element
    of the list
    :param int verbosity: sets the verbosity level
    :rtype: any
    """

    # If value is not a str there is no need to process it
    if not value or not isinstance(value, str):
        return value

    value = value.lstrip().rstrip()

    try:
        converted_value = literal_eval(value)
    except (ValueError, SyntaxError):
        if test_for_boolean:
            if value.lower() in ['false', 'off', 'no']:
                return False
            elif value.lower() in ['true', 'on', 'yes']:
                return True

        if test_for_dict:
            import re
            try:
                converted_value = {detect_type(each_key): detect_type(each_value.split('#')[0], test_for_list=True,
                                                                      verbosity=verbosity)
                                   for each_pair in re.split('[;\n]', value)
                                   if len(each_pair.rstrip().lstrip()) > 0
                                   and each_pair.rstrip().lstrip()[0] not in ['#']
                                   for each_key, each_value in [re.split('[:=]', each_pair)]}
            except (ValueError, IndexError):
                if value.count(';') > 0 or value.count(',') > 0:
                    local_print('Your input "{}" seems to be a dictionary, but could not be parsed as such. Maybe you '
                                'want to check your input.'.format(value), msg_verbosity=verbosity_level.warning,
                                current_verbosity=verbosity)
                return value
            else:
                return converted_value

        if test_for_list:
            import re

            converted_value = [detect_type(each_value) for each_value in
                               re.split('[;,\n]', value, maxsplit=list_max_split) if each_value]
            if len(converted_value) <= 1:
                try:
                    converted_value = [int(i) for i in re.split(r'\s+', value, maxsplit=list_max_split) if i]
                except ValueError:
                    try:
                        converted_value = [float(i) for i in re.split(r'\s+', value, maxsplit=list_max_split) if i]
                    except ValueError:
                        if value.count(';') > 0 or value.count(',') > 0:
                            local_print('Your input "{}" seems to be a list, but could not be parsed as such. Maybe '
                                        'you want to check your input.'.format(value),
                                        msg_verbosity=verbosity_level.warning, current_verbosity=verbosity)
                        return value
            return converted_value

        return value
    else:
        # Convert tuples to lists (to make sure the return is mutable)
        if isinstance(converted_value, tuple):
            converted_value = list(converted_value)
        return converted_value


def local_print(this_string, msg_verbosity=0, logfile=None, current_verbosity=0):
    """ Prints formatted messages depending on the verbosity

    :param str this_string: string to be printed
    :param int msg_verbosity: verbosity level of the message
    :param logfile: prints all messages to this file as well
    :param int current_verbosity: current verbosity level
    """

    verbosity_name_dict = {verbosity_level.error: 'ERROR',
                           verbosity_level.warning: 'WARNING',
                           verbosity_level.info: 'INFO',
                           verbosity_level.debug: 'DEBUG'}

    if current_verbosity >= msg_verbosity or msg_verbosity == verbosity_level.error:
        if msg_verbosity == verbosity_level.debug:
            formatted_string = '[{}] {}'.format(verbosity_name_dict[msg_verbosity],
                                                '\n[{}] '.format(verbosity_name_dict[msg_verbosity])
                                                .join(this_string.split('\n')))
        elif msg_verbosity == verbosity_level.error:
            formatted_string = '[{}] {}'.format(verbosity_name_dict[msg_verbosity],
                                                '\n[{}] '.format(verbosity_name_dict[msg_verbosity])
                                                .join(this_string.split('\n')))
            formatted_string += '\n{:=^50}\n{}{:=^50}'.format(' STACK INFO ', ''.join(traceback.format_stack()),
                                                              ' STACK INFO ')
        elif msg_verbosity != verbosity_level.default:
            formatted_string = '[{}] {}'.format(verbosity_name_dict[msg_verbosity],
                                                '\n[{}] '.format(verbosity_name_dict[msg_verbosity])
                                                .join(this_string.split('\n')))
        else:
            formatted_string = this_string

        print(formatted_string)

    if logfile:
        if isinstance(logfile, str):
            # Logfile is a filename, append to it
            with open(logfile, 'a+') as fh:
                fh.write('{}\n'.format(this_string))
        elif isinstance(logfile, TextIOBase):
            # Logfile is textfile object, write to it
            logfile.write('{}\n'.format(this_string))
        else:
            raise TypeError("logfile must be str or TextIO, got {} instead".format(type(logfile)))


def recursive_map(function_f, iterable_i, args=(), kwargs=None):
    """ Recursively apply function_f to iterable_i, unpack args e kwargs """

    if kwargs is None:
        kwargs = {}
    if isinstance(iterable_i, str):
        return function_f(iterable_i, *args, **kwargs)
    elif isinstance(iterable_i, dict):
        tempdict = iterable_i.copy()
        for inner_key, inner_value in tempdict.items():
            tempdict[inner_key] = recursive_map(function_f, inner_value, *args, **kwargs)
        return tempdict
    else:
        return iterable_i


def recursive_update(base, updater):
    """ Implementation of a recursive update for dicts

    :param base: dictionary to be updated
    :param updater: source of new values
    :rtype: dict
    """

    for k, v in updater.items():
        if isinstance(v, dict) and isinstance(base.get(k, {}), dict):
            base[k] = recursive_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def natural_sort_key(s):
    """ Prepare a natural sorting key list. Copied from
    https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort

    :param str s: input list of str
    :rtype: list
    """

    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def wrapper_fn(fn, args, kwargs):
    return fn(*args, **kwargs)


def starmap_unpack(function, pool, args_iter=None, kwargs_iter=None):
    """ Wrapper around multiprocessing.starmap

    :param function: run this function with arg from and **kwargs
    :param multiprocessing.Pool pool: use this pool
    :param iter args_iter: use args from this iter
    :param iter kwargs_iter: use kwargs from this iter
    """

    if args_iter and kwargs_iter:
        assembled_args = zip(itertools.repeat(function), args_iter, kwargs_iter)
    elif args_iter is None:
        assembled_args = zip(itertools.repeat(function), itertools.repeat([]), kwargs_iter)
    elif kwargs_iter is None:
        assembled_args = zip(itertools.repeat(function), args_iter, itertools.repeat({}))
    else:
        raise ValueError('args_iter or kwargs_iter mandatory')

    return pool.starmap(wrapper_fn, list(assembled_args))


def inner_search(needle, haystack, apply_filter=None, find_last=False, die_on_error=False):
    """ Search for needle in items in haystack, returning the index of the first found occurrence

    :param needle: what to search. If callable, needle will be called for each item in haystack.
    :param list haystack: where to search
    :param [str, function] apply_filter: filterfalse lines using this function or, if str, by removing strings that
                                         starts with apply_filter
    :param bool find_last: search for the last occurrence instead of the first one
    :param bool die_on_error: when needle not in haystack, if true, raise VauleError, if False, return False
    :return: int
    """

    def search_func(needle, item):
        if callable(needle):
            return needle(item)
        elif isinstance(needle, set) and isinstance(item, set):
            return needle.issubset(item)
        else:
            try:
                return needle in item
            except TypeError:
                return needle == item

    if apply_filter is not None and not callable(apply_filter):
        filter_str = apply_filter
        apply_filter = lambda line: line.startswith(filter_str)

    last_occur, idx = -1, -1
    for idx, i in enumerate(haystack):
        if apply_filter is not None and apply_filter(i):
            continue
        if search_func(needle, i):
            if find_last:
                last_occur = idx
            else:
                break
    else:
        if die_on_error and ((not find_last) or (find_last and last_occur == -1)):
            raise ValueError("{} not in the iterator".format(needle))
        elif (not die_on_error) and ((not find_last) or (find_last and last_occur == -1)):
            return False
    if find_last:
        return last_occur
    else:
        return idx


def file_copy(src, dest, follow_symlinks=True, error_if_exists=False, verbosity=0):
    """ Copy file, data and metadata, optionally returning an error if dest exists

    :param str src: source file
    :param str dest: destination file or path
    :param bool follow_symlinks: if false, and src is a symbolic link, dst will be created as a symbolic link; if true
                                 and src is a symbolic link, dst will be a copy of the file src refers to.
    :param bool error_if_exists: raise an error if file exists
    :param int verbosity: verbosity level
    :return: str
    """

    if error_if_exists and (os.path.exists(dest) and not os.path.isdir(dest)):
        destfile = os.path.join(dest, os.path.basename(src)) if os.path.isdir(dest) else dest
        raise FileExistsError("File {} exists".format(destfile))
    else:
        local_print('Copying {} to {}'.format(src, dest),
                    current_verbosity=verbosity, msg_verbosity=verbosity_level.debug)
        return shutil.copy2(src, dest, follow_symlinks=follow_symlinks)
