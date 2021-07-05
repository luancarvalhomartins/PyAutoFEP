#! /usr/bin/env python3
#
#  test_os_util.py
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

import sys
import os_util


def test_detect_type():
    assert os_util.detect_type('-1') == -1
    assert os_util.detect_type('2.5') == 2.5
    assert os_util.detect_type('float("inf")') == 'float("inf")'
    assert os_util.detect_type('[1,2,3]') == [1, 2, 3]
    assert os_util.detect_type('{1,2,3}') == {1, 2, 3}
    assert os_util.detect_type('{"a": 1, "b": 2, "c": 3}') == {"a": 1, "b": 2, "c": 3}
    assert os_util.detect_type('off') is False
    assert os_util.detect_type('oN') is True
    assert os_util.detect_type('off', test_for_boolean=False) == 'off'
    assert os_util.detect_type(' a   = 1, b : 2    ;c=3', test_for_dict=True) == 'a   = 1, b : 2    ;c=3'
    assert os_util.detect_type(' a   = 1; b : 2    ;c=3', test_for_dict=True) == {"a": 1, "b": 2, "c": 3}
    assert os_util.detect_type(' a   = 1  2   3; b : 2    ;c=3', test_for_dict=True) == {"a": [1, 2, 3], "b": 2, "c": 3}
    assert os_util.detect_type('1 ; 2,3', test_for_list=True) == [1, 2, 3]
    assert os_util.detect_type('1 2 3', test_for_list=True) == [1, 2, 3]
    assert os_util.detect_type('1 2a 3', test_for_list=True) == '1 2a 3'
    assert os_util.detect_type('1 ; "2",3', test_for_list=True) == [1, "2", 3]
    assert os_util.detect_type('/home/user\n//\\//123', test_for_list=True) == ['/home/user', '//\\//123']
    assert os_util.detect_type('abc: 123\ndef:123\n   \n\n  das:abc   com  # comm', test_for_dict=True) == \
           {'abc': 123, 'def': 123, 'das': 'abc   com'}


def test_assembly_shell_command():
    from subprocess import run
    from tempfile import TemporaryDirectory
    import os

    with TemporaryDirectory() as tempdir:
        command_txt = os_util.assemble_shell_command(['sed'], ['s/123/456/'], cwd=tempdir, input_data='test123str\n',
                                                     die_on_error=True, output_file='testfile.txt')
        tmpscript = os.path.join(tempdir, 'testfile.sh')
        with open(tmpscript, 'w') as fh:
            fh.write(command_txt)
        os.chmod(tmpscript, 0o700)
        run(tmpscript, shell=True, capture_output=True)
        with open(os.path.join(tempdir, 'testfile.txt'), 'r') as fh:
            data = fh.read()
    assert data == 'test456str\n'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tests os_util.py')
    parser.add_argument('tests', type=str, nargs='*', default=['detect_type', 'assemble_shell_command'],
                        help="Run these tests (default: run all tests)")
    arguments = parser.parse_args()

    for each_test in arguments.tests:
        if each_test == 'detect_type':
            test_detect_type()
        elif each_test == 'assemble_shell_command':
            test_assembly_shell_command()
        else:
            raise ValueError('Unknown test {}'.format(each_test))



