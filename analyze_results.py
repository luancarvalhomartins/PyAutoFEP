#! /usr/bin/env python3
#

import matplotlib

matplotlib.use('SVG')
import matplotlib.pyplot as pl
pl.rcParams['svg.fonttype'] = 'none'

import multiprocessing
import threading
import os
import sys
import tempfile
import subprocess
import argparse
import pickle
import time
from collections import OrderedDict
import networkx
import pandas
import numpy
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.font_manager import FontProperties
from alchemlyb.estimators import MBAR, BAR
import pymbar.timeseries
import re
import all_classes
import savestate_util
import os_util
from all_classes import Namespace
from io import StringIO

kB_kJ = 1.3806504 * 6.02214129 / 1000.0  # Boltzmann's constant (kJ/mol/K).
formatted_energy_units = {
    'kJmol': Namespace({'text': 'kJ/mol', 'kB': kB_kJ}),
    'kcal': Namespace({'text': 'kcal/mol', 'kB': kB_kJ / 4.184}),
    'kBT': Namespace({'text': 'k_BT', 'kB': 1})
}

# The GROMACS unit is ps, time_units.unit.multi * ps will convert to desired the unit
formatted_time_units = {
    'us': Namespace({'text': 'µs', 'mult': 1e-6}),
    'ns': Namespace({'text': 'ns', 'mult': 1e-3}),
    'ps': Namespace({'text': 'ps', 'mult': 1}),
    'fs': Namespace({'text': 'fs', 'mult': 1e3})
}


def read_replica_exchange_from_gromacs(input_log_file, verbosity=0):
    """ Read and parse and Gromacs log file for replica exchange data

    :param str input_log_file: file to be read
    :param int verbosity: verbosity level
    :rtype: all_classes.Namespace
    """

    raw_data = os_util.read_file_to_buffer(input_log_file, die_on_error=True, return_as_list=True,
                                           error_message='Could not read Gromacs log file.',
                                           verbosity=verbosity)

    # Read and split replica exchange lines. First and last elements after split are not part of the data
    log_data = [re.split(r'\s*[0-9]+\s*', line.replace('Repl ex', ''))[1:-1]
                for line in raw_data if line.startswith('Repl ex')]

    if not len(log_data):
        os_util.local_print('No replica exchange info found in log file {}.'.format(input_log_file),
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        return None

    n_transitions = [0] * len(log_data[0])
    sampling_path = {val: [val] for val in range(len(log_data[0]) + 1)}
    # Count the number of exchanges and demux coordinates along the hamiltonians
    for line_n, line in enumerate(log_data):
        for start, transition in enumerate(line):
            # Test for allowed exchange for this round (see Gromacs REMD implementation for reasoning)
            if line_n % 2 != start % 2:
                continue
            current_hamilt_start = [k for k, v in sampling_path.items() if v[-1] == start][0]
            current_hamilt_end = [k for k, v in sampling_path.items() if v[-1] == start + 1][0]

            if not transition:
                sampling_path[current_hamilt_start].append(start)
                sampling_path[current_hamilt_end].append(start + 1)
            else:
                n_transitions[start] += 1
                sampling_path[current_hamilt_start].append(start + 1)
                sampling_path[current_hamilt_end].append(start)

        # There is a chance that the first or last hamiltonians could not exchange in this round, so the are one
        # element shorted. If it happened, the same coordinate is still on that hamiltonian. Test and fix this.
        found_lights = {len(hamiltonians) for hamiltonians in sampling_path.values()}
        if len(found_lights) > 1:
            for hamiltonians in sampling_path.values():
                if len(hamiltonians) < max(found_lights):
                    hamiltonians.append(hamiltonians[-1])

    # Prepare a empirical transition matrix
    empirical_transition_mat = numpy.zeros([len(log_data[0]) + 1, len(log_data[0]) + 1])
    for start, num in enumerate(n_transitions):
        empirical_transition_mat[start, start + 1] += num
        empirical_transition_mat[start + 1, start] += num
    empirical_transition_mat /= len(log_data)
    numpy.fill_diagonal(empirical_transition_mat, 1 - empirical_transition_mat.sum(axis=0))

    return all_classes.Namespace({'transition_matrix': empirical_transition_mat, 'sampling_path': sampling_path,
                                  'transitions_per_hamiltonian': n_transitions})


def convergence_analysis(u_nk, estimators=None, convergence_step=None, first_frame=0, calculate_tau_c=True,
                         detect_equilibration=False, temperature=298.15, units='kJmol', plot=True,
                         output_file=None, no_checks=False, verbosity=0, **kwargs):
    """ Run convergence analysis to forward and reversed u_nk at each convergence_step, plot results (with plot=True).
        kwargs will be passed to __init__ method of the estimator

    :param pandas.Dataframe u_nk: u_nk matrix
    :param dict estimators: estimator dictionary as {estimator_name: estimator_function}, default
                            {"mbar": alchemlyb.estimators.MBAR}
    :param [float, list] convergence_step: calculate ddG and ddG error every this time step, if float, or at these time
                                           steps, if list. Note that this values will are in regard of the u_nk data
                                           after first_frame
    :param int first_frame: start analysis from this time, in ps (default: 0, start from the first frame)
    :param bool calculate_tau_c: print subsampling info about data (default: Falso)
    :param bool detect_equilibration: automatically detects equilibration (default: off)
    :param float temperature: absolute temperature of the sampling (default: 298.15 K)
    :param str units: energy units to be used
    :param bool plot: plot convergence graphs
    :param [str, NoneType] output_file: save plot to this file, default: save a svg pwd
    :param bool no_checks: ignore checks and keep going
    :param int verbosity: verbosity level
    :rtype: dict
    """

    if not estimators:
        estimators = {'mbar': MBAR}

    try:
        beta = 1 / (formatted_energy_units[units].kB * temperature)
    except KeyError:
        os_util.local_print('Energy unit {} unknown. Please use one of {}'.format(units, [k for k in formatted_energy_units]),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    u_nk_after_first = u_nk.iloc[(u_nk.index.to_frame()['time'] >= first_frame).values]

    if isinstance(convergence_step, (int, float)):
        max_time = int(u_nk_after_first.index.to_frame()['time'].max())
        if convergence_step:
            time_value_list = list(numpy.arange(first_frame + convergence_step, max_time, convergence_step))
        else:
            time_value_list = []
        time_value_list.append(max_time)
    elif convergence_step is None:
        max_time = int(u_nk_after_first.index.to_frame()['time'].max())
        if max_time > 5000:
            convergence_step = 1000
        elif max_time > 15000:
            convergence_step = 2500
        elif max_time > 25000:
            convergence_step = 5000
        else:
            convergence_step = 500
        time_value_list = list(numpy.arange(first_frame + convergence_step, max_time, convergence_step))
        time_value_list.append(max_time)
    elif callable(getattr(convergence_step, '__getitem__', None)):
        time_value_list = list(convergence_step)
    else:
        os_util.local_print('convergence_step (value: {}) must be a list of float, but got type {}. Cannot continue.'
                            ''.format(convergence_step, type(convergence_step)),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise TypeError('float or list expected, got {} instead'.format(type(convergence_step)))

    os_util.local_print('Doing convergence analysis for the follwing times (in ps): {}'.format(time_value_list),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

    # Reverse each fep-lambda of u_nk_after_first, maintaining index, so the same
    columns = [*['time', 'fep-lambdas'], *u_nk_after_first.columns]
    index = ['time', 'fep-lambdas']
    reversed_data = pandas.DataFrame([], columns=columns).set_index(index)
    for i in sorted(set(u_nk_after_first.index.get_level_values(1))):
        this_block = u_nk_after_first.loc[(slice(None), i), :]
        this_index = this_block.index.get_level_values(0).to_numpy().reshape([-1, 1])
        block_array = numpy.concatenate((this_index, numpy.zeros(this_index.shape) + i, this_block[::-1].to_numpy()),
                                        axis=1)
        reversed_data = reversed_data.append(pandas.DataFrame(block_array, columns=columns).set_index(index))
    reversed_data.index = reversed_data.index.set_levels([reversed_data.index.levels[0],
                                                          reversed_data.index.levels[1].astype(int)])

    forward_ddgs = []
    reverse_ddgs = []
    forward_ddgs_errors = []
    reverse_ddgs_errors = []
    return_data = {each_name: {'forward': [], 'reverse': []} for each_name in estimators.keys()}
    return_data.update(units=units, convergence_steps=time_value_list, temperature=temperature, beta=beta)

    # Calculate ddG and error for each time for convergence plot. Note that this will be run with conv_step = max_time
    # even if convervenge_step == 0. The last value in
    for (dataframe, ddgs, ddgs_errors, direction) in [[u_nk_after_first, forward_ddgs, forward_ddgs_errors, 'forward'],
                                                      [reversed_data, reverse_ddgs, reverse_ddgs_errors, 'reverse']]:
        for conv_step in time_value_list:
            os_util.local_print('Doing MBAR using data from {} to {} ps'.format(first_frame, conv_step),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)

            this_data = dataframe.iloc[(dataframe.index.to_frame()['time'] <= conv_step).values]
            subsampled_u_nk_after_first = preprocess_data_table(this_data, detect_equilibration, calculate_tau_c,
                                                                verbosity=verbosity)

            for name, each_estimator in estimators.items():
                estimator_obj = each_estimator(**kwargs)

                alchemlyb_stdout = StringIO()
                sys.stdout = alchemlyb_stdout
                sys.stderr = alchemlyb_stdout
                try:
                    estimator_obj.fit(subsampled_u_nk_after_first)
                except BaseException as error:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    if not no_checks:
                        os_util.local_print('Error while running estimator {}. Error was: {}.\n{}'
                                            ''.format(name, error, alchemlyb_stdout.getvalue()),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                        if threading.current_thread() is threading.main_thread():
                            # Use os._exit to halt execution from from a thread
                            os._exit(1)
                        else:
                            raise SystemExit(1)
                    else:
                        os_util.local_print('Error while running estimator {}. Because you are running with no_checks, '
                                            'I will try to go on. Error was: {}.'
                                            ''.format(name, error),
                                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                finally:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    alchemlyb_stdout.close()

                this_ddg = -1 * estimator_obj.delta_f_[0].iloc[-1] / beta
                if numpy.isnan(this_ddg):
                    this_ddg = 0.0
                ddgs.append(this_ddg)
                this_ddg_error = estimator_obj.d_delta_f_[0].iloc[-1] / beta
                if numpy.isnan(this_ddg_error):
                    this_ddg_error = 0.0
                ddgs_errors.append(this_ddg_error)

                if name == 'mbar':

                    from distutils.version import LooseVersion
                    from alchemlyb import __version__
                    if LooseVersion(__version__) > LooseVersion('0.3.1'):
                        alchemlyb_stdout = StringIO()
                        sys.stdout = alchemlyb_stdout
                        sys.stderr = alchemlyb_stdout
                        overlap_matrix = estimator_obj._mbar.computeOverlap()['matrix']
                    else:
                        alchemlyb_stdout = StringIO()
                        sys.stdout = alchemlyb_stdout
                        sys.stderr = alchemlyb_stdout
                        overlap_matrix = estimator_obj._mbar.computeOverlap()[2]

                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    os_util.local_print('Full alchemlyb output while running computeOverlap:\n{}'
                                        ''.format(alchemlyb_stdout.getvalue()),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                    alchemlyb_stdout.close()

                    overlap_mat = pandas.DataFrame(overlap_matrix,
                                                   columns=u_nk.columns.values.tolist(),
                                                   index=u_nk.columns.values.tolist())
                    alchemlyb_stdout = StringIO()
                    sys.stdout = alchemlyb_stdout
                    sys.stderr = alchemlyb_stdout
                    enthalpy_entropy = dict(zip(['Delta_u', 'dDelta_u', 'Delta_s', 'dDelta_s'],
                                                [pandas.DataFrame(mat, columns=u_nk.columns.values.tolist(),
                                                                  index=u_nk.columns.values.tolist())
                                                 for mat in estimator_obj._mbar.computeEntropyAndEnthalpy(
                                                    warning_cutoff=False)[2:]]))
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    os_util.local_print('Full alchemlyb output while running computeEntropyAndEnthalpy:\n{}'
                                        ''.format(alchemlyb_stdout.getvalue()),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                    alchemlyb_stdout.close()
                else:
                    overlap_mat = None
                    enthalpy_entropy = {}

                return_data[name][direction].append({'ddg': -1 * estimator_obj.delta_f_[0].iloc[-1],
                                                     'error': estimator_obj.d_delta_f_[0].iloc[-1],
                                                     'delta_f_': estimator_obj.delta_f_,
                                                     'd_delta_f_': estimator_obj.d_delta_f_,
                                                     'overlap_matrix': overlap_mat,
                                                     'enthalpy_entropy': enthalpy_entropy})

    if plot:
        if len(time_value_list) <= 1:
            os_util.local_print('In order to plot the convergence graphs, a convergence_step is required, but you did '
                                'not provide one.',
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        else:
            os_util.local_print(f'Will plot ddg vs. time to {output_file}',
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
            plot_ddg_vs_time(numpy.array(forward_ddgs), numpy.array(reverse_ddgs), numpy.array(forward_ddgs_errors),
                             numpy.array(reverse_ddgs_errors), forward_timestep=time_value_list, energy_units=units,
                             output_file=output_file)

    return return_data


def plot_overlap_matrix(overlap_matrix, output_file=None, skip_lambda_index=()):
    """ Plots the probability of observing a sample from state i (row) in state j (column).  For convenience, the
    neighboring state cells are fringed in bold.

    :param numpy.array overlap_matrix: overlap matrix to be plotted
    :param str output_file: save plot to this file, default: svg to pwd
    :param list skip_lambda_index: do not use these lambda indexes in the analysis
    """

    if not output_file:
        output_file = "overlap_matrix.svg"

    n_states = overlap_matrix.shape[0]

    max_prob = overlap_matrix.max()
    fig = pl.figure(figsize=(n_states / 2., n_states / 2.))
    fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    for i in range(n_states):
        if i != 0:
            pl.axvline(x=i, ls='-', lw=0.5, color='k', alpha=0.25)
            pl.axhline(y=i, ls='-', lw=0.5, color='k', alpha=0.25)
        for j in range(n_states):
            if overlap_matrix[j, i] < 0.005:
                ii = ''
            elif overlap_matrix[j, i] > 0.995:
                ii = '1.00'
            else:
                ii = ("%.2f" % overlap_matrix[j, i])[1:]
            alf = overlap_matrix[j, i] / max_prob
            pl.fill_between([i, i + 1], [n_states - j, n_states - j], [n_states - (j + 1), n_states - (j + 1)],
                            color='k', alpha=alf)
            pl.annotate(ii, xy=(i, j), xytext=(i + 0.5, n_states - (j + 0.5)), size=8, textcoords='data', va='center',
                        ha='center', color=('k' if alf < 0.5 else 'w'))

    if skip_lambda_index:
        ks = [int(l) for l in skip_lambda_index]
        ks = numpy.delete(numpy.arange(n_states + len(ks)), ks)
    else:
        ks = list(range(n_states))
    for i in range(n_states):
        pl.annotate(ks[i], xy=(i + 0.5, 1), xytext=(i + 0.5, n_states + 0.5), size=10, textcoords=('data', 'data'),
                    va='center', ha='center', color='k')
        pl.annotate(ks[i], xy=(-0.5, n_states - (j + 0.5)), xytext=(-0.5, n_states - (i + 0.5)), size=10,
                    textcoords=('data', 'data'), va='center', ha='center', color='k')
    pl.annotate('$\lambda$', xy=(-0.5, n_states - (j + 0.5)), xytext=(-0.5, n_states + 0.5), size=10,
                textcoords=('data', 'data'),
                va='center', ha='center', color='k')
    pl.plot([0, n_states], [0, 0], 'k-', lw=4.0, solid_capstyle='butt')
    pl.plot([n_states, n_states], [0, n_states], 'k-', lw=4.0, solid_capstyle='butt')
    pl.plot([0, 0], [0, n_states], 'k-', lw=2.0, solid_capstyle='butt')
    pl.plot([0, n_states], [n_states, n_states], 'k-', lw=2.0, solid_capstyle='butt')

    cx = sorted(2 * list(range(n_states + 1)))
    cy = sorted(2 * list(range(n_states + 1)), reverse=True)
    pl.plot(cx[2:-1], cy[1:-2], 'k-', lw=2.0)
    pl.plot(numpy.array(cx[2:-3]) + 1, cy[1:-4], 'k-', lw=2.0)
    pl.plot(cx[1:-2], numpy.array(cy[:-3]) - 1, 'k-', lw=2.0)
    pl.plot(cx[1:-4], numpy.array(cy[:-5]) - 2, 'k-', lw=2.0)

    pl.xlim(-1, n_states)
    pl.ylim(0, n_states + 1)
    pl.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
    pl.close(fig)


def plot_curve_fitting(u_nk, ddg_all_pairs, ddg_error_all_pairs, output_file=None, skip_lambda_index=(),
                       num_bins=100):
    """ A graphical representation of what Bennett calls 'Curve-Fitting Method'. This function was ported/adapted from
        alchemlyb_analysis (https://github.com/MobleyLab/alchemical-analysis)

    :param pandas.DataFrame u_nk: u_nk matrix
    :param dict ddg_all_pairs: a dict where keys = estimator methods and values are a list of the estimates dG between
                               adjacent windows
    :param dict ddg_error_all_pairs: a dict where keys = estimator methods and values are a list of the errors estimates
    :param str output_file: save plot to this file, default: svg to pwd
    :param list skip_lambda_index: do not use these lambda indexes in the analysis
    :param int num_bins: number of bins
    """

    def find_optimal_min_max(ar):
        c = list(zip(*numpy.histogram(ar, bins=10)))
        thr = int(ar.size / 8.)
        mi, ma = ar.min(), ar.max()
        for (i, j) in c:
            if i > thr:
                mi = j
                break
        for (i, j) in c[::-1]:
            if i > thr:
                ma = j
                break
        return mi, ma

    def strip_zeros(a, aa, b, bb):
        z = numpy.array([a, aa[:-1], b, bb[:-1]])
        til = 0
        for i, j in enumerate(a):
            if j > 0:
                til = i
                break
        z = z[:, til:]
        til = 0
        for i, j in enumerate(b[::-1]):
            if j > 0:
                til = i
                break
        z = z[:, :len(a) + 1 - til]
        a, aa, b, bb = z
        return a, numpy.append(aa, 100), b, numpy.append(bb, 100)

    if not output_file:
        output_file = "bennet_curve_fitting.svg"

    # get N_k
    u_nk = u_nk.sort_index(level=u_nk.index.names[1:])
    groups = u_nk.groupby(level=u_nk.index.names[1:])
    N_k = [(len(groups.get_group(i)) if i in groups.groups else 0) for i in u_nk.columns]

    # and convert u_nk to u_kln
    u_kln = numpy.zeros([len(set(u_nk.index.get_level_values(1))), len(u_nk.columns),
                         len(set(u_nk.index.get_level_values(0)))])

    for i in u_nk.columns:
        for j in set(u_nk.index.get_level_values(1)):
            u_kln[j, i, :] = u_nk.loc[(slice(None), j), i]

    K = len(u_kln)
    yy = []
    for k in range(0, K - 1):
        upto = min(N_k[k], N_k[k + 1])
        righ = -u_kln[k, k + 1, : upto]
        left = u_kln[k + 1, k, : upto]
        min1, max1 = find_optimal_min_max(righ)
        min2, max2 = find_optimal_min_max(left)

        mi = min(min1, min2)
        ma = max(max1, max2)

        (counts_l, xbins_l) = numpy.histogram(left, bins=num_bins, range=(mi, ma))
        (counts_r, xbins_r) = numpy.histogram(righ, bins=num_bins, range=(mi, ma))

        counts_l, xbins_l, counts_r, xbins_r = strip_zeros(counts_l, xbins_l, counts_r, xbins_r)
        counts_r, xbins_r, counts_l, xbins_l = strip_zeros(counts_r, xbins_r, counts_l, xbins_l)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            log_left = numpy.log(counts_l) - 0.5 * xbins_l[:-1]
            log_righ = numpy.log(counts_r) + 0.5 * xbins_r[:-1]
            diff = log_left - log_righ
        yy.append((xbins_l[:-1], diff))

    sq = (len(yy)) ** 0.5
    h = int(sq)
    w = h + 1 + 1 * (sq - h > 0.5)
    scale = round(w / 3., 1) + 0.4 if len(yy) > 13 else 1
    sf = numpy.ceil(scale * 3) if scale > 1 else 0
    fig = pl.figure(figsize=(8 * scale, 6 * scale))
    matplotlib.rc('axes', facecolor='#E3E4FA')
    matplotlib.rc('axes', edgecolor='white')
    if skip_lambda_index:
        ks = [int(l) for l in skip_lambda_index]
        ks = numpy.delete(numpy.arange(K + len(ks)), ks)
    else:
        ks = list(range(K))
    for i, (xx_i, yy_i) in enumerate(yy):
        ax = pl.subplot(h, w, i + 1)
        ax.plot(xx_i, yy_i, color='r', ls='-', lw=3, marker='o', mec='r')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=6)
        ax.fill_between(xx_i, ddg_all_pairs['BAR'][i] - ddg_error_all_pairs['BAR'][i],
                        ddg_all_pairs['BAR'][i] + ddg_error_all_pairs['BAR'][i], color='#D2B9D3', zorder=-1)

        ax.annotate(r'$\mathrm{%d-%d}$' % (ks[i], ks[i + 1]), xy=(0.5, 0.9),
                    xycoords=('axes fraction', 'axes fraction'), xytext=(0, -2), size=14,
                    textcoords='offset points', va='top', ha='center', color='#151B54',
                    bbox=dict(fc='w', ec='none', boxstyle='round', alpha=0.5))
        pl.xlim(xx_i.min(), xx_i.max())
    pl.annotate(r'$\mathrm{\Delta U_{i,i+1}\/(reduced\/units)}$', xy=(0.5, 0.03), xytext=(0.5, 0),
                xycoords=('figure fraction', 'figure fraction'), size=20 + sf, textcoords='offset points',
                va='center', ha='center', color='#151B54')
    pl.annotate(r'$\mathrm{\Delta g_{i+1,i}\/(reduced\/units)}$', xy=(0.06, 0.5), xytext=(0, 0.5), rotation=90,
                xycoords=('figure fraction', 'figure fraction'), size=20 + sf, textcoords='offset points',
                va='center', ha='center', color='#151B54')
    pl.savefig(output_file, 'cfm.svg')
    pl.close(fig)


def preprocess_data_table(this_u_nk, detect_equilibration=False, calculate_tau_c=True, verbosity=0):
    """ Reads and preprocess and dataframe containing MD data

    :param pandas.DataFrame this_u_nk: dataframe containing MD data
    :param bool detect_equilibration: automatic detect equilibration using pymbar's routine
    :param bool calculate_tau_c: print subsampling info
    :param int verbosity: sets verbosity level
    :rtype: pandas.DataFrame
    """

    # Prepare to iterate over each coord data, first collect unique coord lambdas, then iterate over them
    coord_lambda_labels = [i[1] for i in this_u_nk.index.values]
    unique_coord_lambda = list(sorted(set(coord_lambda_labels)))

    processed_u_nk = None
    for each_coord_lambda in unique_coord_lambda:
        # Extract coordinate lambda data
        each_structure_block = this_u_nk.xs(each_coord_lambda, level=1, drop_level=False)
        each_structure_traj = each_structure_block[each_coord_lambda]

        if detect_equilibration:
            # Detect equilibration and print subsampling info
            equilibration, tau_c, uncorr_frames = pymbar.timeseries.detectEquilibration(each_structure_traj)
            os_util.local_print('Trajectory {}: equilibration endeded at frame {}.'
                                ''.format(each_coord_lambda, equilibration),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

            each_structure_traj = each_structure_traj[equilibration:]
            if calculate_tau_c:
                uncorrelated_index = pymbar.timeseries.subsampleCorrelatedData(each_structure_traj, g=tau_c)
                each_structure_block = each_structure_block[equilibration:]
                # each_structure_block = each_structure_block.iloc[uncorrelated_index]
                os_util.local_print('Trajectory {}: tau_c is {], therefore there are {} uncorrelated frames'
                                    ''.format(each_coord_lambda, tau_c, len(uncorr_frames)),
                                    msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
            else:
                each_structure_block = each_structure_block[equilibration:]
        elif calculate_tau_c:
            # Subsample trajectory
            uncorrelated_index = pymbar.timeseries.subsampleCorrelatedData(each_structure_traj)
            # each_structure_block = each_structure_block.iloc[uncorrelated_index]
            os_util.local_print('Trajectory {} have {} uncorrelated frames'
                                ''.format(each_coord_lambda, len(uncorrelated_index)),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
        elif verbosity > 0:
            os_util.local_print('Trajectory {} is {} frames long.'
                                ''.format(each_coord_lambda, len(each_structure_traj)),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        # Rejoin data
        if processed_u_nk is not None:
            processed_u_nk = processed_u_nk.append(each_structure_block)
        else:
            processed_u_nk = each_structure_block

    return processed_u_nk


def plot_ddg_vs_time(forward_ddgs, reverse_ddgs, forward_ddg_errors, reverse_ddg_errors, forward_timestep,
                     reverse_timestep=None, energy_units='kJmol', time_units='ns', colormap='viridis',
                     output_file='ddg_vs_time.svg', verbosity=0):
    """ Plots the free energy change computed using the equilibrated snapshots between the proper target time frames in
        both forward and reverse directions. This function was ported/adapted from alchemlyb_analysis
        (https://github.com/MobleyLab/alchemical-analysis)

    :param numpy.array forward_ddgs: calculated values for forward ddG
    :param numpy.array reverse_ddgs: calculated values for reverse ddG
    :param numpy.array forward_ddg_errors: associated errors to forward_ddgs
    :param numpy.array reverse_ddg_errors: associated errors to reverse_ddgs
    :param numpy.array forward_timestep: timesteps used to calculate values in forward_ddG
    :param [NoneType, numpy.array] reverse_timestep: timesteps used to calculate values in reverse_ddG (default:
                                                     forward_timestep in reverse order)
    :param str energy_units: energy units to be used, one of kJmol, kcal or kBT
    :param str time_units: time units to use in x axis, one of us, ns, ps, fs
    :param str colormap: matplotlib color map to be used
    :param str output_file: save plot to this file, default: save ddg_vs_time.svg to current dir
    :param int verbosity: set verbosity level
    """

    if not reverse_timestep:
        reverse_timestep = forward_timestep

    # Get two colors from colormap
    colormap = cm.get_cmap(colormap)
    forward_color = colormap(0.8)
    reverse_color = colormap(0.2)

    # The input time units in ps, lets convert it time_units
    reverse_timestep = reverse_timestep * formatted_time_units[time_units].mult
    forward_timestep = forward_timestep * formatted_time_units[time_units].mult

    fig, ax = pl.subplots(figsize=(5, 4))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    line1, _, _ = pl.errorbar(forward_timestep, forward_ddgs, yerr=forward_ddg_errors, color=forward_color, ls='-',
                              lw=2, marker='o', mew=2.5, mec=forward_color, ms=6, zorder=2)
    pl.fill_between(forward_timestep, forward_ddgs + forward_ddg_errors,
                    forward_ddgs - forward_ddg_errors, color=forward_color, alpha=0.3, zorder=-4)
    line2, _, _ = pl.errorbar(reverse_timestep, reverse_ddgs, yerr=reverse_ddg_errors, color=reverse_color, ls='-',
                              lw=2, marker='s', mew=2.5, mec=reverse_color, ms=6, zorder=1)
    pl.fill_between(forward_timestep, reverse_ddgs + reverse_ddg_errors,
                    reverse_ddgs - reverse_ddg_errors, color=reverse_color, alpha=0.3, zorder=-5)

    ax.set_xlim(forward_timestep[0], forward_timestep[-1])
    pl.yticks(fontsize=10)
    units = formatted_energy_units[energy_units].text
    ax.set_xlabel('Simulation time ({})'.format(formatted_time_units[time_units].text), fontsize=12)
    ax.set_ylabel('ΔΔG ({})'.format(units), fontsize=12)
    ax.set_xticks(forward_timestep)
    ax.set_xticklabels(['{:.1f}'.format(i) for i in forward_timestep])
    pl.legend((line1, line2), ['Forward', 'Reverse'], loc='best', prop=FontProperties(size=12))
    pl.tight_layout()
    pl.savefig(output_file)
    pl.close(fig)


def get_color(colors):
    return ['#00FFCC']


def plot_ddg_vs_lambda1(ddg_all_pairs, ddg_error_all_pairs, units='kJmol', colors='tab20', output_file=None):
    """ Plots the free energy differences evaluated for each pair of adjacent states for all methods. This function was
        ported/adapted from alchemlyb_analysis (https://github.com/MobleyLab/alchemical-analysis)

    :param dict ddg_all_pairs: a dict where keys = estimator methods and values are a list of the estimates dG between
                               adjacent windows
    :param dict ddg_error_all_pairs: a dict where keys = estimator methods and values are a list of the errors estimates
    :param str units: energy units to be used
    :param [str, list] colors: use this colors to plot bars, if list, use colors in the list order, if str, will
                               generate a pallete from the correspondingly matplotlib colormap. Default: use tab20
                               colormap
    :param str output_file: save plot to this file, default: svg to pwd
    """

    if not output_file:
        output_file = 'ddg_vs_lambda1.svg'

    colors = get_color(colors)

    n_values = len(ddg_all_pairs[list(ddg_all_pairs.keys())[0]])
    x = numpy.arange(n_values)
    if x[-1] < 8:
        fig = pl.figure(figsize=(8, 6))
    else:
        fig = pl.figure(figsize=(len(x), 6))
    width = 1. / (n_values + 1)
    elw = 30 * width

    lines = tuple()
    for i, ((name, values), (_, error)) in enumerate(zip(sorted(ddg_all_pairs.items()),
                                                         sorted(ddg_error_all_pairs.items()))):
        y = [each_value / formatted_energy_units[units].kB for each_value in values]
        ye = [each_value / formatted_energy_units[units].kB for each_value in error]
        line = pl.bar(x + float(i) / len(ddg_all_pairs), y, color=colors, yerr=ye, lw=0.5,
                      error_kw={'elinewidth': 0.5, 'ecolor': 'black', 'capsize': 0.5})
        lines += (line[0],)
    pl.xlabel('States', fontsize=12, color='#151B54')
    pl.ylabel(r'$\Delta G$ ' + formatted_energy_units[units].text, fontsize=12, color='#151B54')
    pl.xticks(x + 0.5 * width * len(ddg_all_pairs), tuple(['%d-%d' % (i, i + 1) for i in x]), fontsize=8)
    pl.yticks(fontsize=8)
    pl.xlim(x[0], x[-1] + len(lines) * width)
    ax = pl.gca()
    for each_dir in ['right', 'top', 'bottom']:
        ax.spines[each_dir].set_color('none')
    ax.yaxis.set_ticks_position('left')
    for tick in ax.get_xticklines():
        tick.set_visible(False)

    leg = pl.legend(lines, ddg_all_pairs.keys(), loc=0, ncol=2, prop=FontProperties(size=10), fancybox=True)
    leg.get_frame().set_alpha(0.5)
    pl.title('The free energy change breakdown', fontsize=12)
    pl.savefig(output_file, bbox_inches='tight')
    pl.close(fig)


def plot_coordinates_demuxed_scatter(sampling_path, n_rows=None, n_cols=None, max_time=None,
                                     output_file=None):
    """ Plot the demuxed coordinate trajectory along hamiltonians of the multisim as multiple scatter plots

    :param dict sampling_path: demuxed coordinates along trajectories
    :param [int, NoneType] n_rows: number of rows in subplot, default: auto
    :param [int, NoneType] n_cols: number of columns in subplot, default: auto
    :param [float, NoneType] max_time: Total simulation length; if None, will not convert frame
                                             number to time
    :param str output_file: save plot to this file, default: svg to pwd
    """
    if not output_file:
        output_file = 'hrex_trajectory_demux.svg'

    if n_rows and n_cols is None:
        n_cols = int(numpy.ceil(len(sampling_path) / float(n_rows)))
    elif n_cols and n_rows is None:
        n_rows = int(numpy.ceil(len(sampling_path) / float(n_cols)))
    elif n_cols is None and n_rows is None:
        n_cols = int(numpy.sqrt(len(sampling_path)))
        n_rows = int(numpy.ceil(len(sampling_path) / float(n_cols)))

    # Plot paths
    figure, axes = pl.subplots(n_rows, n_cols, figsize=[6, 6])
    figure.subplots_adjust(wspace=0, hspace=0)

    for i, a in zip(range(len(sampling_path)), axes.flatten()):
        a.plot(sampling_path[i], '.', color='#000000')
        if i % n_cols == 0:
            a.set_ylabel('Hamiltonian')
        elif i % n_cols == n_cols - 1:
            a.set_yticklabels([])
            a.set_yticks([])
            a.yaxis.set_label_position("right")
            a.set_ylabel('Reps {}'.format(', '.join([str(j) for j in
                                                     numpy.arange(n_rows * i, n_rows * i + n_rows, 1)])))
        else:
            a.set_yticklabels([])
            a.set_yticks([])

        if i < n_rows * n_cols - n_rows:
            a.set_xticklabels([])
        if i < n_cols:
            a.set_title('Reps {}'.format(', '.join([str(j) for j in numpy.arange(n_rows * i, n_rows * i + n_rows, 1)])))
        else:
            if max_time is not None:
                locs = a.get_xticks()
                a.set_xticks([j * max_time for j in locs])
                a.set_xlabel('Time (ns)')
            else:
                a.set_xlabel('Frame')
        a.set_ylim([-1, len(sampling_path)])

    for i in range(n_rows * n_cols - 1, len(sampling_path) - 1, -1):
        axes.flatten()[i].set_xticklabels([])
        axes.flatten()[i].set_yticklabels([])

    figure.suptitle('Replica trajectory along hamiltonians', fontsize=12)
    pl.savefig(output_file)
    pl.close(figure)


def plot_stacked_bars(data_matrix, bar_width=0.5, colormap='tab20', output_file=None):
    """ Plot a stacked bar plot from a nxn numpy array

    :param numpy.array data_matrix: data to be plotted
    :param float bar_width: width of the bar
    :param str colormap: matplotlib.cm color map
    :param str output_file: save plot to this file, default: svg to pwd
    """

    if not output_file:
        output_file = 'hrex_coord_hamiltonians.svg'

    n_rep = data_matrix.shape[0]
    fig, ax = pl.subplots(figsize=[4, 4])
    colormap = cm.get_cmap(colormap, n_rep)
    data_matrix = numpy.divide(data_matrix, data_matrix[:, 0].sum())
    ind = numpy.arange(n_rep)
    # FIXME: if the user uses a uniform colormap, this will fail
    stacked_bars = [ax.bar(ind, data_matrix[0], bar_width, color=colormap.colors[0])]
    for i in numpy.arange(n_rep):
        stacked_bars.append(ax.bar(ind, data_matrix[i], bar_width, bottom=data_matrix[0:i, :].sum(axis=0),
                                   color=colormap.colors[i]))

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Hamiltonian')
    ax.set_title('Replica sampling per hamiltonian')

    norm = BoundaryNorm(numpy.linspace(0, n_rep, n_rep + 1), colormap.N)
    colorbar_handler = pl.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap),
                                   ticks=numpy.arange(0, n_rep + 1) + 0.5)
    colorbar_handler.ax.set_yticklabels(numpy.arange(0, n_rep + 1))
    colorbar_handler.set_label('Replica', rotation=270)

    pl.tight_layout()
    pl.savefig(output_file)
    pl.close(fig)


def analyze_perturbation(perturbation_input, perturbation_data=None, gromacs_log='', estimators_data=None,
                         analysis_types=('all'), convergence_analysis_step=False, start_time=0, calculate_tau_c=True,
                         detect_equilibration=False, temperature=None, units='kJmol', plot=True, output_directory=None,
                         no_checks=False, verbosity=0):
    """ Run analysis for a perturbation edge, optionally, plot data

    :param str perturbation_input: pickle file generated by collect_results_from_xvg[.py]
    :param dict perturbation_data: perturbation data generated by collect_results_from_xvg[.py]. If this is supplied,
                                   perturbation_input will not be read.
    :param str gromacs_log: mdrun log to extract replica exchange data
    :param dict estimators_data: estimators to be used as {estimator_name: callable}, default: {'mbar':
                                 alchemlyb.estimators.mbar}
    :param list analysis_types: run these analysis, default: run all
    :param float convergence_analysis_step: use this step size to convergence analysis and, optionally, plots
    :param int start_time: start analysis from this time
    :param bool calculate_tau_c: print subsampling info about series
    :param bool detect_equilibration: automatically detect (and ignore) non-equilibrium region
    :param float temperature: absolute temperature of the sampling, default: read from perturbation_input
    :param str units: use this unit in analysis
    :param bool plot: plot data
    :param str output_directory: save plots to this dir, default: `pwd`
    :param bool no_checks: ignore checks and try to go on
    :param int verbosity: set verbosity
    :rtype: dict
    """

    if not estimators_data:
        from alchemlyb.estimators import MBAR
        estimators_data = {'mbar': MBAR}

    if output_directory is None:
        output_directory = os.getcwd()

    os_util.local_print('Analyzing {}, with {} estimators, and {} as output units'
                        ''.format(perturbation_input, estimators_data.keys(), formatted_energy_units[units].text),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    if perturbation_data is None:
        with open(perturbation_input, 'rb') as fh:
            try:
                perturbation_data = pickle.load(fh)
            except FileNotFoundError:
                os_util.local_print('Input file {} not found. Please, check your input option or argument.'
                                    ''.format(perturbation_input),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
            except BaseException as error:
                os_util.local_print('Failed to read input file {}. Please, check the input file integrity. Error was '
                                    '{}'.format(perturbation_input, error),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)

    # Read and check temperature
    if temperature and 'temperature' in perturbation_data:
        if temperature == perturbation_data['temperature']:
            os_util.local_print('You supplied temperature and there is also a temperature in file {}. Both are {} '
                                'K, so I am going on.'
                                ''.format(perturbation_input, perturbation_data['temperature']),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
            pass
        elif no_checks:
            os_util.local_print('Temperature found in {} is {} K, but input temperature (from command line or config '
                                'file) is {} K. You are using no_checks, so I will ignore temperature in {} and use '
                                '{} K.'.format(perturbation_input, perturbation_data['temperature'], temperature,
                                               perturbation_input, temperature),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        else:
            os_util.local_print('Temperature found in {} is {} K, but input temperature (from command line or config '
                                'file) is {} K. Please, check your input or run with no_checks.'
                                ''.format(perturbation_input, perturbation_data['temperature'], temperature),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise SystemExit(1)
    elif 'temperature' in perturbation_data:
        temperature = perturbation_data['temperature']
        os_util.local_print('Reading temperature from {}: {} K'
                            ''.format(perturbation_input, perturbation_data['temperature']),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)
    else:
        os_util.local_print('Temperature not found in {} and not given as and input option. Cannot continue.'
                            ''.format(perturbation_input),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)
    if temperature <= 0:
        os_util.local_print('Invalid temperature {} K.'
                            ''.format(temperature),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    # Check for required fields on read data (do not check for temperature, as the user can select it using input)
    if not set(perturbation_data).intersection({'converted_table', 'column_names', 'indexes'}) == \
           {'converted_table', 'column_names', 'indexes'}:
        os_util.local_print('Could not parse data from file {}. Please, check the input file integrity. I read the '
                            'following data: {}'
                            ''.format(perturbation_input, ', '.join(perturbation_data.keys())),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
        raise SystemExit(1)

    # Create the final DataFrame, remove zero-valued columns
    u_nk = pandas.DataFrame(perturbation_data['converted_table'],
                            columns=perturbation_data['column_names']).set_index(perturbation_data['indexes'])

    u_nk = u_nk[u_nk < float('inf')].dropna()

    if 'convergence' in analysis_types or 'all' in analysis_types:
        ddg_data = convergence_analysis(u_nk, estimators=estimators_data, convergence_step=convergence_analysis_step,
                                        first_frame=start_time, calculate_tau_c=calculate_tau_c,
                                        detect_equilibration=detect_equilibration, temperature=temperature,
                                        units=units, plot=plot,
                                        output_file=os.path.join(output_directory, 'ddg_vs_time.svg'),
                                        no_checks=no_checks, verbosity=verbosity)
    else:
        ddg_data = convergence_analysis(u_nk, estimators=estimators_data, convergence_step=0.0,
                                        first_frame=start_time, calculate_tau_c=calculate_tau_c,
                                        detect_equilibration=detect_equilibration, temperature=temperature,
                                        units=units, plot=False,
                                        output_file=os.path.join(output_directory, 'ddg_vs_time.svg'),
                                        no_checks=no_checks, verbosity=verbosity)

    if plot:
        if 'overlap_matrix' in analysis_types or 'all' in analysis_types:
            if 'mbar' not in ddg_data:
                os_util.local_print('MBAR is required to "overlap_matrix", but your estimators are {}.'
                                    ''.format(', '.join([k for k in estimators_data.keys()])),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
            else:
                plot_overlap_matrix(ddg_data['mbar']['forward'][-1]['overlap_matrix'].to_numpy(),
                                    output_file=os.path.join(output_directory, 'overlap_matrix.svg'))

        if 'neighbor_ddg' in analysis_types or 'all' in analysis_types:
            plot_ddg_vs_lambda1({'mbar': ddg_data['mbar']['forward'][-1]['delta_f_'][0].to_numpy()},
                                {'mbar': ddg_data['mbar']['forward'][-1]['d_delta_f_'][0].to_numpy()},
                                units=units, output_file=os.path.join(output_directory, 'ddg_vs_lambda1.svg'))

        if 'replica_exchange' in analysis_types or 'all' in analysis_types:
            if not gromacs_log:
                os_util.local_print('A Gromacs log file is required to Replica Exchange analysis, but no log file was '
                                    'found. Going on.',
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
            else:
                hrex_data = read_replica_exchange_from_gromacs(gromacs_log, verbosity=verbosity)
                if hrex_data is not None:
                    plot_coordinates_demuxed_scatter(hrex_data['sampling_path'],
                                                     output_file=os.path.join(output_directory,
                                                                              'hrex_trajectory_demux.svg'))
                    plot_overlap_matrix(hrex_data['transition_matrix'],
                                        output_file=os.path.join(output_directory, 'hrex_transition_matrix.svg'))

                    hamiltonian_vs_coord = numpy.array([[each_coord.count(i) for i in
                                                         range(len(hrex_data['sampling_path']))]
                                                        for _, each_coord in
                                                        sorted(hrex_data['sampling_path'].items())])

                    plot_stacked_bars(hamiltonian_vs_coord,
                                      output_file=os.path.join(output_directory, 'hrex_coord_hamiltonians.svg'))
                else:
                    os_util.local_print('No replica-exchange info found in the GROMACS log file {}. Assuming you did '
                                        'not run using HREX.'.format(gromacs_log),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

    os_util.local_print('Done analyzing {}'.format(perturbation_input),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

    return ddg_data


def sum_path(g0, path, ddg_key='final_ddg'):
    """ Sums ddG along a path

    :param networkx.DiGraph g0:  graph containing ddG (as a ddg edge data)
    :param list path: path to sum over.
    :param Any ddg_key: key to the ddG value of g0 edges
    :rtype: float
    """

    this_dg = 0.0
    for node_i, node_j in zip(path[:-1], path[1:]):
        try:
            this_dg += float(g0.edges[(node_i, node_j)][ddg_key])
        except KeyError:
            try:
                this_dg -= float(g0.edges[(node_j, node_i)][ddg_key])
            except KeyError:
                break
    else:
        return this_dg

    raise networkx.exception.NetworkXNoPath('Edge between {} {} not found in graph {}'.format(node_i, node_j, g0))


def ddg_to_center_ddg(ddg_graph, center, method='shortest', ddg_key='final_ddg', plot=False, no_checks=False,
                      verbosity=0):
    """ Converts pairwise ddG to ddG in respect to a reference molecule

    :param networkx.DiGraph ddg_graph: graph containing ddG (as a ddg edge data)
    :param str center: reference molecule name
    :param str method: averaging method, one of "shortest" (default), "shortest_average", "all_averages",
                       "all_weighted_averages"
    :param str ddg_key: key to the ddG data in the graph
    :param str plot: save a graph representation of the perturbations to this file
    :param bool no_checks: ignore checks and try to go on
    :param int verbosity: set verbosity level
    :rtype: dict
    """

    if plot:
        labels_data = {(i, j): '{:0.1f}'.format(data) if data is not None else 'NA'
                       for i, j, data in ddg_graph.edges(data='final_ddg')}
        pos = networkx.spring_layout(ddg_graph, k=5)
        networkx.draw_networkx(ddg_graph, with_labels=True, pos=pos)
        networkx.draw_networkx_edge_labels(ddg_graph, pos=pos, edge_labels=labels_data)
        matplotlib.pyplot.savefig(plot)

    # Remove edges without ddG data, but preserve the original graph
    ddg_graph = ddg_graph.copy()
    [ddg_graph.remove_edge(i, j) for i, j in [(i, j) for i, j, v in ddg_graph.edges(data='val') if v == 0]]

    if method == 'shortest':
        paths = networkx.single_source_shortest_path(ddg_graph.to_undirected(), center)
        ddg_to_center = {center: 0.0}
        for each_node, each_path in paths.items():
            if len(each_path) == 1:
                pass
            else:
                try:
                    ddg_to_center[each_node] = sum_path(ddg_graph, each_path, ddg_key=ddg_key)
                except networkx.exception.NetworkXNoPath as error:
                    if no_checks:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Going on, as you are '
                                            'using no_checks. This likely means that one of the legs of the '
                                            'pertubation failed. Networkx error was: {}'
                                            ''.format(each_node, center, error),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                    else:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Cannot go on. Check the '
                                            ' perturbations and your input graph. Alternatively, rerun with no_checks '
                                            'to force execution and ignore this error.'
                                            ''.format(each_node, center),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                        raise networkx.exception.NetworkXNoPath(error)

    elif method == 'shortest_average':

        ddg_to_center = {center: 0.0}

        for each_node in ddg_graph.nodes:
            ddg_to_center[each_node] = 0.0
            all_paths = list(networkx.all_shortest_paths(ddg_graph.to_undirected(), center, each_node))
            for each_path in all_paths:
                try:
                    ddg_to_center[each_node] += sum_path(ddg_graph, each_path, ddg_key=ddg_key) / len(all_paths)
                except networkx.exception.NetworkXNoPath as error:
                    if no_checks:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Going on, as you are '
                                            'using no_checks. This likely means that one of the legs of the '
                                            'pertubation failed. Networkx error was: {}'
                                            ''.format(each_node, center, error),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                    else:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Cannot go on. Check the '
                                            ' perturbations and your input graph. Alternatively, rerun with no_checks '
                                            'to force execution and ignore this error.'
                                            ''.format(each_node, center),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                        raise networkx.exception.NetworkXNoPath(error)

    elif method == 'all_averages':

        ddg_to_center = {center: 0.0}

        for each_node in ddg_graph.nodes:
            ddg_to_center[each_node] = 0.0
            all_paths = list(networkx.all_simple_paths(ddg_graph.to_undirected(), center, each_node))
            total_denominator = len(all_paths)
            for each_path in all_paths:
                try:
                    ddg_to_center[each_node] += sum_path(ddg_graph, each_path, ddg_key=ddg_key)
                except networkx.exception.NetworkXNoPath as error:
                    if no_checks:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Going on, as you are '
                                            'using no_checks. This likely means that one of the legs of the '
                                            'pertubation failed. Networkx error was: {}'
                                            ''.format(each_node, center, error),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                    else:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Cannot go on. Check the '
                                            ' perturbations and your input graph. Alternatively, rerun with no_checks '
                                            'to force execution and ignore this error.'
                                            ''.format(each_node, center),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                        raise networkx.exception.NetworkXNoPath(error)

            try:
                ddg_to_center[each_node] /= total_denominator
            except ZeroDivisionError:
                pass

    elif method == 'all_weighted_averages':

        ddg_to_center = {center: 0.0}

        for each_node in ddg_graph.nodes:
            ddg_to_center[each_node] = 0.0
            all_paths = list(networkx.all_simple_paths(ddg_graph.to_undirected(), center, each_node))
            total_denominator = sum([len(p) for p in all_paths])
            for each_path in all_paths:
                try:
                    ddg_to_center[each_node] += sum_path(ddg_graph, each_path, ddg_key=ddg_key) * len(each_path)
                except networkx.exception.NetworkXNoPath as error:
                    if no_checks:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Going on, as you are '
                                            'using no_checks. This likely means that one of the legs of the '
                                            'pertubation failed. Networkx error was: {}'
                                            ''.format(each_node, center, error),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                    else:
                        os_util.local_print('Could not find a path to connect nodes {} and {}. Cannot go on. Check the '
                                            ' perturbations and your input graph. Alternatively, rerun with no_checks '
                                            'to force execution and ignore this error.'
                                            ''.format(each_node, center),
                                            msg_verbosity=os_util.verbosity_level.error,
                                            current_verbosity=verbosity)
                        raise networkx.exception.NetworkXNoPath(error)

            try:
                ddg_to_center[each_node] /= total_denominator
            except ZeroDivisionError:
                pass

    else:
        os_util.local_print('Unknown method "{}" to convert pairwise ΔΔG to ΔΔG from {}. Please, use one of '
                            '"shortest" "shortest_average", "all_averages".'
                            ''.format(method, center))
        raise ValueError()

    return ddg_to_center


def dummy_hysteresis(g0):
    return_data = {}
    for path in networkx.algorithms.cycles.simple_cycles(g0.to_undirected().to_directed()):
        if len(path) < 3:
            continue

        this_cycle_dg = sum_path(g0, path)

        try:
            this_cycle_dg -= float(g0.edges[(path[0], path[-1])]['ddg'])
        except KeyError:
            this_cycle_dg += float(g0.edges[(path[-1], path[0])]['ddg'])

        return_data[path] = this_cycle_dg

    return return_data


if __name__ == '__main__':
    import process_user_input

    Parser = argparse.ArgumentParser(description='Read and analyze FEP data, preparing plots')
    Parser.add_argument('--input', default=None, help='Loads FEP data from this file')
    Parser.add_argument('--center_molecule', default=None, type=str,
                        help='Use this molecule as reference to convert pairwise \u0394\u0394G to \u0394\u0394G in '
                             'refecence to a molecule. Default: do not perform network analysis and do not calculate '
                             '\u0394\u0394G to a reference.')
    Parser.add_argument('-T', '--temperature', default=None, type=float,
                        help='Absolute temperature of the sampling. Default: read from input. Only use this option to '
                             'force a different temperature.')
    Parser.add_argument('--units', default=None, type=str, choices=['kJmol', 'kcal', 'kBT'],
                        help='Use this unit in the output (Default: kJ/mol)')
    Parser.add_argument('--first_frame', type=int, default=None,
                        help='First frame (ps) to read from trajectory, in ps. Default: 0 ps')
    Parser.add_argument('--last_frame', type=int, default=None,
                        help='Last frame (ps) to read from trajectory, in ps. Default: -1 = read all')
    Parser.add_argument('--convergence', default=None, type=int,
                        help='Calculates the \u0394\u0394G estimate and error for successive truncated trajectories '
                             'using this step (ps). (Default: autodetect)')
    Parser.add_argument('--calculate_tau_c', default=None, type=str,
                        help='Pre-analyze frames and calculate \u03C4c. (Default: no)')
    Parser.add_argument('--detect_equilibration', action='store_const', const=True,
                        help='Autodetect end of equilibration in the sample. (Default: off)')
    Parser.add_argument('--estimators', default=None, type=str,
                        help='Use these estimators to calculate relative free energy. Default: mbar.')
    Parser.add_argument('--ddg_systems', default=None, type=str,
                        help='Use these systems to calculate \u0394\u0394G. Default: read from input. Currently, this '
                             'will be "protein", "water" and \u0394\u0394G = \u0394\u0394G(protein) - '
                             '\u0394\u0394G(water)')
    Parser.add_argument('--analysis_types', default=None, type=str,
                        help='Run these analysis. (Default: run all)')
    Parser.add_argument('--center_ddg_method', default=None, choices=["shortest", "shortest_average", "all_averages",
                                                                      "all_weighted_averages"],
                        help='Use this method to calculate \u0394\u0394G to the center molecules.')

    output_args = Parser.add_argument_group('Options controlling the data output and saving')
    output_args.add_argument('--output_uncompress_directory', default=None, type=str,
                             help='Save uncompressed data to this directory (Default: use a temporary location, will '
                                  'be removed on script exit)')
    output_args.add_argument('--output_plot_directory', default=None, type=str,
                             help='Save plots to this directory (Default: save to PWD, or '
                                  'to output_uncompress_directory, if used, or to the input data dir, if a dir is '
                                  'used)')
    # TODO: save pertubation data to a text file
    # output_args.add_argument('--output_perturbations_data', default=None, type=str,
    #                          help='Save free energy data of each perturbation to this file')
    output_args.add_argument('--output_ddg_to_center', default=None, type=str,
                             help='Save \u0394\u0394G to center molecule (center_molecule)')

    process_user_input.add_argparse_global_args(Parser)
    arguments = process_user_input.read_options(Parser, unpack_section='analyze_results')

    all_analysis = ['convergence', 'overlap_matrix', 'neighbor_ddg', 'replica_exchange', 'all']
    arguments.analysis_types = os_util.detect_type(arguments.analysis_types, test_for_list=True)
    if not arguments.analysis_types:
        arguments.analysis_types = ['all']
    else:
        if isinstance(arguments.analysis_types, str):
            arguments.analysis_types = [arguments.analysis_types]
        for each_analysis in arguments.analysis_types:
            if each_analysis not in all_analysis:
                os_util.local_print('Analysis {} not known. analysis_types is {}.'
                                    ''.format(each_analysis, arguments.analysis_types),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
                raise SystemExit(1)
    if 'all' not in arguments.analysis_types:
        os_util.local_print('The following analysis will be run: {}.'.format(', '.join(arguments.analysis_types)),
                            msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)
    else:
        os_util.local_print('All available analysis will be run',
                            msg_verbosity=os_util.verbosity_level.default, current_verbosity=arguments.verbose)

    try:
        kB = formatted_energy_units[arguments.units].kB
        os_util.local_print('Using units {}'.format(formatted_energy_units[arguments.units].text),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)
    except KeyError:
        os_util.local_print('Could not understand unit {}. Please select either kJmol, kcal or kBT'
                            ''.format(arguments.units),
                            msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
        raise SystemExit(1)

    estimators_data = {}
    arguments.estimators = os_util.detect_type(arguments.estimators, test_for_list=True)
    if isinstance(arguments.estimators, str):
        arguments.estimators = [arguments.estimators]
    for estimator in arguments.estimators:
        if estimator == 'mbar':
            estimators_data[estimator] = MBAR
        elif estimator == 'bar':
            estimators_data[estimator] = BAR
        else:
            os_util.local_print('Estimator {} not implemented. Please, select between "mbar" and "bar"'
                                ''.format(estimator),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=arguments.verbose)
            raise SystemExit(1)
    os_util.local_print('Using the following estimators: {}'.format(estimators_data.keys()),
                        msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)

    if not os.path.isdir(arguments.input):
        os_util.local_print('Loading data from {}. Interpreting it as a tgz file.'.format(arguments.input),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)

        # Extract the input file to a temporary dir. Do no use tarfile for security reasons
        #  (see https://bugs.python.org/issue21109)
        if arguments.output_uncompress_directory is None:
            tempdir = tempfile.TemporaryDirectory()
        else:
            tempdir = all_classes.Namespace({'name': arguments.output_uncompress_directory, 'cleanup': lambda: None})
            os_util.local_print('Uncompressing data to {}. This directory will not be removed after the run.'
                                ''.format(tempdir.name),
                                msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)
            os_util.makedir(tempdir.name, verbosity=arguments.verbose)

        os_util.local_print('Decompressing {} to {}'.format(arguments.input, tempdir.name),
                            current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.info)
        subprocess.check_output(['tar', '-xzf', arguments.input, '--directory', tempdir.name])
        os_util.local_print('Done extracting. Starting analysis.'.format(arguments.input),
                            current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.info)
        data_dir = tempdir.name
    else:
        data_dir = arguments.input
        arguments.output_plot_directory = arguments.input
        os_util.local_print('Loading data from {}, interpreting it as a directory.'.format(arguments.input),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=arguments.verbose)
        if arguments.output_uncompress_directory is not None and not arguments.no_checks:
            os_util.local_print('Because {} is a directory, no data will be uncompressed, so '
                                'output_uncompress_directory ({}) will be ignored.'
                                ''.format(arguments.input, arguments.output_uncompress_directory),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)

    # Try to find files in extracted dir
    found_systems = []
    pertubations_unk_data = {}
    saved_data = {}

    os_util.local_print('Searching a progress file in {}'.format(data_dir),
                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)
    # First, try to load a progress file from the data dir to use the perturbation map from this progress file
    for each_entry in os.listdir(data_dir):
        if os.path.splitext(each_entry)[-1] == '.pkl':
            os_util.local_print('Testing pickle file {}'.format(each_entry),
                                msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)
            try:
                saved_data = savestate_util.SavableState(input_file=os.path.join(data_dir, each_entry))
            except (ValueError, EOFError):
                # Safeguard if the file is not and savestate pickle file
                pass
            else:
                os_util.local_print('{} was identified as a saved state'.format(each_entry),
                                    msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)

    if not saved_data:
        if arguments.no_checks:
            os_util.local_print('Could not find a progress file from your data {}. Because you are running with '
                                'no_checks, I will go on and try to reconstruct a map from your data. Be aware that '
                                'this may fail.'
                                ''.format(arguments.input), msg_verbosity=os_util.verbosity_level.error,
                                current_verbosity=arguments.verbose)
            saved_data['no_progress'] = True
        else:
            os_util.local_print('Could not find a progress file from your data {}, so I cannot read a map. You can '
                                'retry using --no_checks to force the building of a map from the perturbations found '
                                'on your data.'
                                ''.format(arguments.input), msg_verbosity=os_util.verbosity_level.error,
                                current_verbosity=arguments.verbose)
            raise SystemExit(-1)

    for each_entry in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, each_entry)):
            continue

        os_util.local_print('Processing directory {}'.format(each_entry),
                            msg_verbosity=os_util.verbosity_level.debug, current_verbosity=arguments.verbose)

        for system in os.listdir(os.path.join(data_dir, each_entry)):
            # This should be a pickle file containing a dict to assemble the u_nk matrix
            pkl_file = os.path.join(data_dir, each_entry, system, 'md', 'rerun', 'unk_matrix.pkl')
            if os.path.isfile(pkl_file):
                pertubations_unk_data.setdefault(each_entry, {}).__setitem__(system, {'pkl': pkl_file})
                if system not in found_systems:
                    found_systems.append(system)
            else:
                os_util.local_print('Could not find result pkl file for {} run {}. Estimatives and analysis for this '
                                    'system will not be run.',
                                    current_verbosity=arguments.verbose,
                                    msg_verbosity=os_util.verbosity_level.warning)

            # And this is a Gromacs run log to possibly extract the replica exchange info. Note that if no
            log_file = os.path.join(data_dir, each_entry, system, 'md', 'lambda0', 'lambda.log')
            if os.path.isfile(log_file):
                try:
                    pertubations_unk_data[each_entry][system]['log'] = log_file
                except KeyError:
                    os_util.local_print('A log file for {} run {} was found, but no result pkl file was found. No '
                                        'analysis will be performed.',
                                        current_verbosity=arguments.verbose,
                                        msg_verbosity=os_util.verbosity_level.warning)
                else:
                    if 'perturbation_map' not in saved_data:
                        if arguments.no_checks:
                            if 'no_progress' not in saved_data:
                                os_util.local_print('No perturbation map found in {}. Because you are running with '
                                                    'no_checks, I will force the reconstruction of the map from the '
                                                    'perturbation data. This may fail.'.format(arguments.input),
                                                    current_verbosity=arguments.verbose,
                                                    msg_verbosity=os_util.verbosity_level.error)
                            saved_data['perturbation_map'] = networkx.DiGraph()
                        else:
                            os_util.local_print('No perturbation map found in {}. Cannot go on. You can run with '
                                                'no_checks to try force the reconstruction of the map from the '
                                                'perturbation data. But this should not happen'
                                                ''.format(arguments.input),
                                                current_verbosity=arguments.verbose,
                                                msg_verbosity=os_util.verbosity_level.error)
                            raise SystemExit(-1)

            else:
                os_util.local_print('No log file for {} run {} was found. Replica exchange analysis will not be '
                                    'performed.', current_verbosity=arguments.verbose,
                                    msg_verbosity=os_util.verbosity_level.warning)

    if not saved_data:
        if arguments.no_checks:
            os_util.local_print('Failed to load a progress file from {}. Because you are using no_checks, I will try '
                                'to construct a map from the perturbations data. But this should not happen.'
                                ''.format(arguments.input),
                                current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.error)
            saved_data['perturbation_map'] = networkx.DiGraph()
        else:
            os_util.local_print('Failed to load a progress file from {}. Because you are using no_checks, Cannot go '
                                'on. You can run with no_checks to try force the reconstruction of the map from the '
                                'perturbation data.'
                                ''.format(arguments.input),
                                current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(-1)

    if arguments.ddg_systems is not None:
        found_systems = os_util.detect_type(arguments.ddg_systems, test_for_list=True)
    else:
        # This will ensure the order of found_systems. Later on, found_system[0] - found_system[1] will be used to
        #  calculate the final ddG in each edge. First, try to read from saved data, if it fails fallback to default
        if 'systems' in saved_data:
            if set(found_systems) != set(saved_data['systems']):
                if arguments.no_checks:
                    os_util.local_print('The leg systems saved on your progress file are {} while the systems read '
                                        'from the input file {} are {}. Because you are using no_checks, I will try '
                                        'to go on.'.format(saved_data['systems'], arguments.input, found_systems),
                                        current_verbosity=arguments.verbose,
                                        msg_verbosity=os_util.verbosity_level.error)
                    if set(found_systems) == {'protein', 'water'}:
                        found_systems = ['protein', 'water']
                else:
                    os_util.local_print('The leg systems saved on your progress file are {} while the systems read '
                                        'from the input file {} are {}. Data mismatch, cannot continue.'
                                        ''.format(saved_data['systems'], arguments.input, found_systems),
                                        current_verbosity=arguments.verbose,
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise SystemExit(-1)
            else:
                found_systems = saved_data['systems']
        elif set(found_systems) == {'protein', 'water'}:
            found_systems = ['protein', 'water']
        else:
            os_util.local_print('Could not parse leg systems data from progress file neither guess it from {}. Cannot '
                                'continue'.format(found_systems),
                                current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(-1)

    # NOTE: in the future, more than two systems will be allowed
    if len(found_systems) == 2:
        os_util.local_print('Calculating \u0394\u0394G as {} - {}'.format(*found_systems),
                            current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.info)
        header_str = '=' * 50
        header_str += '\n{:^30} {:^9} {:^9}'.format("Perturbation", *found_systems)
        os_util.local_print(header_str, current_verbosity=arguments.verbose,
                            msg_verbosity=os_util.verbosity_level.default)
    elif arguments.no_checks:
        os_util.local_print('Leg data corrupt. I read {} as legs from your data. Because your are using no_checks, I '
                            'am going on.'.format(arguments.input),
                            current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.error)
    else:
        os_util.local_print('Leg data corrupt. Cannot continue.'.format(arguments.input),
                            current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.error)
        raise SystemExit(-1)

    # Sets and create the output dir
    output_directory = os.getcwd()
    if arguments.output_plot_directory is not None:
        output_directory = arguments.output_plot_directory
    elif arguments.output_uncompress_directory is not None:
        output_directory = arguments.output_uncompress_directory
    os_util.makedir(output_directory, verbosity=arguments.verbose)

    arguments_data_holder = OrderedDict()
    for each_perturbation, each_data in pertubations_unk_data.items():
        key = [(mol_i, mol_j) for (mol_i, mol_j) in saved_data['perturbation_map'].edges
               if '{}-{}'.format(mol_i, mol_j) == each_perturbation]
        if not key:
            if arguments.no_checks:
                if not 'no_progress' in saved_data:
                    os_util.local_print('Failed to find a egde for {} in the perturbation graph. Because you are '
                                        'using no_checks, I will try to create a new edge on the graph and try to '
                                        'move on.'.format(each_perturbation), current_verbosity=arguments.verbose,
                                        msg_verbosity=os_util.verbosity_level.error)
                key = each_perturbation.split('-')
                if len(key) != 2:
                    os_util.local_print('Failed to convert perturbation name {} in graph edges. Cannot go on.'
                                        ''.format(each_perturbation), current_verbosity=arguments.verbose,
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise SystemExit(-1)
                else:
                    saved_data['perturbation_map'].add_edge(*key)
                    os_util.local_print('Adding edge {} to the graph'.format(key), current_verbosity=arguments.verbose,
                                        msg_verbosity=os_util.verbosity_level.warning)

            else:
                os_util.local_print('Failed to find an egde for {} in the perturbation graph. Cannot go on. You can '
                                    'run with no_checks to try force the reconstruction of the missing the edges '
                                    'from the perturbation data. This should not happen, though.'
                                    ''.format(each_perturbation),
                                    current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.error)
                raise SystemExit(-1)
        else:
            key = key[0]

        for system, files in each_data.items():
            this_dir = os.path.join(output_directory, '{}-{}'.format(*key), system)
            os_util.makedir(this_dir, parents=True, verbosity=arguments.verbose)

            with open(files['pkl'], 'rb') as fh:
                try:
                    perturbation_data = pickle.load(fh)
                except FileNotFoundError:
                    os_util.local_print('Input file {} not found. Please, check your input option or argument.'
                                        ''.format(files['pkl']),
                                        msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)
                    raise SystemExit(1)
                except BaseException:
                    os_util.local_print('Failed to read input file {}. Please, check the input file integrity.'
                                        ''.format(files['pkl']),
                                        msg_verbosity=os_util.verbosity_level.error,
                                        current_verbosity=arguments.verbose)
                    raise SystemExit(1)

            kwargs = {'perturbation_input': files['pkl'], 'perturbation_data': perturbation_data,
                      'gromacs_log': files.setdefault('log', None), 'estimators_data': estimators_data,
                      'analysis_types': arguments.analysis_types, 'convergence_analysis_step': arguments.convergence,
                      'start_time': arguments.first_frame, 'calculate_tau_c': arguments.calculate_tau_c,
                      'detect_equilibration': arguments.detect_equilibration, 'temperature': arguments.temperature,
                      'units': arguments.units, 'plot': arguments.plot, 'output_directory': this_dir,
                      'no_checks': arguments.no_checks, 'verbosity': arguments.verbose}
            arguments_data_holder[(tuple(key), system)] = kwargs

    os_util.local_print('Starting {} \u0394\u0394G estimations using {} threads'
                        ''.format(len(arguments_data_holder), arguments.threads), current_verbosity=arguments.verbose,
                        msg_verbosity=os_util.verbosity_level.info)

    if arguments.threads == -1:
        ddg_data_result = [analyze_perturbation(**each_kwargs) for each_kwargs in arguments_data_holder.values()]
    else:
        with multiprocessing.Pool(arguments.threads) as threads:
            ddg_data_result = os_util.starmap_unpack(analyze_perturbation, threads,
                                                     kwargs_iter=arguments_data_holder.values())

    for (key, system), ddg_data in zip(arguments_data_holder.keys(), ddg_data_result):
        saved_data['perturbation_map'].edges[key][system] = ddg_data

    for key in set([key for key, system in arguments_data_holder.keys()]):
        each_perturbation = '{}\u2192{}'.format(*key)

        if set(saved_data['perturbation_map'].edges[key]).intersection(set(found_systems)) != set(found_systems):
            os_util.local_print('Skipping \u0394\u0394G calculation for {} because only {} systems are present (and '
                                'you selected {} to analyze)'
                                ''.format(each_perturbation, saved_data['perturbation_map'].edges[key].keys(),
                                          found_systems),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)
            continue

        # Use MBAR, falling back to BAR to do the graph and global analysis
        if 'mbar' in saved_data['perturbation_map'].edges[key][found_systems[0]]:
            estimator_to_use = 'mbar'
        elif 'bar' in saved_data['perturbation_map'].edges[key][found_systems[0]]:
            estimator_to_use = 'bar'
        else:
            os_util.local_print('You are not using mbar nor bar estimators. I will not plot any graph.',
                                current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.warning)
            estimator_to_use = None

        if estimator_to_use == 'mbar':
            this_edge = saved_data['perturbation_map'].edges[key]

            # ddG will be ddG(A) - ddG(B). This will also ensure the last data from convergence analysis (the one
            # obtained by using [first_frame:-1] data in the u_nk table) will be used
            system_a = this_edge[found_systems[0]][estimator_to_use]['forward'][-1]
            system_b = this_edge[found_systems[1]][estimator_to_use]['forward'][-1]

            # Test the beta are equal, so the values can be combined
            if this_edge[found_systems[0]]['beta'] == this_edge[found_systems[1]]['beta']:
                beta = this_edge[found_systems[0]]['beta']
                value1 = '{:0.1f}\u00B1{:0.1f} {}'.format(system_a['ddg'] / beta, system_a['error'] / beta,
                                                          formatted_energy_units[arguments.units].text)
                value2 = '{:0.1f}\u00B1{:0.1f} {}'.format(system_b['ddg'] / beta, system_b['error'] / beta,
                                                          formatted_energy_units[arguments.units].text)
                value3 = '\u0394\u0394G = {:0.1f} {}'.format((system_a['ddg'] - system_b['ddg']) / beta,
                                                             formatted_energy_units[arguments.units].text)

                os_util.local_print('{:^25} {:^20} {:^20} {:^20}'
                                    ''.format(each_perturbation, value1, value2, value3),
                                    current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.default)
                this_edge['final_ddg'] = (system_a['ddg'] - system_b['ddg']) / beta
                this_edge['final_d_ddg'] = (system_a['error'] / system_a['ddg']
                                            + system_b['error'] / system_b['ddg']) / beta
            else:
                os_util.local_print('Temperature or units are not the same for {} and {} systems of perturbation {}. I '
                                    'will not be able to combine their \u0394\u0394G values. Make sure this is what '
                                    'you really want.'.format(*found_systems, each_perturbation),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)
                beta_a = this_edge[found_systems[0]]['beta']
                beta_b = this_edge[found_systems[1]]['beta']

                value1 = '{:0.1f}\u00B1{:0.1f} {} @ {:0.2f} K' \
                         ''.format(system_a['ddg'] / beta_a, system_a['error'] / beta_a,
                                   formatted_energy_units[this_edge[found_systems[0]]['units']].text,
                                   this_edge[found_systems[0]]['temperature'])
                value2 = '{:0.1f}\u00B1{:0.1f} {} @ {:0.2f} K' \
                         ''.format(system_b['ddg'] / beta_b, system_b['error'] / beta_b,
                                   formatted_energy_units[this_edge[found_systems[1]]['units']].text,
                                   this_edge[found_systems[1]]['temperature'])
                os_util.local_print('{:^25} {:^20} {:^20}'
                                    ''.format(each_perturbation, value1, value2),
                                    current_verbosity=arguments.verbose, msg_verbosity=os_util.verbosity_level.default)
                this_edge['final_ddg'] = None
                this_edge['final_d_ddg'] = None

        else:
            os_util.local_print('{:^25} {:^20} {:^20}'.format(each_perturbation, "Done", "Done"),
                                current_verbosity=arguments.verbose,
                                msg_verbosity=os_util.verbosity_level.default)

    saved_data['perturbation_map_{}'.format(time.strftime('%H%M%S_%d%m%Y'))] = saved_data['perturbation_map'].copy()

    if arguments.progress_file is not None:
        if 'no_progress' in saved_data:
            progress_file = savestate_util.SavableState(arguments.progress_file)
            progress_file.update(saved_data)
            del progress_file['no_progress']
            progress_file.save_data()
        else:
            saved_data.data_file = arguments.progress_file
            saved_data.save_data()
    else:
        os_util.local_print('You did not supply a progress_file, so I wont save data to a portable format.',
                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=arguments.verbose)

    if arguments.output_ddg_to_center is not False and \
            (arguments.output_ddg_to_center or (arguments.output_ddg_to_center is None
                                                and (arguments.center_molecule is not None
                                                     or 'center_molecule' in saved_data))):
        if arguments.output_ddg_to_center is None:
            arguments.output_ddg_to_center = 'ddg_to_center.csv'

        # FIXME: add an error estimate
        ddg_to_center = ddg_to_center_ddg(saved_data['perturbation_map'], center=arguments.center_molecule,
                                          method=arguments.center_ddg_method)
        csv_data = pandas.DataFrame({'Name': list(ddg_to_center.keys()),
                                     'ddg': list(ddg_to_center.values()),
                                     'error': [0.0] * len(ddg_to_center)}).set_index('Name')
        csv_data.to_csv(os.path.join(output_directory, arguments.output_ddg_to_center))