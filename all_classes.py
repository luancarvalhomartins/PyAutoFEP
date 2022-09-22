#! /usr/bin/env python3
#
#  all_classes.py
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
import os.path
import tempfile

import networkx
import numpy
from collections import OrderedDict
import os_util
import re
from copy import deepcopy


def namedlist(typename, field_names, defaults=(), types=()):
    """ Returns a new subclass of list with named fields. Recipe by Sergey Shepelev at
    https://code.activestate.com/recipes/578041-namedlist/ altered by Luan Carvalho Martins

    :param str typename: name of the class
    :param list field_names: list of fields
    :param list  defaults: right-to-left list of defaults
    :param list types: list of types
    :rtype: function
    """
    if isinstance(field_names, str):
        field_names = field_names.split()
    fields_len = len(field_names)

    if types:
        if len(types) != fields_len:
            raise TypeError("Expected {} types, got {}".format(len(types), fields_len))

    class ResultType(list):
        __slots__ = ('_parent', )
        _fields = field_names

        def _fixed_length_error(*args, **kwargs):
            raise TypeError(u"Named list has fixed length")

        append = _fixed_length_error
        insert = _fixed_length_error
        pop = _fixed_length_error
        remove = _fixed_length_error

        def sort(self):
            raise TypeError(u"Sorting named list in place would corrupt field accessors. Use sorted(x)")

        def _replace(self, **kwargs):
            values = map(kwargs.pop, field_names, self)
            if kwargs:
                raise TypeError(u"Unexpected field names: {s!r}".format(kwargs.keys()))

            if len(values) != fields_len:
                raise TypeError(u"Expected {e} arguments, got {n}".format(
                    e=fields_len, n=len(values)))

            return ResultType(*values)

        def __repr__(self):
            items_repr = ", ".join("{name}={value!r}".format(name=name, value=value)
                                   for name, value in zip(field_names, self))
            return "{typename}({items})".format(typename=typename, items=items_repr)

        def __init__(self, *args, **kwargs):
            self._parent = None
            self.extend([None] * fields_len)
            if len(args) + len(kwargs) + len(defaults) < len(field_names):
                raise TypeError("__init__() missing {} required arguments"
                                "".format(len(field_names) - (len(args) + len(kwargs) + len(defaults))))
            elif len(args) + len(kwargs) > len(field_names):
                raise TypeError("__init__() takes {} positional arguments but {} were given"
                                "".format(len(field_names), len(args) + len(kwargs)))

            [setattr(self, k, v) for k, v in zip(reversed(field_names), defaults)]
            [setattr(self, k, v) for k, v in zip(field_names, args)]
            [setattr(self, k, v) for k, v in kwargs.items()]
            for v, t in zip(self, types):
                if t is not False and type(v) != t:
                    raise TypeError("Unsupported operand type(s) __init__(): '{}' and '{}'".format(type(v), t))

        def __deepcopy__(self, memodict={}):
            return self.__class__(**{k: v for k, v in zip(field_names, self)})

        def keys(self):
            return self._fields[:]

    ResultType.__name__ = typename

    for i, name in enumerate(field_names):
        fget = eval("lambda self: self[{0:d}]".format(i))
        fset = eval("lambda self, value: self.__setitem__({0:d}, value)".format(i))
        setattr(ResultType, name, property(fget, fset))

    return ResultType


class Namespace(OrderedDict):
    """ A simple ordered namespace class """

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __str__(self):
        items_data = []
        for k, v in self.items():
            if len(str(v)) < 50:
                items_data.append('{}={}'.format(k, v))
            else:
                try:
                    items_data.append('<{} object at {} with len {}>'.format(v.__class__, hex(id(v)), len(v)))
                except (TypeError, UnboundLocalError):
                    items_data.append('<{} object at {}>'.format(v.__class__, hex(id(v))))

        this_str = 'Namespace({})'.format(', '.join(items_data))
        return this_str

    def __repr__(self):
        return self.__str__()


class AntSolver:
    """ Stores a network and use ACO to find a solution minimizing AntSolver.calculate_network_cost """

    class Solution:
        """ Stores solution data """

        def __init__(self, network_graph, solution_cost, pheromone_multiplier=0):
            """ Create a new instance of Solution

            :param networkx.Graph network_graph: solution network
            :param float solution_cost: solution cost
            :param float pheromone_multiplier: pheromone multiplier used in run
            :rtype: float
            """

            self.graph = network_graph
            self.cost = float(solution_cost)
            self.pheromone_multiplier = pheromone_multiplier

    @property
    def cost_list(self):
        """ List of all stored costs

        :rtype: list
        """
        if len(self._cost_list) != len(self.solutions):
            self._cost_list = [each_cost.cost for each_cost in self.solutions]
        return self._cost_list

    @property
    def mean_cost(self):
        """ Mean value of all stored costs

        :rtype: float
        """
        return float(sum(self.cost_list)) / len(self.solutions)

    @property
    def best_solution(self):
        """ Return the best solution found or False if no solution was stored

        :rtype: AntSolver.Solution
        """
        if len(self.cost_list) == 0:
            return False
        else:
            this_index = self.cost_list.index(min(self.cost_list))
            return self.solutions[this_index]

    @property
    def pheromone_intensity(self):
        """ Intensity of pheromone to be deposited

        :rtype: float
        """
        return self._pheromone_intensity

    @pheromone_intensity.setter
    def pheromone_intensity(self, value):
        if value >= 0.0:
            self._pheromone_intensity = value
        else:
            raise ValueError('pheromone_intensity must be non-negative')

    def calculate_network_cost(self, network_graph, decompose=False):
        """ Calculates the network cost

        :param networkx.Graph network_graph: network te be calculated
        :param bool decompose: return the components to the final value
        :rtype: network cost
        """

        if self.length_exponent:
            all_lengths = dict(networkx.all_pairs_shortest_path_length(network_graph))
            lengths_matrix = numpy.array([j for i in all_lengths.values() for j in list(i.values())], dtype=float)
            length_cost = numpy.sum(lengths_matrix ** self.length_exponent)
        else:
            length_cost = 0
        if self.degree_multiplier != 0:
            degree_cost = numpy.array(list(dict(networkx.degree(network_graph)).values()))
            degree_cost = ((degree_cost - self.degree_target) ** self.degree_exponent).sum() * self.degree_multiplier
        else:
            degree_cost = 0

        perturbation_cost = [i[2] ** self.perturbation_exponent for i in network_graph.edges(data='cost')]
        perturbation_cost = self.perturbation_multiplier * numpy.array(perturbation_cost).sum()

        if decompose is False:
            return length_cost + perturbation_cost + degree_cost
        else:
            return {'total': length_cost + perturbation_cost + degree_cost,
                    'length': length_cost,
                    'perturbation': perturbation_cost,
                    'degree': degree_cost}

    def run_multi_ants(self, runs, **kwargs):
        """ Run several ants, pass all kwargs to run_ant

        :param int runs: run this many ants
        :rtype: list
        """

        if runs <= 0:
            raise ValueError('runs must be positive')

        return [self.run_ant(**kwargs) for _ in range(runs)]

    def run_ant(self, pheromone_intensity=None, algorithm=None, verbosity=0):
        """ Do the ant work

        :param float pheromone_intensity: pheromone intensity multiplier (None: use default; 0: do not deposit)
        :param str algorithm: select algorithm: "classic" (original ACO) or "modified" (modified version to generate
                              optimized maps), default: use value from object
        :param int verbosity: set verbosity level
        :rtype: self.Solution
        """

        # Regenerate random seed (necessary in threaded execution)
        rand_gen = numpy.random.default_rng()

        if not algorithm:
            algorithm = self.algorithm

        if algorithm == 'modified':
            worker_network = self.complete_network_undirect.copy()
            edge_list = list(worker_network.edges())

            # Prepare the probabilities of removing each edge
            desirability_list = numpy.array([each_edge[2]['desirability'] ** self.alpha
                                             * each_edge[2]['cost'] ** self.beta
                                             for each_edge in worker_network.edges(data=True)])

            if self.unbound_runs != -1 and len(self.solutions) > self.unbound_runs:
                # Do not select static edges
                indices_larger = numpy.zeros(desirability_list.shape, dtype=bool)
                indices_larger[numpy.argpartition(desirability_list, -self.minimum_edges)[-self.minimum_edges:]] = True
                indices_above_threshold = desirability_list > self.permanent_edge_threshold
                desirability_list[indices_larger & indices_above_threshold] = 1e9

            desirability_list = 1 / desirability_list

            n_min_edges_per_node, n_path_long, n_is_connected = 0, 0, 0

            while True:

                # normalize desirability_list (required by rand_gen.choice)
                desirability_list = desirability_list / desirability_list.sum()

                # Select a edge to remove
                selected_edge = edge_list[rand_gen.choice(range(len(edge_list)), p=desirability_list)]

                # Test if removed edge is valid (ie: network is still connected and not any path > path_threshold)
                temp_network = worker_network.copy()
                temp_network.remove_edge(*selected_edge)
                if self.min_edges_per_node >= 2 \
                        and any((v < self.min_edges_per_node for k, v in temp_network.degree)):
                    # at least one of the network edges has less then min_edges_per_node edges
                    n_min_edges_per_node += 1
                    pass
                elif max([j for i in dict(networkx.all_pairs_shortest_path_length(temp_network)).values()
                          for j in i.values()]) > self.path_threshold:
                    # at least one of the network paths would be longer than threshold
                    n_path_long += 1
                    pass
                elif not networkx.is_connected(temp_network.to_undirected(as_view=True)):
                    # network would be disconnected
                    n_is_connected += 1
                    pass
                else:
                    worker_network.remove_edge(*selected_edge)

                desirability_list = numpy.delete(desirability_list, edge_list.index(selected_edge))
                del edge_list[edge_list.index(selected_edge)]
                if len(edge_list) == self.minimum_edges:
                    break
                if self.extra_edge_beta > 0 \
                        and rand_gen.exponential(self.extra_edge_beta) > len(edge_list) - self.minimum_edges:
                    break

            os_util.local_print('Modified ACO statistics: n_min_edges_per_node {}, n_path_long {}, n_is_connected {}'
                                ''.format(n_min_edges_per_node, n_path_long, n_is_connected),
                                msg_verbosity=os_util.verbosity_level.debug)
            this_cost = self.calculate_network_cost(worker_network)

        elif algorithm == 'classic':

            # Setup worker network
            worker_network = deepcopy(self.complete_network_undirect)
            for e in worker_network.edges:
                worker_network.edges[e]['visited'] = False
            for n in worker_network.nodes:
                worker_network.nodes[n]['visited'] = False

            first_node = list(worker_network.nodes)[rand_gen.integers(len(worker_network.nodes))]
            current_node = first_node

            while True:

                worker_network.nodes[current_node]['visited'] = True

                # Find near, unvisited nodes and edges
                near_nodes = set([node for node in worker_network.adj[current_node]
                                  if not worker_network.nodes[node]['visited']])
                near_edges = [each_edge for each_edge in worker_network.edges(current_node, data=True)
                              if not each_edge[2]['visited']
                              and near_nodes.intersection(set(each_edge[:2]))]

                if len(near_edges) == 0:
                    # All nodes visited, finish
                    worker_network.edges[(current_node, first_node)]['visited'] = True
                    results_map = deepcopy(self.complete_network_undirect)
                    results_map.remove_edges_from(self.complete_network_undirect.edges)
                    results_map.add_edges_from([e for e in worker_network.edges.data() if e[2]['visited']])
                    worker_network = results_map

                    break

                desirability_list = numpy.array([e[2]['desirability'] for e in near_edges])

                # Normalize desirability_list (required by rand_gen.choice)
                desirability_list = desirability_list / desirability_list.sum()

                # Select an edge to visit
                selected_edge = near_edges[rand_gen.choice(range(len(near_edges)), p=desirability_list)]

                # Move
                worker_network.edges[selected_edge[:2]]['visited'] = True
                current_node = selected_edge[0] if selected_edge[0] != current_node else selected_edge[1]

            this_cost = self.calculate_network_cost(worker_network)

        else:
            os_util.local_print('Could not understand algorithm selection {}. Please, choose either "classic" or '
                                '"modified"'.format(algorithm), msg_verbosity=os_util.verbosity_level.error)
            raise ValueError("Unknown algorithm {}".format(algorithm))

        if pheromone_intensity != 0.0:
            try:
                last_n_elements = self.cost_list[-self.sliding_window:]
                pheromone_multiplier = (sum(last_n_elements) / len(last_n_elements)) ** self.pheromone_exponent
                pheromone_multiplier /= this_cost ** self.pheromone_exponent
            except ZeroDivisionError:
                # First run
                pheromone_multiplier = 1

            return self.Solution(worker_network, this_cost, pheromone_multiplier)
        else:
            return self.Solution(worker_network, this_cost)

    def deposit_pheromone(self, pheromone_multiplier, solution_graph):
        """ Deposit a pheromone trail

        :param float pheromone_multiplier: pheromone intensity multiplier
        :param networkx.Graph solution_graph: solution graph use by ant
        :rtype: True
        """

        for each_edge in solution_graph.edges:
            this_pheromone = min(self.pheromone_intensity * pheromone_multiplier, self.max_pheromone_deposited)
            self.complete_network_undirect[each_edge[0]][each_edge[1]]['desirability'] += this_pheromone
        return True

    def evaporate_pheromone(self, evaporating_rate=None):
        """ Reduces the pheromone trail

        :param float evaporating_rate: evaporate this much pheromone (Default: use stored)
        :rtype: True
        """
        if evaporating_rate is None:
            evaporating_rate = self.evaporating_rate

        for _, _, each_data in self.complete_network_undirect.edges(data=True):
            each_data['desirability'] *= 1.0 - evaporating_rate
            if each_data['desirability'] < self.min_edge_desirability:
                each_data['desirability'] = self.min_edge_desirability

    def __init__(self, network_graph, alpha=1, beta=1, path_threshold=-1, perturbation_multiplier=20,
                 perturbation_exponent=4.0, length_exponent=0.0, degree_multiplier=0.0, degree_target=None,
                 degree_exponent=2.0, pheromone_intensity=0.0, pheromone_exponent=2, max_pheromone_deposited=-1,
                 sliding_window=0, evaporating_rate=0.02, min_edge_desirability=0.1, min_unbound=-1,
                 permanent_edge_threshold=-1, extra_edge_beta=2, min_edges_per_node=2, algorithm='modified'):
        """ Creates a new instance of AntSolver

        :param networkx.Graph network_graph: graph to run ant on
        :param float alpha: pheromone biasing exponent
        :param float beta: cost biasing exponent
        :param float path_threshold: max allowed path (-1: do not limit)
        :param float perturbation_multiplier: multiplier for network cost
        :param float perturbation_exponent: exponent for network cost
        :param float length_exponent: raise exponent cost to this power
        :param float degree_multiplier: multiplier for degree cost
        :param float degree_target: target node degree
        :param float pheromone_intensity: pheromone intensity multiplier (0: deposit nothing)
        :param float pheromone_exponent: pheromone intensity exponent
        :param float max_pheromone_deposited: deposit at most this much pheromone per run
        :param int sliding_window: use the last sliding_window solutions to normalize the pheromone to be deposited
        :param float evaporating_rate: evaporate this much pheromone per run
        :param float min_edge_desirability: minimum edge desirability when reducing pheromone
        :param float min_unbound: minimum number of runs when all edges can be removed (-1: off)
        :param float permanent_edge_threshold: edges with this much pheromone become static (-1: off)
        :param int min_edges_per_node: each edge must have at least this much nodes
        :param str algorithm: select "classic" (original ACO) or "modified" (modified version to generate
                              optimized maps) algorithm
        """

        # Data holder to store solutions found in multiple runs
        self.solutions = []
        self._cost_list = []

        self.path_threshold = path_threshold if path_threshold != -1 else float('inf')
        self.perturbation_multiplier = perturbation_multiplier
        self.perturbation_exponent = perturbation_exponent
        self.length_exponent = length_exponent
        self.degree_target = degree_target if degree_target else min_edges_per_node
        self.degree_multiplier = degree_multiplier
        self.degree_exponent = degree_exponent
        self.pheromone_intensity = pheromone_intensity
        self.pheromone_exponent = pheromone_exponent
        self.min_edge_desirability = min_edge_desirability
        self.evaporating_rate = evaporating_rate
        self.unbound_runs = min_unbound if min_unbound != -1 else float('inf')
        self.permanent_edge_threshold = permanent_edge_threshold if permanent_edge_threshold != -1 else float('inf')
        self.extra_edge_beta = extra_edge_beta
        self.max_pheromone_deposited = max_pheromone_deposited if max_pheromone_deposited != -1 else float('inf')
        self.min_edges_per_node = min_edges_per_node
        self.sliding_window = sliding_window
        self.algorithm = algorithm
        self.alpha = alpha
        self.beta = beta

        self.complete_network = network_graph.copy()
        self.complete_network_undirect = network_graph.to_undirected()

        # Minimum number of edges is a Harary graph. See: Harary, F. "The Maximum Connectivity of a Graph." Proc. Nat.
        # Acad. Sci. USA 48, 1142-1146, 1962. Also https://mathworld.wolfram.com/HararyGraph.html
        self.minimum_edges = numpy.ceil((self.complete_network.number_of_nodes() * self.min_edges_per_node) / 2)


class TopologyData:
    """ A class to store topology data.
    """

    class MoleculeTypeData:
        """ A class to store moleculetype data

        """

        class DataHolder(list):
            """ Simple class allowing to search using atom indices
            """

            def __getitem__(self, item):
                if isinstance(item, (int, slice)):
                    return super().__getitem__(item)
                elif isinstance(item, frozenset):
                    return self.search_by_index(item)
                else:
                    raise TypeError("DataHolder indices must be integers, slices, or frozenset, not {}"
                                    "".format(type(item)))

            def search_by_index(self, atoms):
                """ Search DataHolder for elements containing atom

                :param frozenset atoms: atom index
                :rtype: generator
                """

                atoms = frozenset(atoms) if not isinstance(atoms, frozenset) else atoms

                for each_element in self:
                    if atoms in frozenset(each_element[0:self.n_fields]):
                        yield each_element

            def search_all_with_index(self, id_list):
                """ Search DataHolder for all elements containing atom

                :param list id_list: list of atoms
                :rtype: generator
                """

                for each_element in self:
                    # True if any of elements from id_list is in the atom indices fields in each_element
                    if not set(id_list).isdisjoint(set(each_element[0:self.n_fields])):
                        yield each_element

            def __init__(self, *args, n_fields):
                if type(n_fields) != int and n_fields is not None:
                    raise TypeError("Expected int or NoneType, got {} instead".format(type(n_fields)))
                self.n_fields = n_fields
                super().__init__(args)

            def check_a_b_topology(self, item):
                """Checks for the equivalence of A and B parameters for item

                Parameters
                ----------
                item : object
                    Term to check

                Returns
                -------
                bool
                """

                # Get field names
                atom_fields = item.keys()[:self.n_fields]

                # Test that parameters A and B are the same
                get_a_b_terms = {i.replace('_A', '').replace('_B', '')
                                 for i in item.keys()
                                 if i not in atom_fields + ['function', 'comments']}
                if not all((item.__getattribute__('{}_A'.format(i)) == item.__getattribute__('{}_B'.format(i))
                            for i in get_a_b_terms)):
                    return False
                else:
                    return True

            def append(self, x):
                super().append(x)
                x._parent = self

        class MolNameDummy(str):
            """ Super dummy class to return molecule name line """

            def __init__(self, parent_class, molecule_name='LIG', nrexcl=3):
                self.parent_class = parent_class
                super().__init__()
                self._name = molecule_name
                self._nrexcl = nrexcl

            def __str__(self):
                try:
                    return '{:<10} {}'.format(self.parent_class.name, self._nrexcl) if self.parent_class.name \
                        else '{:<10} {}'.format(self._name, self._nrexcl)
                except (UnboundLocalError, AttributeError):
                    return '{:<10} {}'.format(self._name, self._nrexcl)

        @property
        def num_atoms(self):
            """ Get the number of atoms in this MoleculeType

            :rtype: int
            """
            return len(self.atoms_dict)

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, molecule_name):
            """ Sets the molecule name, propagating to residue name in atoms

            :param str molecule_name: new molecule name
            """

            self._name = molecule_name
            for a in self.atoms_dict.values():
                a.residue_name = molecule_name
            self.name_line._name = molecule_name

        def __init__(self, parent_class=None, molecule_name=''):
            """ Constructs a new MoleculeTypeData.

            :param TopologyData parent_class: outer TopologyData class
            :param str molecule_name: name of this molecule
            """

            self.output_sequence = []
            self._name = None

            self.name_line = self.MolNameDummy(self)
            if parent_class:
                self.parent_class = parent_class

            # FIXME: support pairs_nb
            self.atoms_dict = OrderedDict()
            self.bonds_dict = self.DataHolder(n_fields=2)
            self.pairs_dict = self.DataHolder(n_fields=2)
            self.pairsnb_dict = self.DataHolder(n_fields=2)
            self.exclusions_dict = self.DataHolder(n_fields=None)
            self.angles_dict = self.DataHolder(n_fields=3)
            self.dihe_dict = self.DataHolder(n_fields=4)
            self.constraints_dict = self.DataHolder(n_fields=2)
            self.settles_dict = self.DataHolder(n_fields=1)
            self.vsites2_dict = self.DataHolder(n_fields=3)
            self.vsites3_dict = self.DataHolder(n_fields=4)
            self.vsites4_dict = self.DataHolder(n_fields=5)

            if molecule_name:
                self.name = molecule_name

        def __str__(self):
            """ Returns a formatted representation of a topology block

            :rtype: str
            """

            return_data = []
            for each_element in self.output_sequence:
                if isinstance(each_element, str):
                    return_data.append(each_element.__str__())
                else:
                    return_data.extend(map(self._format_inline,
                                           self.parent_class.find_online_parameter(each_element, self.atoms_dict)))

            return '\n'.join(return_data)

        @staticmethod
        # TODO: automatically detect what are indexes and parameters and format accordingly
        def _format_inline(index_list, parameter_list=None, comment='', align_size=7):
            """Formats an parameter line

            Parameters
            ----------
            index_list : list
                list of indexes
            parameter_list : list
                list of parameters, if present
            comment : str
                inline comment, if any
            align_size : int
                column size for parameters_list, if present, for index_list otherwise if parameter_list=None

            Returns
            -------
            str
                Returns the formatted term line
            """

            if parameter_list is not None:
                terms = [index_list.__getattribute__(i) for i in index_list.keys()[:index_list._parent.n_fields + 1]]
                terms += [index_list.__getattribute__(i) for i in parameter_list]
                inline_param_str = ('{:<{align_size}} ' * len(terms)).format(*terms, align_size=align_size)
            else:
                # Test if this term holds topology A and B data, if so, print only top A
                if index_list.__class__.__name__ in ["BondDataDual", "AngleDataDual", "DihedralDataDual",
                                                     "ConstraintDataDual"]:
                    terms = [index_list.__getattribute__(i) for i in index_list.keys() if not i.endswith('_B')]
                    comment += 'Suppressed topology B data: {}' \
                               ''.format('; '.join(['{}: {}'.format(i, index_list.__getattribute__(i))
                                                   for i in index_list.keys() if i.endswith('_B')]))
                else:
                    terms = [index_list.__getattribute__(i) for i in index_list.keys()]
                if not comment and isinstance(index_list[-1], str):
                    # Comment is stored in index_list
                    comment = index_list[-1]
                    inline_param_str = ('{:<{align_size}} ' * (len(terms) - 1)).format(*terms[:-1],
                                                                                       align_size=align_size)
                else:
                    inline_param_str = ('{:<{align_size}} ' * len(terms)).format(*terms, align_size=align_size)

            if comment and comment.lstrip()[0] != ';':
                inline_param_str += '; {}'.format(comment)

            return inline_param_str

    # Fields for unpacking atomtypes
    __atomtype_dict = {0: namedlist('AtomTypeData', ['atom_type', 'm_u', 'q_e', 'particle_type', 'V', 'W', 'comments'],
                                    defaults=['']),
                       1: namedlist('AtomTypeData', ['atom_type', 'bonded_type', 'm_u', 'q_e', 'particle_type', 'V', 'W',
                                                     'comments'], defaults=['']),
                       2: namedlist('AtomTypeData', ['atom_type', 'atomic_number', 'm_u', 'q_e', 'particle_type', 'V', 'W',
                                                     'comments'], defaults=['']),
                       3: namedlist('AtomTypeData', ['atom_type', 'bonded_type', 'atomic_number', 'm_u', 'q_e',
                                                     'particle_type', 'V', 'W', 'comments'], defaults=['']), }

    # Fields for unpacking atoms
    __atomdata_fields = {1: ['atom_index', 'atom_type', 'residue_number', 'residue_name', 'atom_name',
                             'charge_group_number', 'q_e', 'm_u', 'comments'],
                         -1: ['atom_index', 'atom_type', 'residue_number', 'residue_name', 'atom_name',
                              'charge_group_number', 'q_e', 'comments']}

    # Fields for unpacking bonds (assembles a dict of possible fields list)
    __bonddata_dict = {1: ['b0', 'kb', 'comments'],  # Bond
                       2: ['b0', 'kb', 'comments'],  # G96 bond
                       3: ['b0', 'D', 'beta', 'comments'],  # Morse
                       4: ['b0', 'C', 'comments'],  # cubic bond
                       5: ['comments'],  # connection
                       6: ['b0', 'kb', 'comments'],  # harmonic potential
                       7: ['b0', 'kb', 'comments'],  # FENE bond
                       8: ['table_number', 'k', 'comments'],  # tabulated bond
                       9: ['table_number', 'k', 'comments'],  # tabulated bond
                       10: ['low', 'up1', 'up2', 'kdr', 'comments'],  # restraint potential
                       -1: ['comments']}  # bond read from somewhere else
    __bonddata_fields = {function: ['atom_i', 'atom_j', 'function', *parameters]
                         for function, parameters in __bonddata_dict.items()}

    # Fields for unpacking bonds for lines bearing both topology A and B data (assembles a dict of possible fields list)
    __bonddata_dict_dualtop = {code: ['{}_A'.format(i) for i in this_terms[:-1]]
                                     + ['{}_B'.format(i) for i in this_terms[:-1]]
                                     + [this_terms[-1]]
                               for code, this_terms in __bonddata_dict.items()
                               if code in [1, 2, 3, 6, 8, 9]}
    __bonddata_fields_dualtop = {function: ['atom_i', 'atom_j', 'function', *parameters]
                                 for function, parameters in __bonddata_dict_dualtop.items()}

    # Fields for unpacking pairs (assembles a dict of possible fields list)
    __pairsdata_dict = {1: ['V', 'W', 'comments'],  # extra LJ or Coulomb
                        2: ['fudge_QQ', 'qi', 'qj', 'V', 'W', 'comments'],  # extra LJ or Coulomb
                        -1: ['comments']}  # pair read from somewhere else
    __pairsdata_fields = {function: ['atom_i', 'atom_j', 'function', *parameters]
                          for function, parameters in __pairsdata_dict.items()}

    # Fields for unpacking pairs for lines bearing both topology A and B data (only fn=1 pairs allowed)
    __pairsdata_fields_dualtop = {1: ['atom_i', 'atom_j', 'function', 'V_A', 'W_A', 'V_B', 'W_B', 'comments']}

    __pairsnb_dict = {1: ['qi', 'qj', 'V', 'W', 'comments'],  # extra LJ or Coulomb
                      -1: ['comments']}  # nb pair read from somewhere else
    __pairsnbdata_fields = {function: ['atom_i', 'atom_j', 'function', *parameters]
                            for function, parameters in __pairsdata_dict.items()}

    # Fields for unpacking angles (assembles a dict of possible fields list)
    __angledata_dict = {1: ['theta0', 'k', 'comments'],  # angle
                        2: ['theta0', 'k', 'comments'],  # G96 angle
                        3: ['r1e', 'r2e', 'krr', 'comments'],  # cross bond-bond
                        4: ['r1e', 'r2e', 'r3e', 'kr_theta', 'comments'],  # cross bond-angle
                        5: ['tetha0', 'k0', 'r13', 'kUB', 'comments'],  # Urey-Bradley
                        6: ['theta0', 'C0', 'C1', 'C2', 'C3', 'C4', 'comments'],  # quartic angle
                        8: ['table_number', 'k', 'comments'],  # tabulated angle
                        9: ['table_number', 'comments'],  # tabulated bond
                        10: ['theta0', 'k0', 'comments'],  # restricted bending potential
                        -1: ['comments']}  # angle read from somewhere else
    __angledata_fields = {function: ['atom_i', 'atom_j', 'atom_k', 'function', *parameters]
                          for function, parameters in __angledata_dict.items()}

    # Fields for unpacking angle for lines bearing both topology A and B data (assembles a dict of possible fields list)
    __angledata_dict_dualtop = {code: ['{}_A'.format(i) for i in this_terms[:-1]]
                                        + ['{}_B'.format(i) for i in this_terms[:-1]]
                                        + [this_terms[-1]]
                                  for code, this_terms in __angledata_dict.items()
                                  if code in [1, 2, 5, 8]}
    __angledata_fields_dualtop = {function: ['atom_i', 'atom_j', 'atom_k', 'function', *parameters]
                                  for function, parameters in __angledata_dict_dualtop.items()}

    # Fields for unpacking dihedrals (assembles a dict of possible fields list)
    __dihedata_dict = {1: ['phi', 'k', 'multiplicity', 'comments'],  # proper dihedral
                       2: ['zeta0', 'k', 'comments'],  # improper dihedral
                       3: ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'comments'],  # Ryckaert-Bellemans dihedral
                       4: ['pih', 'k', 'multiplicity', 'comments'],  # periodic improper dihedral
                       5: ['C1', 'C2', 'C3', 'C4', 'comments'],  # Fourier dihedral
                       8: ['table_number', 'k', 'comments'],  # tabulated dihedral
                       9: ['phi', 'k', 'multiplicity', 'comments'],  # proper dihedral (multiple)
                       10: ['phi0', 'k', 'comments'],  # restricted dihedral
                       11: ['a0', 'a1', 'a2', 'a3', 'a4', 'comments'],  # combined bending-torsion potential
                       -1: ['comments']}  # dihedral read from somewhere else
    __dihedata_fields = {function: ['atom_i', 'atom_j', 'atom_k', 'atom_l', 'function', *parameters]
                         for function, parameters in __dihedata_dict.items()}

    # Fields for unpacking dihedral for lines bearing both topology A and B data (assembles a dict of possible fields
    # list)
    __dihedata_dict_dualtop = {code: ['{}_A'.format(i) for i in this_terms[:-1]]
                                      + ['{}_B'.format(i) for i in this_terms[:-1]]
                                      + [this_terms[-1]]
                                for code, this_terms in __dihedata_dict.items()
                                if code in [1, 2, 3, 4, 5, 8, 9]}
    __dihedata_fields_dualtop = {function: ['atom_i', 'atom_j', 'atom_k', 'atom_l', 'function', *parameters]
                                 for function, parameters in __dihedata_dict_dualtop.items()}

    # Fields for unpacking constraints (assembles a dict of possible fields list)
    __constraint_dict = {1: ['b0', 'comments'],  # proper dihedral
                         2: ['b0', 'comments'],  # improper dihedral
                         -1: ['comments']}  # dihedral read from somewhere else
    __constraint_fields = {function: ['atom_i', 'atom_j', 'function', *parameters]
                           for function, parameters in __constraint_dict.items()}

    # Fields for unpacking constraints for lines bearing both topology A and B data (assembles a dict of possible
    # fields list)
    __constraint_dict_dualtop = {code: ['{}_A'.format(i) for i in this_terms[:-1]]
                                       + ['{}_B'.format(i) for i in this_terms[:-1]]
                                       + [this_terms[-1]]
                                 for code, this_terms in __constraint_dict.items() if code in [1, 2]}
    __constraint_fields_dualtop = {function: ['atom_i', 'atom_j', 'function', *parameters]
                                   for function, parameters in __constraint_dict_dualtop.items()}

    # Fields for unpacking SETTLE directives. No FE is allowed and only fn = 1 is supported with a single atom
    __setttles_data = namedlist('SettlesData', ['atom_ow', 'function', 'doh', 'dhh', 'comments'], defaults=[''])

    # Fields for unpacking 2-atom virtual site
    __vsite2_data = namedlist('VirtualSite2Data', ['site', 'atom_i', 'atom_j', 'function', 'a', 'comments'],
                              defaults=[''])

    # Fields for unpacking 3-atom virtual site (assembles a dict of possible fields list)
    __vsite3_dict = {1: ['a', 'b', 'comments'],  # type 3
                     2: ['a', 'd', 'comments'],  # type 3fd
                     3: ['theta', 'd', 'comments'],  # type 3fad
                     4: ['a', 'b', 'c', 'comments']}  # type 3out
    __vsite3_fields = {function: ['site', 'atom_i', 'atom_j', 'atom_k', 'function', *parameters]
                       for function, parameters in __vsite3_dict.items()}

    # FIXME: implement support for missing directives and constraints

    def __init__(self, topology_files=None, verbosity=0):
        """ Constructor of TolopogyData

        :param [str, list] topology_files: reads topology from this file/these files
        :param int verbosity: set verbosity level
        """

        # This holds the unmodified lines from the file and references to elements
        self.atomtype_dict = OrderedDict()
        self.output_sequence = []
        self.molecules = []

        self.__online_bondtypes = {}
        self.__online_pairtypes = {}
        self.__online_angletypes = {}
        self.__online_dihedraltypes = {}
        self.__online_constrainttypes = {}

        self.type_directive_dict = {'bondtypes': {'data_fields': self.__bonddata_fields,
                                                  'function_index': 2,
                                                  'online_dict': self.__online_bondtypes,
                                                  'class_name': 'BondData'},
                                    'pairtypes': {'data_fields': self.__pairsdata_fields,
                                                  'function_index': 2,
                                                  'online_dict': self.__online_pairtypes,
                                                  'class_name': 'PairData'},
                                    'angletypes': {'data_fields': self.__angledata_fields,
                                                   'function_index': 3,
                                                   'online_dict': self.__online_angletypes,
                                                   'class_name': 'AngleData'},
                                    'dihedraltypes': {'data_fields': self.__dihedata_fields,
                                                      'function_index': 4,
                                                      'online_dict': self.__online_dihedraltypes,
                                                      'class_name': 'DihedralData'},
                                    'constrainttypes': {'data_fields': self.__constraint_fields,
                                                        'function_index': 2,
                                                        'online_dict': self.__online_constrainttypes,
                                                        'class_name': 'ConstraintData'}}

        self.online_terms_dict = {each_element['class_name']: each_element for each_element
                                  in self.type_directive_dict.values()}

        if topology_files is not None:
            if isinstance(topology_files, str):
                self.read_topology([topology_files], verbosity=verbosity)
            else:
                self.read_topology(topology_files, verbosity=verbosity)

    @staticmethod
    def type_converter(value):
        """ Splits a string into parts converts them to appropriated types

        :param str value: string to be converted
        :rtype: list
        """

        return_list = []
        value = value.split(';', 1)
        for each_part in value[0].split():
            return_list.append(os_util.detect_type(each_part, test_for_boolean=False))
        try:
            return_list.append(value[1])
        except IndexError:
            pass
        finally:
            return return_list

    @staticmethod
    def detect_solute_molecule_name(input_file, test_sol_molecules=None, gmx_bin='gmx', no_checks=False, verbosity=0):
        """Process file, trying to guess the name of the water molecule

        Parameters
        ----------
        input_file : str
            Input file. Must be one of GROMACS-compatible index (.ndx), GROMACS run file (.tpr), GROMACS-compatible
            topology (.top or .itp), or PDB. Type will be guessed from extension.
        test_sol_molecules : list
            Try this molecule names
        gmx_bin : str
            GROMACS binary
        no_checks : bool
            Ignore checks and keep going
        verbosity : int
            Sets verbosity level

        Returns
        -------
        str
        """

        from prepare_dual_topology import read_index_data, make_index

        if not test_sol_molecules:
            test_sol_molecules = ['SOL', 'WAT', 'HOH', 'TIP3']
        if isinstance(test_sol_molecules, str):
            test_sol_molecules = [test_sol_molecules]

        file_type = os.path.splitext(input_file)[1]

        if file_type == '.ndx':
            molecule_names = read_index_data(input_file, verbosity=verbosity)
            if not molecule_names:
                os_util.local_print('Failed to find process index file {} while trying to guess the name of the '
                                    'water molecules. Cannot continue.'.format(input_file),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
            else:
                molecule_names = molecule_names.keys()
        elif file_type == '.tpr':
            # Input file is a tpr, use gmx make_ndx, prepare a temporary index file, then process it. make_index is
            # called with method=internal because mdanalysis method requires the water molecule name for the default
            # groups.
            with tempfile.TemporaryDirectory() as this_tmpdir:
                tmp_index = os.path.join(this_tmpdir, 'index.ndx')
                make_index(new_index_file=tmp_index, structure_data=input_file, gmx_bin=gmx_bin,
                                                 method='internal', verbosity=verbosity)
                molecule_names = read_index_data(tmp_index, verbosity=verbosity).keys()
        elif file_type in ['.top', '.itp']:
            # File is a GROMACS-compatible topology file, read [ system ] definition
            topology_data = os_util.read_file_to_buffer(input_file, die_on_error=True, return_as_list=True,
                                                        verbosity=verbosity,
                                                        error_message='Failed to read topology file when trying to '
                                                                      'guess water model.')
            fn_match = re.compile(r'\s*\[\s+system\s+]', flags=re.IGNORECASE).match
            position = os_util.inner_search(fn_match, topology_data, apply_filter=';')
            if position is False:
                os_util.local_print('Failed to find a [ system ] directive in topology file {} while trying to guess '
                                    'the name of the water molecules. Cannot continue.'
                                    ''.format(input_file),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise SystemExit(1)
            molecule_names = [each_line.split()[0] for each_line in topology_data[position:]
                              if each_line.strip() and not each_line.lstrip().startswith(';')]

        elif file_type in ['.pdb']:
            # Trying to read residues from PDB file
            # system_data = PDBFile(input_file=input_file)
            # molecule_names = {i.resname for i in system_data.residues}
            with tempfile.TemporaryDirectory() as this_tmpdir:
                tmp_index = os.path.join(this_tmpdir, 'index.ndx')
                make_index(new_index_file=tmp_index, structure_data=input_file, gmx_bin=gmx_bin,
                           method='internal', verbosity=verbosity)
                molecule_names = read_index_data(tmp_index, verbosity=verbosity).keys()

        else:
            os_util.local_print('Cannot detect a water model in file {} because its format ({}) is now known. Please, '
                                'use one of .ndx, .tpr, .top, .itp, or .pdb.'
                                ''.format(input_file, file_type),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError("File format {} unknown".format(file_type))

        if len(molecule_names) == 0:
            os_util.local_print('Cannot find water model in file {}. Cannot go on.'
                                ''.format(input_file),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError

        found_water_molecules = set(molecule_names).intersection(set(test_sol_molecules))
        if len(found_water_molecules) == 0:
            # No matches found
            os_util.local_print('None of the water molecules {} was found in file {}. The following molecule names '
                                'were read: {}'
                                ''.format(test_sol_molecules, input_file, molecule_names),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise KeyError("Water molecule not found".format(file_type))
        elif len(found_water_molecules) == 1:
            # Water molecule found, return it
            return set(molecule_names).intersection(set(test_sol_molecules)).pop()
        else:
            # There are multiple molecules names matching candidates test_sol_molecules
            if no_checks:
                selected_water_name = molecule_names[0]
                for selected_water_name in molecule_names:
                    if selected_water_name in test_sol_molecules:
                        break

                os_util.local_print('More than a single molecule name in {} matches candidates water molecule names '
                                    '({}), namely {}. Because you are running with no_checks, I am selecting {} and '
                                    'going on!'
                                    ''.format(input_file, test_sol_molecules,
                                              set(molecule_names).intersection(set(test_sol_molecules)),
                                              selected_water_name),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                return selected_water_name
            else:
                os_util.local_print('More than a single molecule name in {} matches candidates water molecule names '
                                    '({}), namely {}. Cannot go on. Check your inputs or rerun using no_checks to '
                                    'suppress this warning.'
                                    ''.format(input_file, test_sol_molecules,
                                              set(molecule_names).intersection(set(test_sol_molecules))),
                                    msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                raise KeyError("Multiple water names found.")

    @staticmethod
    def count_exact_atom_matches(atom_types_a, atom_types_b):
        """Count the number of exact matches between the two groups of atom types

        Parameters
        ----------
        atom_types_a : tuple
            Atom types
        atom_types_b : tuple
            Atom types
        Returns
        -------
        int
            Number of exact matches, -1 if no matches are found
        """

        if atom_types_a[-1] != atom_types_b[-1]:
            # Dihedral types don't match, break
            return -1

        if 'X' not in atom_types_b:
            if atom_types_a == atom_types_b:
                # If there are no wildcards in B, the only possible match is all atoms matching
                return 4
            else:
                return -1

        n_matches = 0
        for i, j in zip(atom_types_a[:-1], atom_types_b[:-1]):
            if i == j:
                # Exact match
                n_matches += 1
            elif j == 'X':
                # Wildcard, accept, but do not count as exact
                pass
            else:
                # Atom types do no match
                break
        else:
            return n_matches

        # for was not broken, therefore a non-match was found
        return -1

    def find_online_param_wildcard(self, each_element, online_dict):
        """ Finds online parameters for dihedrals using wildcards and returns them as a list. GROMACS wildcard logic is
        described in the manual as "For [ dihedraltypes ] wildcard atom type names can be specified with the letter X
        in one or more of the four positions. Thus one can for example assign proper dihedral parameters based on the
        types of the middle two atoms. The parameters for the entry with the most exact matches, i.e. the least
        wildcard matches, will be used.".

        Parameters
        ----------
        each_element : list
            list of parameter data (actually, a class generated from namedlist is expected)
        online_dict : dict
            topology atom data

        Returns
        -------
            list
        """

        n_matches = [self.count_exact_atom_matches(each_element, each_dihedral) for each_dihedral in online_dict]

        for i in range(3, 0, -1):
            try:
                # This will get the first line with n_matches matches
                key = n_matches.index(i)
            except ValueError:
                pass
            else:
                break
        else:
            # No matching dihedral line was found
            return False

        # TODO: this is O(n), which isn't great. In general, the online dict shouldn't be modified after read, so maybe
        #       it could be possible to cache list(online_dict). Or maybe don't use OrderedDicts at all. Or something
        #       that inherits from OrderedDict, caches itself as a list and update the cache if needed. For molecules
        #       using many dihedrals defined using wildcards, this *may* become a problem.
        return list(online_dict.values())[key]

    def find_online_parameter(self, each_element, atoms_dict):
        """ Finds online parameters and returns them as a list

        Parameters
        ----------
        each_element : list
            list of parameter data (actually, a class generated from namedlist is expected)
        atoms_dict : dict
            topology atom data

        Returns
        -------
            list
        """

        this_term_name = each_element.__class__.__name__

        return_data = []

        # Inline the online terms
        if this_term_name in self.online_terms_dict and \
                len(each_element) == self.online_terms_dict[this_term_name]['function_index'] + 2:
            function_index = self.online_terms_dict[this_term_name]['function_index']
            online_dict = self.online_terms_dict[this_term_name]['online_dict']

            # Build the index using atom data
            this_index = [atoms_dict[i].atom_type for i in each_element[:function_index]]
            this_index.append(each_element[function_index])
            this_index = tuple(this_index)

            try:
                this_param_data = online_dict[this_index]
            except KeyError as error1:
                if this_term_name == 'PairData':
                    # It's a pair, no term is ok
                    return_data.append(each_element)
                else:
                    # Try to match a bonded data using atomtype data
                    this_index_2 = [self.atomtype_dict[atoms_dict[i].atom_type].bonded_type
                                    for i in each_element[:function_index]]
                    this_index_2.append(each_element[function_index])
                    this_index_2 = tuple(this_index_2)
                    try:
                        this_param_data = online_dict[this_index_2]
                    except KeyError as error2:
                        # In case it's a dihedral, fall back to matching wildcards ("X" in GROMACS). Does GROMACS match
                        # only atom types to wildcards? Or does it match using atom names? I implemented only the
                        # former, as I didn't figure out from toppush.cpp the code.
                        if this_term_name == 'DihedralData':
                            this_param_data = self.find_online_param_wildcard(this_index_2, online_dict)
                            if not this_param_data:
                                os_util.local_print('Could not find an online parameter for dihedral "{}" while '
                                                    'parsing line {}. Please, make sure your topology includes all '
                                                    'the required parameters. The likely cause for this is that your '
                                                    'ligand topology uses parameters from the force field which are '
                                                    'not included in the ligand topology. This is, in principle, not '
                                                    'supported. Specifically, if you used CGenFF, make sure to '
                                                    'select "Include parameters that are already in CGenFF". Please, '
                                                    'see the manual for further info regarding this topic and your '
                                                    'alternatives.'
                                                    ''.format(' '.join(map(str, this_index)), each_element),
                                                    msg_verbosity=os_util.verbosity_level.error)
                                raise error2
                            else:
                                os_util.local_print('Using dihedral data {} for dihedral {}, after matching exactly {} '
                                                    'atoms.'
                                                    ''.format(' '.join(map(str, this_index)), each_element,
                                                              this_param_data),
                                                    msg_verbosity=os_util.verbosity_level.info)
                        else:
                            os_util.local_print('Could not find an online parameter for "{}" while parsing line {}. '
                                                'Please, make sure your topology includes all the required '
                                                'parameters. The likely cause for this is that your ligand topology '
                                                'uses parameters from the force field which are not included in the '
                                                'ligand topology. This is, in principle, not supported. '
                                                'Specifically, if you used CGenFF, make sure to select "Include '
                                                'parameters that are already in CGenFF". Please, see the manual for '
                                                'further info regarding this topic and your alternatives.'
                                                ''.format(' '.join(map(str, this_index)), each_element),
                                                msg_verbosity=os_util.verbosity_level.error)
                            raise KeyError(error2)

            if this_term_name == 'DihedralData' and each_element[function_index] == 9:
                # Assemble a new term list with online parameters
                for each_line in this_param_data:
                    new_list = [*each_element[:function_index + 1], *each_line[function_index + 1:-1],
                                '{} - {}'.format(each_element[-1], each_line[-1])]
                    each_element = namedlist(this_term_name,
                                             self.online_terms_dict[this_term_name]['data_fields'][
                                                 each_element.function],
                                             defaults=[''])(*new_list)
                    return_data.append(each_element)
            elif this_term_name != 'PairData':
                new_list = [*each_element[:function_index + 1], *this_param_data[function_index + 1:-1],
                            '{} - {}'.format(each_element[-1], this_param_data[-1])]
                each_element = namedlist(this_term_name,
                                         self.online_terms_dict[this_term_name]['data_fields'][each_element.function],
                                         defaults=[''])(*new_list)
                return_data.append(each_element)

        else:
            # Not online or not allowed to be online
            return_data.append(each_element)

        return return_data

    def add_atom(self, atom_string, molecule_type, verbosity):
        """Parse an atom line and adds it to a MoleculeTypeData object

        Parameters
        ----------
        atom_string : str
            Atom line to be processed, read from a GROMACS-compatible topology file
        molecule_type : MoleculeTypeData
            MoleculeTypeData object to associate atom data to
        verbosity : int
            Sets the verbosity level
        """

        this_atom_data = self.type_converter(atom_string)
        try:
            # Try to interpret atom as if the mass is explicit
            this_atom = namedlist('AtomData', self.__atomdata_fields[1], defaults=[''])(*this_atom_data)
        except TypeError:
            # Try to interpret atom as if the mass is online
            try:
                this_atom = namedlist('AtomData', self.__atomdata_fields[-1], defaults=[''])(*this_atom_data)
            except TypeError as error:
                os_util.local_print('Error while parsing atom data:\n{}'.format(self.type_converter(atom_string)),
                                    msg_verbosity=os_util.verbosity_level.error)
                raise TypeError(error)

        try:
            atom_idx = int(this_atom.atom_index)
            if atom_idx not in molecule_type.atoms_dict:
                molecule_type.atoms_dict[atom_idx] = this_atom
            else:
                os_util.local_print('Atom {} redeclared in line "{}" for moleculetype {}.'
                                    ''.format(molecule_type.atoms_dict[atom_idx], atom_string, molecule_type),
                                    msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                while atom_idx in molecule_type.atoms_dict:
                    atom_idx = '{}\''.format(atom_idx)
                molecule_type.atoms_dict[atom_idx] = this_atom
        except TypeError as error:
            os_util.local_print('Error while parsing atom line "{}" in molecule {} with error {}'
                                ''.format(atom_string, molecule_type.name, error),
                                msg_verbosity=os_util.verbosity_level.error)

            raise TypeError('Could not convert atom index value {} to integer'.format(this_atom.atom_index))
        else:
            molecule_type.output_sequence.append(this_atom)

    def add_bond(self, bond_string, molecule_type):
        """ Reads a bond line

        :param str bond_string: atom line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate bond to
        """

        this_bond_data = self.type_converter(bond_string)
        try:
            # Try to interpret bond as if no parameters are explicit
            this_bond = namedlist('BondData', self.__bonddata_fields[-1], defaults=[''])(*this_bond_data)
        except (TypeError, KeyError):
            # It failed, try to read the parameters as if it bears a A and B topology data
            try:
                this_bond = namedlist('BondDataDual', self.__bonddata_fields_dualtop[int(this_bond_data[2])],
                                      defaults=[''])(*this_bond_data)
            except (TypeError, KeyError):
                # It failed, try to read the parameters as if it's bears only topology A parameters
                try:
                    this_bond = namedlist('BondData', self.__bonddata_fields[int(this_bond_data[2])],
                                          defaults=[''])(*this_bond_data)
                except (TypeError, KeyError) as error:
                    # It failed, the function does not match the number of parameters
                    os_util.local_print('Error while parsing bond line "{}" in molecule {} with error {}'
                                        ''.format(bond_string, molecule_type.name, error),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not understand bond line {}'.format(bond_string))
                else:

                    molecule_type.bonds_dict.append(this_bond)
                    molecule_type.output_sequence.append(molecule_type.bonds_dict[-1])
            else:
                # Test that parameters A and B are the same
                if not molecule_type.bonds_dict.check_a_b_topology(this_bond):
                    os_util.local_print('Error while parsing bond line "{}" in molecule {}. A and B topology terms do '
                                        'not match. Using GROMACS topology interpolation is not supported. Please, see '
                                        'the manual.'
                                        ''.format(bond_string, molecule_type.name),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not parse bond line {}'.format(bond_string))

                molecule_type.bonds_dict.append(this_bond)
                molecule_type.output_sequence.append(molecule_type.bonds_dict[-1])
        else:
            # Parameters are online, try to inline them
            new_term_list = self.find_online_parameter(this_bond, molecule_type.atoms_dict)
            molecule_type.bonds_dict.extend(new_term_list)
            molecule_type.output_sequence.extend(new_term_list)

    def add_pair(self, pair_string, molecule_type):
        """ Reads a bond line

        :param str pair_string: atom line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate pair to
        """

        this_pair_data = self.type_converter(pair_string)
        try:
            # Try to interpret pair as if no parameters are explicit
            this_pair = namedlist('PairData', self.__pairsnbdata_fields[-1], defaults=[''])(*this_pair_data)
        except TypeError:
            # It failed, try to read the parameters
            try:
                this_pair = namedlist('PairData', self.__pairsdata_fields[int(this_pair_data[2])],
                                      defaults=[''])(*this_pair_data)
            except (TypeError, KeyError) as error:
                # It failed, the function does not match the number of parameters
                os_util.local_print('Error while parsing pair line {} in molecule {} with error {}'
                                    ''.format(pair_string, molecule_type, error),
                                    msg_verbosity=os_util.verbosity_level.error)
                raise TypeError('Could not understand pair line {}'.format(pair_string))
            else:
                molecule_type.pairs_dict.append(this_pair)
                molecule_type.output_sequence.append(molecule_type.pairs_dict[-1])
        else:
            new_term_list = self.find_online_parameter(this_pair, molecule_type.atoms_dict)
            molecule_type.pairs_dict.extend(new_term_list)
            molecule_type.output_sequence.extend(new_term_list)

    # FIXME: this is wrong!!! handle pair_nb correctly
    def add_pair_nb(self, pair_string, molecule_type):
        """ Reads a non-bonded line

        :param str pair_string: bond line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate pair_nb to

        """

        this_pair_data = self.type_converter(pair_string)
        try:
            # Try to interpret pair as if no parameters are explicit
            this_pair = namedlist('PairData', self.__pairsnbdata_fields[-1], defaults=[''])(*this_pair_data)
        except TypeError:
            # It failed, try to read the parameters
            try:
                this_pair = namedlist('PairData', self.__pairsnbdata_fields[int(this_pair_data[2])],
                                      defaults=[''])(*this_pair_data)
            except TypeError as error:
                os_util.local_print('Error while parsing pair line {} in molecule {} with error {}'
                                    ''.format(pair_string, molecule_type, error),
                                    msg_verbosity=os_util.verbosity_level.error)
                raise TypeError('Could not understand pair line {}'.format(pair_string))
            else:
                molecule_type.pairsnb_dict.append(this_pair)
                molecule_type.output_sequence.append(molecule_type.pairsnb_dict[-1])
        else:
            molecule_type.pairsnb_dict.append(this_pair)
            molecule_type.output_sequence.append(molecule_type.pairsnb_dict[-1])

    def add_angle(self, angle_string, molecule_type):
        """ Reads a angle line

        :param str angle_string: angle line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate angle to
        """

        this_angle_data = self.type_converter(angle_string)
        try:
            # Try to interpret angle as if no parameters are explicit
            this_angle = namedlist('AngleData', self.__angledata_fields[-1], defaults=[''])(*this_angle_data)
        except (TypeError, KeyError):
            # It failed, try to read the parameters as if it bears a A and B topology data
            try:
                this_angle = namedlist('AngleDataDual', self.__angledata_fields_dualtop[int(this_angle_data[3])],
                                       defaults=[''])(*this_angle_data)
            except (TypeError, KeyError):
                # It failed, try to read the parameters as if it's bears only topology A parameters
                try:
                    this_angle = namedlist('AngleData', self.__angledata_fields[int(this_angle_data[3])],
                                           defaults=[''])(*this_angle_data)
                except (TypeError, KeyError) as error:
                    # It failed, the function does not match the number of parameters
                    os_util.local_print('Error while parsing angle line "{}" in molecule {} with error {}'
                                        ''.format(angle_string, molecule_type.name, error),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not understand angle line {}'.format(angle_string))
                else:
                    molecule_type.angles_dict.append(this_angle)
                    molecule_type.output_sequence.append(molecule_type.angles_dict[-1])

            else:
                # Test that parameters A and B are the same
                if not molecule_type.angles_dict.check_a_b_topology(this_angle):
                    os_util.local_print('Error while parsing angle line "{}" in molecule {}. A and B topology terms do '
                                        'not match. Using GROMACS topology interpolation is not supported. Please, see '
                                        'the manual.'
                                        ''.format(angle_string, molecule_type.name),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not parse angle line {}'.format(angle_string))

                molecule_type.angles_dict.append(this_angle)
                molecule_type.output_sequence.append(molecule_type.angles_dict[-1])
        else:
            new_term_list = self.find_online_parameter(this_angle, molecule_type.atoms_dict)
            molecule_type.angles_dict.extend(new_term_list)
            molecule_type.output_sequence.extend(new_term_list)

    def add_dihedral(self, dihedral_string, molecule_type):
        """Reads a dihedral dihedral line

        Parameters
        ----------
        dihedral_string : str
            Dihedral line to be processed
        molecule_type : MoleculeTypeData
            MoleculeTypeData object to associate dihedral to
        """

        this_dihedral_data = self.type_converter(dihedral_string)
        try:
            # Try to interpret dihedral as if no parameters are explicit
            this_dihedral = namedlist('DihedralData', self.__dihedata_fields[-1], defaults=[''])(*this_dihedral_data)
        except (TypeError, KeyError):
            # It failed, try to read the parameters as if it bears a A and B topology data
            try:
                this_dihedral = namedlist('DihedralDataDual',
                                          self.__dihedata_fields_dualtop[int(this_dihedral_data[4])],
                                          defaults=[''])(*this_dihedral_data)
            except (TypeError, KeyError):
                # It failed, try to read the parameters as if it's bears only topology A parameters
                try:
                    this_dihedral = namedlist('DihedralData', self.__dihedata_fields[int(this_dihedral_data[4])],
                                           defaults=[''])(*this_dihedral_data)
                except (TypeError, KeyError) as error:
                    # It failed, the function does not match the number of parameters
                    os_util.local_print('Error while parsing dihedral line "{}" in molecule {} with error {}'
                                        ''.format(dihedral_string, molecule_type.name, error),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not understand dihedral line {}'.format(dihedral_string))
                else:
                    molecule_type.dihe_dict.append(this_dihedral)
                    molecule_type.output_sequence.append(molecule_type.dihe_dict[-1])

            else:
                # Test that parameters A and B are the same
                if not molecule_type.dihe_dict.check_a_b_topology(this_dihedral):
                    os_util.local_print('Error while parsing dihedral line "{}" in molecule {}. A and B topology terms '
                                        'do not match. Using GROMACS topology interpolation is not supported. Please, '
                                        'see the manual.'.format(dihedral_string, molecule_type.name),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not parse dihedral line {}'.format(dihedral_string))

                molecule_type.dihe_dict.append(this_dihedral)
                molecule_type.output_sequence.append(molecule_type.dihe_dict[-1])
        else:
            new_term_list = self.find_online_parameter(this_dihedral, molecule_type.atoms_dict)
            molecule_type.dihe_dict.extend(new_term_list)
            molecule_type.output_sequence.extend(new_term_list)

    def add_constraint(self, constraint_string, molecule_type):
        """ Reads a constraint line

        :param str constraint_string: constraint line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate constraint to
        """

        this_constraint_data = self.type_converter(constraint_string)
        try:
            # Try to interpret constraint as if no parameters are explicit
            this_constraint = namedlist('ConstraintData',
                                        self.__constraint_fields[-1], defaults=[''])(*this_constraint_data)
        except (TypeError, KeyError):
            # It failed, try to read the parameters as if it bears a A and B topology data
            try:
                this_constraint = namedlist('ConstraintDataDual',
                                            self.__constraint_fields_dualtop[int(this_constraint_data[3])],
                                            defaults=[''])(*this_constraint_data)
            except (TypeError, KeyError):
                # It failed, try to read the parameters as if it's bears only topology A parameters
                try:
                    this_constraint = namedlist('ConstraintData',
                                                self.__constraint_fields[int(this_constraint_data[3])],
                                                defaults=[''])(*this_constraint_data)
                except (TypeError, KeyError) as error:
                    # It failed, the function does not match the number of parameters
                    os_util.local_print('Error while parsing constraint line "{}" in molecule {} with error {}'
                                        ''.format(constraint_string, molecule_type.name, error),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not understand constraint line "{}"'.format(constraint_string))
                else:
                    molecule_type.constraints_dict.append(this_constraint)
                    molecule_type.output_sequence.append(molecule_type.constraints_dict[-1])

            else:
                # Test that parameters A and B are the same
                if not molecule_type.constraints_dict.check_a_b_topology(this_constraint):
                    os_util.local_print('Error while parsing constraint line "{}" in molecule {}. A and B topology '
                                        'terms do not match. Using GROMACS topology interpolation is not supported. '
                                        'Please, see the manual.'.format(constraint_string, molecule_type.name),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise TypeError('Could not parse constraint line "{}"'.format(constraint_string))

                molecule_type.constraints_dict.append(this_constraint)
                molecule_type.output_sequence.append(molecule_type.constraints_dict[-1])
        else:
            new_term_list = self.find_online_parameter(this_constraint, molecule_type.atoms_dict)
            molecule_type.constraints_dict.extend(new_term_list)
            molecule_type.output_sequence.extend(new_term_list)

    def add_exclusion(self, exclusion_string, molecule_type):
        """ Reads a exclusion line

        :param str exclusion_string: exclusion line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate exclusion to
        """

        this_exclusion_list = self.type_converter(exclusion_string)
        __exclusion_data = namedlist('ExclusionsData',
                                     ['atom_ref'] + ['atom_{}'.format(i) for i in range(len(this_exclusion_list) - 1)]
                                     + ['comments'], defaults=[''])
        this_exclusion = __exclusion_data(*this_exclusion_list)

        molecule_type.exclusions_dict.append(this_exclusion)
        molecule_type.output_sequence.append(molecule_type.exclusions_dict[-1])

    def add_atomtype(self, atomtype_string, verbosity):
        """ Reads a atomtype line. GROMACS manual states that bonded type and atomic number are optional, but it
        doesn't explain how this is done. There is a description at src/gromacs/gmxpreprocess/toppush.cpp:push_at. From
        toppush.cpp (lines 339-367):

            * A. Read in the first six fields as strings
            * B. If field 3 (starting from 0) is a single char, we have neither
            *    bonded_type or atomic numbers.
            * C. If field 5 is a single char we have both.
            * D. If field 4 is a single char we check field 1. If this begins with
            *    an alphabetical character we have bonded types, otherwise atomic numbers.

        :param str atomtype_string: atomtype line
        :param int verbosity: verbosity level
        """

        this_atomtype_data = self.type_converter(atomtype_string)
        if isinstance(this_atomtype_data[3], str) and len(this_atomtype_data[3]) == 1:
            # Field 3 is a single char, neither bonded_type nor atomic numbers were read.
            this_atomtype = self.__atomtype_dict[0](*this_atomtype_data)
        elif isinstance(this_atomtype_data[5], str) and len(this_atomtype_data[5]) == 1:
            # Field 5 is a single char, read both bonded_type and atomic_number are present
            this_atomtype = self.__atomtype_dict[3](*this_atomtype_data)
        elif isinstance(this_atomtype_data[4], str) and len(this_atomtype_data[4]) == 1:
            # Field 4 is a single char, checking field 1
            if isinstance(this_atomtype_data[1], str):
                # It begins with an alphabetical character (in Python, this is a string), therefore we have bonded_type
                this_atomtype = self.__atomtype_dict[1](*this_atomtype_data)
            else:
                # We have atomic_number
                this_atomtype = self.__atomtype_dict[2](*this_atomtype_data)
        else:
            os_util.local_print('Failed to process atomtype data in line {}. There were the fields read: {}'
                                ''.format(atomtype_string, this_atomtype_data),
                                msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
            raise ValueError

        if (this_atomtype.atom_type in self.atomtype_dict) and (verbosity > -1):
            os_util.local_print('Atomtype {} redeclared! Only last entry will be kept.'.format(this_atomtype.atom_type),
                                msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
        self.output_sequence.append(this_atomtype)
        self.atomtype_dict[this_atomtype.atom_type] = self.output_sequence[-1]

    def add_bondedtype(self, bondedtype_string, type_directive):
        """ Reads a bondtype line

        :param str bondedtype_string: bondtype, pairtype, angletype, dihedraltype or constrainttype line
        :param str type_directive: directive name, one of bondtype, pairtype, angletype, dihedraltype or constrainttype
        """

        if type_directive not in self.type_directive_dict.keys():
            raise KeyError('Bonded type {} not recognized. Please use one of: {}'
                           ''.format(type_directive, ', '.join(self.type_directive_dict.keys())))

        this_bonded_data = self.type_converter(bondedtype_string)

        this_functionindex = self.type_directive_dict[type_directive]['function_index']
        this_functioncode = this_bonded_data[this_functionindex]
        if type_directive == 'dihedraltypes':
            # dihedraltypes directive accepts either 2 or 4 atoms. From gromacs/gmxpreprocess/toppush.cpp, it is done
            # like that (gromacs/gmxpreprocess/toppush.cpp:919):
            #      * We first check for 2 atoms with the 3th column being an integer
            #      * defining the type. If this isn't the case, we try it with 4 atoms
            #      * and the 5th column defining the dihedral type.
            if isinstance(this_bonded_data[2], int):
                this_functioncode = this_bonded_data[2]
                # It is a 2-atoms dihedraltype. Move atoms and use 'X' as wildcards (also following GROMACS)
                if this_bonded_data[2] == 2:
                    this_bonded_data.insert(1, 'X')
                    this_bonded_data.insert(1, 'X')
                else:
                    this_bonded_data.insert(0, 'X')
                    this_bonded_data.insert(2, 'X')
            this_linetype = self.type_directive_dict[type_directive]['data_fields'][this_functioncode]

        elif type_directive != 'constrainttype':
            this_linetype = self.type_directive_dict[type_directive]['data_fields'][this_functioncode]
        else:
            this_linetype = self.type_directive_dict[type_directive]['data_fields']
        this_online_dict = self.type_directive_dict[type_directive]['online_dict']

        try:
            this_bonded_type = namedlist('BondedTypeData', this_linetype, defaults=[''])(*this_bonded_data)
        except (TypeError, TypeError) as error:
            # It failed, the function does not match the number of parameters
            raise TypeError('Could not understand {} line {}'.format(type_directive, bondedtype_string))
        else:
            index_tuple = tuple(this_bonded_data[:this_functionindex + 1])
            index_tuple_rev = tuple(this_bonded_data[:this_functionindex][::-1]
                                    + [this_bonded_data[this_functionindex]])

            if type_directive == 'dihedraltypes' and this_bonded_type.function == 9:
                if index_tuple in this_online_dict:
                    this_online_dict[index_tuple].append(this_bonded_type)
                    this_online_dict[index_tuple_rev].append(this_bonded_type)
                else:
                    this_online_dict[index_tuple] = [this_bonded_type]
                    this_online_dict[index_tuple_rev] = [this_bonded_type]

            else:
                this_online_dict[index_tuple] = this_bonded_type
                this_online_dict[index_tuple_rev] = this_bonded_type

        self.output_sequence.append('; {}\n'.format(bondedtype_string))

    def add_settle(self, settle_string, molecule_type):
        """ Reads a settles line

        :param str settle_string: virtual site line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate vsite to
        """

        this_settle_data = self.type_converter(settle_string)
        try:
            this_settle = self.__setttles_data(*this_settle_data)
        except ValueError as error:
            os_util.local_print('Error while parsing settles line {} in molecule {} with error {}'
                                ''.format(settle_string, molecule_type, error),
                                msg_verbosity=os_util.verbosity_level.error)
            raise TypeError('Could not understand settles line {}.'
                            ''.format(settle_string))
        else:
            molecule_type.settles_dict.append(this_settle)
            molecule_type.output_sequence.append(molecule_type.settles_dict[-1])

    def add_vsite2(self, vsite2_string, molecule_type):
        """ Reads a constraint line

        :param str vsite2_string: virtual site line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate vsite to
        """

        this_vsite2_data = self.type_converter(vsite2_string)
        try:
            this_vsite2 = self.__vsite2_data(*this_vsite2_data)
        except ValueError as error:
            os_util.local_print('Error while parsing virtual site line {} in molecule {} with error {}'
                                ''.format(vsite2_string, molecule_type, error),
                                msg_verbosity=os_util.verbosity_level.error)
            raise TypeError('Could not understand virtual site line {}.'
                            ''.format(vsite2_string))
        else:
            molecule_type.vsites2_dict.append(this_vsite2)
            molecule_type.output_sequence.append(molecule_type.vsites2_dict[-1])

    def add_vsite3(self, vsite3_string, molecule_type):
        """ Reads a vsite3 line

        :param str vsite3_string: vsite3 line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate virtual site to
        """

        this_vsite3_data = self.type_converter(vsite3_string)
        try:
            # Get function number
            function_number = int(this_vsite3_data[4])
        except ValueError as error:
            os_util.local_print('Error while parsing vsite3 line {} in molecule {} with error {}'
                                ''.format(vsite3_string, molecule_type, error),
                                msg_verbosity=os_util.verbosity_level.error)
            raise TypeError('Could not understand virtual site line {}.'
                            ''.format(vsite3_string))
        else:
            try:
                this_vsite3 = namedlist('Vsite3Data', self.__vsite3_fields[function_number],
                                        defaults=[''])(*this_vsite3_data)
            except (TypeError, KeyError) as error:
                os_util.local_print('Error while parsing vsite3 line {} in molecule {} with error {}'
                                    ''.format(vsite3_string, molecule_type, error),
                                    msg_verbosity=os_util.verbosity_level.error)
                raise TypeError('Could not understand virtual site line {}.'.format(vsite3_string))
            else:
                molecule_type.vsites3_dict.append(this_vsite3)
                molecule_type.output_sequence.append(molecule_type.vsites3_dict[-1])

    def add_vsite4(self, vsite4_string, molecule_type):
        """ Reads a constraint line

        :param str vsite4_string: virtual site line
        :param MoleculeTypeData molecule_type: MoleculeTypeData object to associate vsite to
        """

        this_vsite4_data = self.type_converter(vsite4_string)
        try:
            this_vsite4 = self.__vsite2_data(*this_vsite4_data)
        except ValueError as error:
            os_util.local_print('Error while parsing virtual site line {} in molecule {} with error {}'
                                ''.format(vsite4_string, molecule_type, error),
                                msg_verbosity=os_util.verbosity_level.error)
            raise TypeError('Could not understand virtual site line {}.'
                            ''.format(vsite4_string))
        else:
            molecule_type.vsites2_dict.append(this_vsite4)
            molecule_type.output_sequence.append(molecule_type.vsites2_dict[-1])

    def read_topology(self, topology_files, verbosity=0):
        """ Reads a topology file and saves data to a structure. Currently, imports and topology B fields are not
        processed.

        :param list topology_files: filename of the topology
        :param int verbosity: print extra info
        """

        supress_code = -1

        # Terms listed in Gromacs manual 5.5
        molecule_directives = {directive: code for code, directive in
                               enumerate(['atoms', 'bonds', 'pairs', 'pairs_nb', 'angles', 'dihedrals', 'exclusions',
                                          'constraints', 'settles', 'virtual_sites2', 'virtual_sites3',
                                          'virtual_sites4', 'virtual_sitesn', 'position_restraints',
                                          'distance_restraints', 'dihedral_restraints', 'orientation_restraints',
                                          'angle_restraints', 'angle_restraints_z'])}

        type_directives = {directive: code + 1000 for code, directive in enumerate(['bondtypes', 'pairtypes',
                                                                                    'angletypes', 'dihedraltypes',
                                                                                    'constrainttypes'])}

        full_topology_file = []
        for each_file in topology_files:
            each_file_data = os_util.read_file_to_buffer(each_file, return_as_list=True, die_on_error=True)
            full_topology_file.extend(each_file_data)

        # Main loop
        file_marker = None
        for each_index, raw_line in enumerate(full_topology_file):

            each_line = raw_line.lstrip().rstrip()

            # Ignore empty, comment lines or macros
            if (len(each_line) == 0) or (each_line[0] in [';', '#']):
                if each_line.startswith('#include'):
                    os_util.local_print('#include directive found in file {}. This is not supported. Suppressing '
                                        'import. Should the topology generation fail, double check your topology.'
                                        ''.format(topology_files),
                                        msg_verbosity=os_util.verbosity_level.warning,
                                        current_verbosity=verbosity)
                    raw_line = '; ' + raw_line + ' ; Suppressed, imports not allowed in ligand topology'
                try:
                    actual_moleculetype.output_sequence.append(raw_line)
                except (UnboundLocalError, AttributeError):
                    self.output_sequence.append(raw_line)

                continue

            # Test whenever a new declaration has started and update file_marker
            this_directive = re.match('(?:\[\s+)(.*)(?:\s+\])', each_line, flags=re.IGNORECASE)
            if this_directive is not None:
                cur_directive = this_directive.group(1).lower()
                if cur_directive == 'moleculetype':
                    os_util.local_print('Reading molecule type directive',
                                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                    self.output_sequence.append(self.MoleculeTypeData(self))
                    actual_moleculetype = self.output_sequence[-1]
                    self.molecules.append(self.output_sequence[-1])
                    actual_moleculetype.output_sequence.append(raw_line)
                    file_marker = 'moleculetype'
                elif cur_directive in ['system', 'molecules', 'defaults']:
                    # system also ends moleculetypes
                    os_util.local_print("Suppressing [ {} ] directive of file {}".format(this_directive.group(1),
                                                                                         topology_files),
                                        msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)
                    self.output_sequence.append('; {} Suppressing [ {} ] directive\n'
                                                ''.format(raw_line.rstrip('\n'), this_directive.group(1)))
                    actual_moleculetype = None
                    file_marker = supress_code
                elif cur_directive == 'atomtypes':
                    os_util.local_print("Reading directive atomtypes from {}".format(topology_files),
                                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                    file_marker = 'atomtypes'
                    self.output_sequence.append(raw_line)
                elif cur_directive in molecule_directives:
                    os_util.local_print("Reading directive {} in molecule {} of file {}"
                                        "".format(each_line, actual_moleculetype.name, topology_files),
                                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                    file_marker = molecule_directives[cur_directive]
                    actual_moleculetype.output_sequence.append(raw_line)
                elif cur_directive in type_directives:
                    os_util.local_print("Reading type directive {} from file {}"
                                        "".format(each_line, topology_files),
                                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                    file_marker = type_directives[cur_directive]
                    self.output_sequence.append('; {} Directive suppressed and all parameters inlined to molecules. '
                                                'Dual topology code does not support online parameters\n'
                                                ''.format(raw_line.rstrip('\n')))
                else:
                    os_util.local_print("Ignoring directive {} of file {}".format(each_line, topology_files),
                                        msg_verbosity=os_util.verbosity_level.debug, current_verbosity=verbosity)
                    file_marker = None
                    self.output_sequence.append(raw_line)

            else:
                try:
                    if file_marker == molecule_directives['atoms']:
                        self.add_atom(each_line, actual_moleculetype, verbosity=verbosity)
                    elif file_marker == molecule_directives['bonds']:
                        self.add_bond(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['pairs']:
                        self.add_pair(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['pairs_nb']:
                        self.add_pair_nb(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['angles']:
                        self.add_angle(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['dihedrals']:
                        self.add_dihedral(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['constraints']:
                        self.add_constraint(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['exclusions']:
                        self.add_exclusion(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['virtual_sites2']:
                        self.add_vsite2(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['virtual_sites3']:
                        self.add_vsite3(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['virtual_sites4']:
                        self.add_vsite4(each_line, actual_moleculetype)
                    elif file_marker == molecule_directives['settles']:
                        self.add_settle(each_line, actual_moleculetype)
                    elif file_marker == 'atomtypes':
                        self.add_atomtype(each_line, verbosity)
                    elif file_marker in type_directives.values():
                        # TODO: avoid this (maybe by not using codes and rather the actual strings?
                        type_str = list(type_directives.keys())[list(type_directives.values()).index(file_marker)]
                        self.add_bondedtype(each_line, type_directive=type_str)
                    elif file_marker == supress_code:
                        try:
                            # If we are inside a molecule type, append to it
                            actual_moleculetype.output_sequence.append('; {}'.format(raw_line))
                        except (UnboundLocalError, AttributeError):
                            # We are not
                            self.output_sequence.append('; {}'.format(raw_line))
                    elif file_marker == 'moleculetype':
                        this_line = each_line.split(';')[0].split()
                        if len(this_line) != 2:
                            error = 'Cannot parse moleculetype data'
                            os_util.local_print('Failed to parse line #{}. This is the line read:\n{}\nError was: {}'
                                                ''.format(each_index, raw_line, error),
                                                msg_verbosity=os_util.verbosity_level.error)
                            raise ValueError(error)
                        try:
                            assert int(this_line[1]) in [3, 4]
                        except ValueError:
                            error = 'Cannot parse moleculetype data'
                            os_util.local_print('Failed to parse line #{}. This is the line read:\n{}\nError was: {}'
                                                ''.format(each_index, raw_line, error),
                                                msg_verbosity=os_util.verbosity_level.error,
                                                current_verbosity=verbosity)
                            raise ValueError(error)
                        except AssertionError:
                            os_util.local_print('Non-bonded interactions exclusion read to be nrexcl = {}. Make sure '
                                                'that this is what you want.'.format(this_line[1]),
                                                msg_verbosity=os_util.verbosity_level.warning,
                                                current_verbosity=verbosity)

                        if not actual_moleculetype.name:
                            # Molecule name was not set, set it from this line
                            actual_moleculetype.name = this_line[0]
                            actual_moleculetype.name_line._nrexcl = int(this_line[1])
                            actual_moleculetype.output_sequence.append(actual_moleculetype.name_line)
                        else:
                            # Either the name is being redeclared or molecule name was set before reading the topology,
                            # which doesn't make sense.
                            os_util.local_print('Molecule name for molecule {} was already set but it was read again '
                                                'in line {}: "{}". This should not happen. Please, check your inputs.'
                                                ''.format(actual_moleculetype.name, each_index, raw_line),
                                                msg_verbosity=os_util.verbosity_level.error,
                                                current_verbosity=verbosity)
                            raise ValueError('Molecule name redeclared/set again')

                except KeyError as error:
                    os_util.local_print('Failed to parse line #{}. This is the line read:\n{}\nError was: {}'
                                        ''.format(each_index, raw_line, error),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise KeyError(error)

    def __str__(self, style='full'):
        """ Returns formatted topology

        :param str style: print this style (full: full topology (default); atomtypes: only atomtypes section; itp:
        suppress atomtypes section)
        :rtype: str
        """

        if style == 'full':
            return_str = []
            for each_element in self.output_sequence:
                if isinstance(each_element, tuple(self.__atomtype_dict.values())):
                    return_str.append(('{:<15} ' * len(each_element)).format(*[a for a in each_element]) + '\n')
                else:
                    return_str.append(each_element.__str__())
            return ''.join(return_str)

        elif style == 'atomtypes':
            return_str = ' [ atomtypes ]\n;\n'
            for each_atomtype in self.atomtype_dict.values():
                if isinstance(each_atomtype, str):
                    return_str += each_atomtype
                else:
                    return_str += '{}\n'.format(('{:<15} ' * len(each_atomtype)).format(*[a for a in each_atomtype]))

            return return_str

        elif style == 'itp':
            return_str = []
            for each_element in self.output_sequence:
                if isinstance(each_element, tuple(self.__atomtype_dict.values())) or \
                        re.match('(?:\[\s+)(atomtypes)(?:\s+\])', str(each_element), flags=re.IGNORECASE) is not None:
                    continue
                else:
                    return_str.append(each_element.__str__())
            return ''.join(return_str)

        else:
            raise ValueError('Unrecognizable printing style "{}"'.format(style))

    @property
    def num_molecules(self):
        """ Get the number of molecule types read

        :rtype: int
        """

        return len(self.molecules)


class DualTopologyData(TopologyData):
    """ Extends TopologyData to allows lambda scaling between states A and B
    """

    __dualatomtype_data = namedlist('DualAtomTypeData', ['name', 'V_a', 'W_a', 'V_b', 'W_b'],
                                    defaults=[float('inf'), float('inf'), float('inf'), float('inf')],
                                    types=[str, float, float, float, float])
    __dualatom_data = namedlist('DualAtomData', ['name', 'charge_a', 'charge_b'],
                                defaults=[float('inf'), float('inf')], types=[str, float, float])

    allowed_scaling = ('vdwA', 'vdwB', 'coulA', 'coulB', 'coul_const', 'vdw_const')

    def __init__(self, topology_file='', lambda_table=None, charge_interpolation='linear'):
        self._lambda_table = None
        if lambda_table is not None:
            self.lambda_table = lambda_table

        if charge_interpolation in ['linear', 'sigmoid']:
            self.charge_interpolation = charge_interpolation
        else:
            raise ValueError("charge_interpolation must be one of 'linear', 'sigmoid'")

        self.atoms_const = []
        self.atoms_const_vdw = []
        self.atoms_A = []
        self.atoms_B = []
        self.dualatomtype_data_dict = OrderedDict()
        self.dualatom_data_dict = OrderedDict()
        self._current_lambda_value = None
        super().__init__(topology_file)

    def add_dual_atom_add_atomtype(self, new_name, input_atom, input_atomtype, mol_region, q_a=0.0, q_b=0.0,
                                   vdw_v_a=0.0, vdw_w_a=0.0, vdw_v_b=0.0, vdw_w_b=0.0, verbosity=0):
        """ Adds a dual atom, sets atom data and adds an correspondingly atomtype. Tries to be smart about charge and
        VdW parameters.

        :param str new_name: new name of the atom
        :param AtomData input_atom: read atom data from this, and update properties in place
        :param AtomTypeData input_atomtype: copy atomtype data from this
        :param str mol_region: atom belongs to regions A, B, or constant region, choices = ['A', 'B', 'const']
        :param [float, None] q_a: charge at state A; if None, will read from input_atom.q_e
        :param [float, None] q_b: charge at state B; if None, will read from input_atom.q_e
        :param [float, None] vdw_v_a: V VdW parameter at state A; if None will read from input_atomtype.V
        :param [float, None] vdw_w_a: W VdW parameter at state A; if None will read from input_atomtype.W
        :param [float, None] vdw_v_b: V VdW parameter at state B; if None, will read from input_atomtype.V
        :param [float, None] vdw_w_b: W VdW parameter at state B; if None, will read from input_atomtype.W
        :param int verbosity: verbosity level
        """

        if q_a is None:
            q_a = input_atom.q_e
        if vdw_v_a is None:
            vdw_v_a = input_atomtype.V
        if vdw_w_a is None:
            vdw_w_a = input_atomtype.W

        if q_b is None:
            q_b = input_atom.q_e
        if vdw_v_b is None:
            vdw_v_b = input_atomtype.V
        if vdw_w_b is None:
            vdw_w_b = input_atomtype.W

        input_atom.atom_name = new_name
        self.add_dual_atom(new_name, mol_region=mol_region, charge_a=q_a, charge_b=q_b)

        # Adds an atomtype for this atom (this is also used to scale the VdW parameters)
        new_atomtype = deepcopy(input_atomtype)
        input_atom.atom_type = new_name
        new_atomtype.atom_type = new_name

        try:
            last_atomtype_index = self.output_sequence.index(next(reversed(self.atomtype_dict.values())))
        except StopIteration:
            # atomtype_dict is empty, so no atomtypes were read. Insert an atomtypes directive at the beginning of file
            # and a correspondingly atom type
            self.output_sequence.insert(0, new_atomtype)
            self.output_sequence.insert(0, '[ atomtypes ]')
        else:
            self.output_sequence.insert(last_atomtype_index + 1, new_atomtype)

        self.atomtype_dict[new_name] = new_atomtype
        self.add_dual_atomtype(new_atomtype.atom_type, vdw_v_a, vdw_w_a, vdw_v_b, vdw_w_b)

    def add_dual_atom(self, name, mol_region, charge_a=None, charge_b=None):
        """ Stores charges in topologies A and B for atom name

        :param str name: atom name
        :param str mol_region: atom belongs to regions A, B, or constant region, choices = ['A', 'B', 'const']
        :param float charge_a: charge in topology A
        :param float charge_b: charge in topology B
        """
        if name in self.dualatom_data_dict:
            os_util.local_print('Atom {} already exists in dualatom_data_dict! Please check your input'.format(name),
                                msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(1)
        else:
            self.dualatom_data_dict[name] = self.__dualatom_data(name)
            if charge_a is not None:
                self.dualatom_data_dict[name].charge_a = charge_a
            if charge_b is not None:
                self.dualatom_data_dict[name].charge_b = charge_b

        # Add atom to the correspondingly mol region
        if mol_region not in ['A', 'B', 'const', 'const_vdw']:
            raise ValueError('Molecular region {} not allowed, please select between "A", "B", "const"'
                             ''.format(mol_region))
        self.__getattribute__('atoms_{}'.format(mol_region)).append(name)

    def add_dual_atomtype(self, name, v_a=None, w_a=None, v_b=None, w_b=None):
        """ Stores VdW parameters in topologies A and B for atomtype name

        :param str name: atom type
        :param float v_a: V in topology A
        :param float w_a: W in topology A
        :param float v_b: V in topology B
        :param float w_b: W in topology B
        """
        if name in self.dualatomtype_data_dict:
            os_util.local_print('Atom {} already exists in dualatomtype_data_dict! Please check your input'
                                ''.format(name), msg_verbosity=os_util.verbosity_level.error)
            raise SystemExit(1)
        else:
            self.dualatomtype_data_dict[name] = self.__dualatomtype_data(name)
            for each_var, each_value in {'V_a': v_a, 'W_a': w_a, 'V_b': v_b, 'W_b': w_b}.items():
                if each_value is not None:
                    setattr(self.dualatomtype_data_dict[name], each_var, each_value)

    @property
    def lambda_count(self):
        if self.lambda_table is None:
            return None
        else:
            return len(self.lambda_table['coulA'])

    @property
    def lambda_table(self):
        return self._lambda_table

    @lambda_table.setter
    def lambda_table(self, lambda_table):
        if not isinstance(lambda_table, dict):
            raise TypeError("Unsupported operand type(s) for lambda_table: '{}' and '{}'"
                            "".format(type(lambda_table), dict))
        if any(each_term not in lambda_table for each_term in ['vdwA', 'vdwB', 'coulA', 'coulB']):
            raise ValueError("lambda_table must contain 'vdwA', 'vdwB', 'coulA' and 'coulB'")

        # Default constant part scaling to use the interpolation method
        for default_term in ['coul_const', 'vdw_const']:
            if default_term not in lambda_table:
                lambda_table[default_term] = [1.0] * len(lambda_table['vdwA'])

        self._lambda_table = lambda_table

    def get_charge_scaling(self, lambda_value):
        """ Calculates scaling lambda factor

        :param int lambda_value: actual lambda value
        :rtype: float
        """

        if self.charge_interpolation == 'linear':
            # Linear scaling
            return 1.0 - (1.0 / (self.lambda_count - 1)) * lambda_value
        elif self.charge_interpolation == 'sigmoid':
            # Hyperbolic scaling
            if lambda_value == 0:
                return 0
            elif lambda_value == self.lambda_count:
                return 1
            else:
                from numpy import tanh
                return (tanh(-3.0 + (6.0 / (self.lambda_count - 1)) * lambda_value) / 2.0) + 0.5
        else:
            raise ValueError("Unknown charge interpolation {}".format(self.charge_interpolation))

    def set_lambda_state(self, lambda_value, verbosity=0):
        """Sets lambda to lambda_value, affecting all dual atoms

        Parameters
        ----------
        lambda_value : int
            Use this lambda value column from lambda_table
        verbosity : int
            Set the verbosity level
        """
        if self.lambda_table is None:
            raise ValueError("lambda_table was not set, set it before setting a lambda_state")
        self._current_lambda_value = lambda_value
        # Scales VdW for dual atoms atomtypes
        for atom_name, atom_data in self.atomtype_dict.items():
            if atom_name in self.dualatomtype_data_dict:
                if atom_name in self.atoms_A + self.atoms_B:
                    # Scale atoms in region A or B mixing VdW from each state
                    atom_data.V = self.dualatomtype_data_dict[atom_name].V_a * self.lambda_table['vdwA'][lambda_value] \
                                  + self.dualatomtype_data_dict[atom_name].V_b * self.lambda_table['vdwB'][lambda_value]
                    atom_data.W = self.dualatomtype_data_dict[atom_name].W_a * self.lambda_table['vdwA'][lambda_value] \
                                  + self.dualatomtype_data_dict[atom_name].W_b * self.lambda_table['vdwB'][lambda_value]
                elif atom_name in self.atoms_const:
                    # Atom is in the constant region and VdW in both endpoints are the same. Scale state A data using
                    # vdw_const
                    atom_data.V = self.dualatomtype_data_dict[atom_name].V_a \
                                  * self.lambda_table['vdw_const'][lambda_value]
                    atom_data.W = self.dualatomtype_data_dict[atom_name].W_a \
                                  * self.lambda_table['vdw_const'][lambda_value]
                elif atom_name in self.atoms_const_vdw:
                    # Atom is in the constant region, but VdW perturbation is allowed. Interpolate states A and B using
                    # get_charge_scaling
                    if (self.dualatomtype_data_dict[atom_data.atom_type].V_b == 0.0
                            or self.dualatomtype_data_dict[atom_data.atom_type].W_b):
                        # Should this throw an error?
                        os_util.local_print('Null Van der Waals parameters for atom {}, but atom is in const_vdw '
                                            'region, where VdW sacling is allowed. This is possibly wrong. Make sure '
                                            'your topology is correct.'.format(atom_name),
                                            msg_verbosity=os_util.verbosity_level.warning, current_verbosity=verbosity)

                    atom_data.V = self.dualatomtype_data_dict[atom_data.atom_type].V_a \
                                  * self.get_charge_scaling(lambda_value) \
                                  + self.dualatomtype_data_dict[atom_data.atom_type].V_b \
                                  * (1.0 - self.get_charge_scaling(lambda_value))
                    atom_data.W = self.dualatomtype_data_dict[atom_data.atom_type].W_a \
                                  * self.get_charge_scaling(lambda_value) \
                                  + self.dualatomtype_data_dict[atom_data.atom_type].W_b \
                                  * (1.0 - self.get_charge_scaling(lambda_value))

                    # Optionally scale the constant region VdW
                    atom_data.V *= atom_data.V * self.lambda_table['vdw_const'][lambda_value]
                    atom_data.W *= atom_data.W * self.lambda_table['vdw_const'][lambda_value]
                else:
                    os_util.local_print('Atom {} not found in internal indexes: atoms_A={}, atoms_B={}, '
                                        'atoms_const={}. Cannot continue.'
                                        ''.format(atom_name,self.atoms_A, self.atoms_B, self.atoms_const),
                                        msg_verbosity=os_util.verbosity_level.error, current_verbosity=verbosity)
                    raise KeyError('Atom {} not found'.format(atom_name))

        # Scales charges for dual atoms atomtypes
        for atom_idx, atom_data in self.molecules[0].atoms_dict.items():
            if atom_data.atom_type in self.dualatom_data_dict:
                # 1. Find out if the atom is in constant part or only in A or B.
                # FIXME: if a constant atom with q_e = 0 in one of the topology A or B is present, this will
                #  incorrectly scale it using the same rules as topology A or B
                if atom_data.atom_type in self.atoms_B:
                    # Atom is present only in B
                    atom_data.q_e = self.dualatom_data_dict[atom_data.atom_type].charge_b \
                                    * self.lambda_table['coulB'][lambda_value]
                elif atom_data.atom_type in self.atoms_A:
                    # Atom is present only in A
                    atom_data.q_e = self.dualatom_data_dict[atom_data.atom_type].charge_a \
                                    * self.lambda_table['coulA'][lambda_value]
                else:
                    # Atom in constant region, scale according to charge_interpolation
                    # charge = scaling_lambda * charge_a + (1.0 - scaling_lambda) * charge_b
                    atom_data.q_e = self.dualatom_data_dict[atom_data.atom_type].charge_a \
                                    * self.get_charge_scaling(lambda_value) \
                                    + self.dualatom_data_dict[atom_data.atom_type].charge_b \
                                    * (1.0 - self.get_charge_scaling(lambda_value))

                    # Optionally scale the constant region as well
                    atom_data.q_e *= self.lambda_table['coul_const'][lambda_value]

    def __str__(self, style='full'):
        if self._current_lambda_value is None:
            raise RuntimeError("Call to __str__ before calling set_lambda_state")
        return super().__str__(style=style)


class MCSResult(dict):
    """ Stores and MCS results and covert an MCS string transparently
    """

    def __init__(self, smarts_string, num_atoms=None, num_bonds=None, canceled=None):
        self._numAtoms = num_atoms
        self._numBonds = num_bonds
        if canceled:
            self.smartsString = ''
            self._numAtoms = 0
            self._numBonds = 0
            self.canceled = True
        elif not smarts_string:
            self.smartsString = ''
            self._numAtoms = 0
            self._numBonds = 0
            self.canceled = True
        else:
            self.smartsString = smarts_string
            self.canceled = False

    @property
    def numAtoms(self):
        if self._numAtoms:
            return self._numAtoms
        else:
            from rdkit.Chem import MolFromSmarts
            this_mol = MolFromSmarts(self.smartsString)
            self._numAtoms = this_mol.GetNumAtoms()
            self._numBonds = this_mol.GetNumBonds()
            return self._numAtoms

    @property
    def numBonds(self):
        if '_numBonds' in self and self._numBonds:
            return self._numBonds
        else:
            from rdkit.Chem import MolFromSmarts
            this_mol = MolFromSmarts(self.smartsString)
            self._numAtoms = this_mol.GetNumAtoms()
            self._numBonds = this_mol.GetNumBonds()
            return self._numBonds

    @property
    def canceled(self):
        try:
            return self._canceled
        except AttributeError:
            if self.smartsString:
                self._canceled = False
            else:
                self._canceled = True
            return self._canceled

    @canceled.setter
    def canceled(self, value):
        self._canceled = value

    def __str__(self):
        if not self.canceled:
            str_repr = '[MCS: {}, numAtoms: {}, numBonds: {}]'.format(self.smartsString, self.numAtoms, self.numBonds)
        else:
            str_repr = '[MCS: '', canceled = True]'
        return str_repr


class PDBFile:

    class PDBModel(list):
        """ Index models in the PDB data """

        def __init__(self, model_num=1):
            super(PDBFile.PDBModel, self).__init__()
            self.model_num = model_num

        def __str__(self):
            output_data = []
            for each_line in self:
                if isinstance(each_line, PDBFile.PDBAtom):
                    output_data.append(each_line.to_line())
                else:
                    output_data.append(each_line)

            return output_data

    class PDBResidue(list):
        """ Index atoms belonging to the same residue """
        def __init__(self, resname=''):
            super(PDBFile.PDBResidue, self).__init__()
            self.resname = resname

        def append(self, newatom) -> None:
            super(PDBFile.PDBResidue, self).append(newatom)
            if not self.resname:
                self.resname = newatom.resname
            newatom.residue = self

        def __str__(self):
            return '<Residue {} containing {} atoms>'.format(self.resname, len(self))

        @property
        def first_line(self):
            return self[0].line_num

        @property
        def lines(self):
            return [i.line_num for i in self]

        def get_atom_names(self):
            return [i.name.replace(' ', '') for i in self]

        def update_atoms(self):
            for each_atom in self:
                each_atom.resname = self.resname

    class PDBAtom:
        """ A class to store PDB atoms, reads atom lines following the PDB structure:
        (from http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM)
        # COLUMNS        DATA  TYPE    FIELD        DEFINITION
        # -------------------------------------------------------------------------------------
        #  1 -  6        Record name   "ATOM  "
        #  7 - 11        Integer       serial       Atom  serial number.
        # 13 - 16        Atom          name         Atom name.
        # 17             Character     altLoc       Alternate location indicator.
        # 18 - 20        Residue name  resName      Residue name.
        # 22             Character     chainID      Chain identifier.
        # 23 - 26        Integer       resSeq       Residue sequence number.
        # 27             AChar         iCode        Code for insertion of residues.
        # 31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
        # 39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
        # 47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
        # 55 - 60        Real(6.2)     occupancy    Occupancy.
        # 61 - 66        Real(6.2)     tempFactor   Temperature  factor.
        # 77 - 78        LString(2)    element      Element symbol, right-justified.
        # 79 - 80        LString(2)    charge       Charge  on the atom.
        """
        def __init__(self, atom_line, line_num):
            self.line_num = line_num
            self.residue = None
            atom_line = atom_line.rstrip().ljust(80)
            self.record_name = atom_line[0:6]
            self.serial = int(atom_line[6:11])
            self.name = atom_line[12:16]
            self.alt_loc = atom_line[16]
            self.resname = atom_line[17:20]
            self.chain = atom_line[21]
            self.res_seq = int(atom_line[22:26])
            self.i_code = atom_line[26]
            self.coords = [float(atom_line[30:38]), float(atom_line[38:46]), float(atom_line[46:54])]
            self.occupancy = float(atom_line[54:60])
            self.beta_temp = float(atom_line[60:66])
            self.element = atom_line[76:78]
            self.charge = atom_line[78:80]
            if self.charge.strip() == '':
                self.charge = ''

        def to_line(self):
            ret_line = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          '\
                       '{:>2s}{:2s}\n'.format(self.record_name, int(self.serial % 1e5), self.name, self.alt_loc,
                                              self.resname, self.chain, self.res_seq, self.i_code, *self.coords,
                                              self.occupancy, self.beta_temp, self.element, self.charge)
            return ret_line

    def __init__(self, input_file):
        """ Very simple class to read PDB files """
        self.atoms = []
        self.residues = []
        self.models = []
        self.file_lines = []
        self.input_file = input_file

        self.read_file(self.input_file)

    def read_file(self, input_file):
        """ Reads a PDB file, parsing atoms and grouping residues
        """
        if isinstance(input_file, str):
            self.file_lines = os_util.read_file_to_buffer(input_file, die_on_error=True, return_as_list=True,
                                                          error_message='Failed to read PDB file {}.'
                                                                        ''.format(input_file))
        else:
            self.file_lines = input_file

        new_res = True
        current_model = None
        for line_num, each_line in enumerate(self.file_lines):
            if len(each_line) > 54 and (each_line[0:4] == 'ATOM' or each_line[0:6] == 'HETATM'):
                try:
                    atom_serial = int(each_line[6:11])
                except ValueError as err:
                    os_util.local_print('Failed to convert atom number {} to an integer at line {} of pdb file {}.'
                                        ''.format(each_line[6:11], line_num, input_file),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise ValueError(err)
                this_atom = PDBFile.PDBAtom(each_line, line_num)
                self.atoms.append(this_atom)
                if new_res or this_atom.res_seq != self.residues[-1][-1].res_seq:
                    this_res = PDBFile.PDBResidue(this_atom.resname)
                    this_res.append(this_atom)
                    self.residues.append(this_res)
                    new_res = False
                else:
                    self.residues[-1].append(this_atom)
                if current_model is not None:
                    current_model.append(this_atom)
                self.file_lines[line_num] = this_atom

            elif len(each_line) >= 5 and each_line[0:5] == 'MODEL':
                try:
                    model_num = int(each_line[5:])
                except ValueError as err:
                    os_util.local_print('Failed to convert model number {} to an integer at line {} of pdb file {}.'
                                        ''.format(each_line[5:], line_num, input_file),
                                        msg_verbosity=os_util.verbosity_level.error)
                    raise ValueError(err)
                else:
                    current_model = PDBFile.PDBModel(model_num=model_num)
                    self.models.append(current_model)
                    current_model.append(each_line)

            else:
                if current_model is not None:
                    current_model.append(each_line)

            if len(each_line) >= 6 and each_line[0:6] == 'ENDMDL':
                current_model = None
            if len(each_line) >= 3 and each_line[0:3] == 'TER':
                new_res = True

    def update_atom_lines(self):
        """ Updates the serial field [6:11] of all atoms
        """
        for n, each_atom in enumerate(self.atoms):
            each_atom.serial = n + 1
            self.file_lines[each_atom.line_num] = each_atom.to_line()

    def update_atoms_from_lines(self):
        """ Updates the atoms line_num from self.file_lines
        """
        tmp_atom_list = []
        for n, each_line in enumerate(self.file_lines):
            if isinstance(each_line, PDBFile.PDBAtom):
                each_line.line_num = n
                tmp_atom_list.append(each_line)
        self.atoms = tmp_atom_list

    def update_resseq(self, first_res=None):
        """Updates the atoms' res_seq from self sequence

        Parameters
        ----------
        first_res : int or None
            Use this value as the first residue seq number. If None, the number of the first residue will be used. Pass
            1 here to get a sequence starting from 1.
        """
        if first_res is None:
            first_res = self.atoms[0].res_seq

        current_res_seq = None
        # n starts as X and will be updated to X+1 when current_res_seq is None, so n must be first_res - 1
        n = first_res - 1
        for each_atom in self.atoms:
            # Which is the current res_seq being read? If we find a new res, update n
            if current_res_seq is None or current_res_seq != each_atom.res_seq:
                current_res_seq = each_atom.res_seq
                n += 1
            each_atom.res_seq = n

    def to_file(self, output_file=None, output_connect=False, verbosity=0):
        """Dumps PDB structure to a PDB file

        Parameters
        ----------
        output_file : str or None
            Output file, will be overwritten if exists. If None, will return the data as list.
        output_connect : bool
            Also write PDB CONECT records. Note: no new CONECTs will be generated, only ones present in the read file
            would be written
        verbosity : int
            Sets verbosity level

        Returns
        -------
        list
            PDB-data, returned if output_file is None
        """
        output_data = []
        for each_line in self.file_lines:
            if (not output_connect) and isinstance(each_line, str) and each_line.startswith('CONECT'):
                continue

            if isinstance(each_line, PDBFile.PDBAtom):
                output_data.append(each_line.to_line())
            else:
                output_data.append(each_line)

        if output_file:
            with open(output_file, 'w') as fh:
                fh.writelines(output_data)
        else:
            return output_data

    def __str__(self):
        fname = self.input_file if isinstance(self.input_file, str) else '<DataStream>'
        description = '<PDBFile {} with {} residues and {} atoms>'.format(fname, len(self.residues), len(self.atoms))
        return description


# mergedtopologies_class = namedlist('MergedTopologies', ['dual_topology', 'dual_molecule', 'mcs', 'common_core_mol',
#                                                         'molecule_a', 'topology_a', 'molecule_b', 'topology_b',
#                                                         'dual_molecule_name', 'delta_charge'])
class MergedTopologies(Namespace):
    """Holds topology data for merged molecules and topologies
    """

    def __init__(self, dual_topology, dual_molecule, mcs, common_core_mol, molecule_a, topology_a, molecule_b,
                 topology_b, dual_molecule_name, delta_charge=0):
        super().__init__()
        for k, v in zip(['dual_topology', 'dual_molecule', 'mcs', 'common_core_mol', 'molecule_a', 'topology_a',
                         'molecule_b', 'topology_b', 'dual_molecule_name', 'delta_charge'],
                        [dual_topology, dual_molecule, mcs, common_core_mol, molecule_a, topology_a, molecule_b,
                         topology_b, dual_molecule_name, delta_charge]):
            self[k] = v

    def to_pdb_block(self, molecule_name=None, confId=-1, verbosity=0):
        """ Returns object as a PDB block

        Parameters
        ----------
        molecule_name : str or None
            Use this as molecule name (this will not overwrite data, only output PDB will be affected)
        confId : int
            Selects which conformation to output (-1 = default)
        verbosity : int
            Sets verbosity level

        Returns
        -------
        str
            PDB block representation of the molecule
        """
        from rdkit.Chem import MolToPDBBlock
        from mol_util import adjust_query_properties

        # Prepare PDB, do not print CONECT records
        flavor = (2 | 8)
        molecule_a_pdb = MolToPDBBlock(self.molecule_a, confId=confId, flavor=flavor).split('\n')

        molecule_b_pdb = [each_line
                          for each_line in MolToPDBBlock(self.molecule_b, confId=confId, flavor=flavor).split('\n')
                          if each_line.find('HETATM') == 0 or each_line.find('ATOM') == 0]
        # Suppress END record and empty line after molecule A
        # TODO edit COMPND record?
        return_list = molecule_a_pdb[:-2]

        core_structure = adjust_query_properties(self.common_core_mol, verbosity=verbosity)

        # Compute atoms present only in topology B
        only_in_b = [each_atom.GetIdx()
                     for each_atom in self.molecule_b.GetAtoms()
                     if each_atom.GetIdx() not in
                     self.molecule_b.GetSubstructMatch(core_structure)]
        os_util.local_print('These are the atoms present only in topology B: {}'.format(only_in_b),
                            msg_verbosity=os_util.verbosity_level.info, current_verbosity=verbosity)

        # Iterate over molecule_b_pdb, adding atoms present only in B to return_list. Use Idx + 1 to match PDB numbering
        for each_atom, new_index in zip((each_line for idx, each_line in enumerate(molecule_b_pdb) if idx in only_in_b),
                                        range(self.molecule_a.GetNumAtoms() + 1,
                                              self.molecule_a.GetNumAtoms() + len(only_in_b) + 1)):
            new_atom_line = each_atom[:6] + '{:>5}'.format(new_index) + each_atom[11:]
            return_list.append(new_atom_line)

        if molecule_name:
            temp_pdb_data = []
            for each_line in return_list:
                if each_line.find('HETATM') == 0 or each_line.find('ATOM') == 0:
                    each_line = each_line[:17] + '{:>3}'.format(molecule_name) + each_line[20:]
                temp_pdb_data.append(each_line)
            return_list = temp_pdb_data

        return_list.extend(['TER', '\n'])

        return '\n'.join(return_list)

    def __repr__(self):
        return '< MergedTopologies object (name="{}") >'.format(self.dual_molecule_name)


class MolecularTopologies(Namespace):
    """Holds topology data for single molecules and topologies
    """

    def __init__(self, molecule_a, topology_a, dual_molecule_name):
        super().__init__()
        for k, v in zip(['dual_topology', 'dual_molecule', 'molecule_a', 'topology_a', 'dual_molecule_name'],
                        [topology_a, molecule_a, molecule_a, topology_a, dual_molecule_name]):
            self[k] = v

    def to_pdb_block(self, molecule_name=None, confId=-1, verbosity=0):
        """ Returns object as a PDB block

        Parameters
        ----------
        molecule_name : str or None
            Use this as molecule name (this will not overwrite data, only output PDB will be affected)
        confId : int
            Selects which conformation to output (-1 = default)
        verbosity : int
            Sets verbosity level

        Returns
        -------
        str
            PDB block representation of the molecule
        """
        from rdkit.Chem import MolToPDBBlock

        # Prepare PDB, do not print CONECT records
        flavor = (2 | 8)
        return_list = MolToPDBBlock(self.molecule_a, confId=confId, flavor=flavor).split('\n')[:-2]

        if molecule_name:
            temp_pdb_data = []
            for each_line in return_list:
                if each_line.find('HETATM') == 0 or each_line.find('ATOM') == 0:
                    each_line = each_line[:17] + '{:>3}'.format(molecule_name) + each_line[20:]
                temp_pdb_data.append(each_line)
            return_list = temp_pdb_data

        return_list.extend(['TER', '\n'])

        return '\n'.join(return_list)

    def __repr__(self):
        return '< MergedTopologies object (name="{}") >'.format(self.dual_molecule_name)


# Class to hold coion data, used for charge perturbations
coion_class = namedlist('CoIonData', ['dual_topology', 'structure_data', 'delta_charge', 'ion_name', 'water_o_atom',
                                      'coion_topology_files'])
