"""
Written by Zitong Li for COMP9418 assignment 2.
"""

import copy
import re
import math
from collections import OrderedDict as odict
from itertools import product

import numpy as np
import pandas as pd
from graphviz import Digraph, Graph
from tabulate import tabulate


class GraphicalModel:
    def __init__(self):
        self.net = dict()
        self.factors = dict()
        self.outcomeSpace = dict()
        self.node_value = dict()

    ####################################
    ######################################
    # Representation
    ########################################
    ##########################################
    def load(self, FileName):
        """
        Load and initiate model from file
        input:
            FileName: String
        """
        self.__init__()
        with open(FileName, 'r') as f:
            content = f.read()
            node_pattern = re.compile(
                r'node (.+) \n\{\n  states = \( \"(.+)\" \);\n\}')
            potential_pattern = re.compile(
                r'potential \( (.+) \) \n\{\n  data = \((.+)\)[ ]?;\n\}')

            nodes_records = node_pattern.findall(content)
            data_records = potential_pattern.findall(content)

            for record in nodes_records:
                outcome = tuple(re.split(r'\" \"', record[1]))
                self.insert(record[0], outcome)
            for record in data_records:
                splits = record[0].split(' | ')
                node = splits[0]
                parents = []
                if len(splits) > 1:
                    parents = list(reversed(splits[1].split()))
                data = [float(i) for i in re.findall(
                    r'[0-1][.][0-9]+', record[1])]
                self.factorize(node, data, parents)

    def connect(self, father, child):
        """
        Connect Two nodes.
        Inputs:
            father: String, name of the father node
            child:  String, name of the child node
        """
        if father in self.net and child in self.net and child not in self.net[father]:
            self.net[father].append(child)

    def disconnect(self, father, child):
        """
        Disconnect Two nodes.
        Inputs:
            father: String, name of the father node
            child:  String, name of the child node
        """
        if father in self.net and child in self.net and child in self.net[father]:
            self.net[father].remove(child)

    def factorize(self, node, data, parents=[]):
        """
        Specify probabilities for a node.
        data is a 1-d array or a simple list.
        Inputs:
            node: Sting, the node you want to specify
            data: 1-D array like, the CPT of the node
            parents: list of strings, parents of the node
        """
        dom = parents + [node]
        dom = tuple(dom)
        for parent in parents:
            self.connect(parent, node)

        self.factors[node] = {'dom': dom, 'table': odict()}
        outcome_product = product(*[self.outcomeSpace[node] for node in dom])
        assert np.prod([len(self.outcomeSpace[node])
                        for node in dom]) == len(data), 'CPT length illegal'
        for i, combination in enumerate(outcome_product):
            self.factors[node]['table'][combination] = data[i]

    def insert(self, Name, Outcome):
        """
        simply insert a node to the graph
        Inputs:
            Outcome: a 1-D array-like, outcome space of this node
            Name:    String, the name of the node
        """
        if Name not in self.net:
            self.net[Name] = []
            self.outcomeSpace[Name] = Outcome
        else:
            print(f'Already have node {Name}')

    def remove(self, node):
        if node in self.net:
            if node in self.factors:
                for child in self.net[node]:
                    if node in self.factors[child]['dom']:
                        self.sum_out(child, node)
                self.factors.pop(node)
            self.net.pop(node)
            self.outcomeSpace.pop(node)
            for other_node in self.net:
                if node in self.net[other_node]:
                    self.net[other_node].remove(node)

    def sum_out(self, node, victim):
        """
        sum out the victim in the factor of node
        Inputs:
            node: String, name of the node
            victim:  String, name of the node to be sum out
        """
        assert victim in self.factors[node]['dom'], 'the node to sum out is not one of the parents'
        f = self.factors[node]
        new_dom = list(f['dom'])
        new_dom.remove(victim)
        table = list()
        for entries in product(*[self.outcomeSpace[node] for node in new_dom]):
            s = 0
            for val in self.outcomeSpace[victim]:
                entriesList = list(entries)
                entriesList.insert(f['dom'].index(victim), val)
                p = f['table'][tuple(entriesList)]
                s = s + p
            table.append((entries, s))
        self.factors[node] = {'dom': tuple(new_dom), 'table': odict(table)}

    def save(self, fileName):
        """
        save the graph to a file
        Inputs:
            fileNamea: String, the path of the file you want to save to.
        """
        f = open(fileName, 'w')
        f.write('net\n{\n}\n')

        # first node domain part
        for node, values in self.outcomeSpace.items():
            outcome = " ".join(
                ['"' + value + '"' for value in values])
            text = 'node %s \n{\n  states = ( %s );\n}\n' % (node, outcome)
            f.write(text)

        # add data
        for node, factor in self.factors.items():
            potential = factor['dom'][-1]
            data = " ".join([str(_) for _ in factor['table'].values()])
            if len(factor['dom']) > 1:
                parents = list(factor['dom'][:-1])
                parents.reverse()
                potential += ' | ' + " ".join(parents)
            text = 'potential ( %s ) \n{\n  data = ( %s );\n}\n' % (
                potential, data)
            f.write(text)
        f.close()

    def printFactor(self, node):
        """
        print the factor table of the node
        """
        f = self.factors[node]
        table = list()
        for key, item in f['table'].items():
            k = list(key)
            k.append(item)
            table.append(k)
        dom = list(f['dom'])
        dom.append('Pr')
        print(tabulate(table, headers=dom, tablefmt='orgtbl'))

    def showGraph(self):
        """
        Visualize the net graph.
        """
        dot = Digraph()
        dot.attr(overlap="False", splines="True")
        for node, children in self.net.items():
            dot.node(node)
            for child in children:
                dot.edge(node, child)
        return dot

    ####################################
    ######################################
    # Pruning and pre-processing techniques for inference
    ########################################
    ##########################################
    def prune(self, query, **evidences):
        """
        Prune the graph based of the query vcariables and evidences
        Inputs:
            query: list of strings, the query variables
            evidences: dictionary, key: node, value: outcome of the node
        Outputs:
            a new graph
        """
        evi_vars = list(evidences.keys())
        qe = set(query + evi_vars)
        assert all([_ in self.net for _ in qe])
        newG = copy.deepcopy(self)
        all_deleted = 0
        # prune nodes
        while not all_deleted:
            all_deleted = 1
            W = set()
            for node, children in newG.net.items():
                if node not in qe and not children:
                    W.add(node)
                    all_deleted = 0
            for leaf in W:
                newG.remove(leaf)
            # clear the child who have been deleted
            for node, children in newG.net.items():
                newG.net[node] = [_ for _ in children if _ not in W]

        netcopy = copy.deepcopy(newG.net)
        for node in evidences:
            netcopy[node] = []

        reachable_from_q = self.spread(self.make_undirected(netcopy), query)
        nodes = list(newG.net.keys())
        for node in nodes:
            if node not in qe | reachable_from_q:
                newG.remove(node)

        # prune edge
        for node, value in evidences.items():
            for child in newG.net[node]:
                newG.factors[child] = self.update(
                    newG.factors[child], node, value, newG.outcomeSpace)
            newG.net[node] = []
            newG.node_value[node] = value

        return newG

    @staticmethod
    def update(factor, node, value, outcomeSpace):
        """
        Specify a value to a node.
        Inputs:
            factor: the factor of the node
            node: the node to update
            value: the value that will be assigned to the node
            outcomeSpace: Dictionary, the outcome space of all nodes
        Return:
            a new factor without node
        """
        assert node in factor['dom'][:-1], 'such node is not in this CPT'
        assert value in outcomeSpace[node], 'no such value for this node'
        new_dom = copy.copy(factor['dom'])
        factor_outcomeSpace = {node: outcomeSpace[node] for node in new_dom}
        factor_outcomeSpace[node] = (value,)
        node_index = new_dom.index(node)
        new_dom_list = list(new_dom)
        new_dom_list.remove(node)
        new_dom = tuple(new_dom_list)
        new_table = odict()
        valid_records = product(*[_ for _ in factor_outcomeSpace.values()])
        for record in valid_records:
            record_list = list(record)
            record_list.pop(node_index)
            new_record = tuple(record_list)
            new_table[new_record] = factor['table'][record]
        return {'dom': new_dom, 'table': new_table}

    def spread(self, graph, source):
        """
        find all nodes reachable from source
        Inputs:
            graph: Dictionary, the graph
            source: list of strings, the node where we start the spread
        Return:
            visted: a set of strings, the nodes reachabel from source.
        """
        visited = set()
        for node in source:
            self.spread_help(graph, node, visited)
        return visited

    def spread_help(self, graph, node, visited):
        visited.add(node)
        for child in graph[node]:
            if child not in visited:
                self.spread_help(graph, child, visited)

    def make_undirected(self, graph):
        """
        Input:
            graph: a directed graph
        Return:
            an undirected (bidirected) graph
        """
        undirectG = graph.copy()
        GT = self.transposeGraph(graph)
        for node in graph:
            undirectG[node] += GT[node]
        return undirectG

    @staticmethod
    def transposeGraph(G):
        """
        Input:
            graph: a directed graph
        Return:
            a transposed graph
        """
        GT = dict((v, []) for v in G)
        for v in G:
            for w in G[v]:
                if w in GT:
                    GT[w].append(v)
                else:
                    GT[w] = [v]
        return GT

    def get_ve_order(self):
        """
        get the variable elimination from the graph
        Return:
            prefix: a list of strings, list of variables in the elimination order
        """
        prefix = []
        moral_graph = self.moralize()
        moral_graph.factors = dict()
        while len(moral_graph.net) > 0:
            low = math.inf
            min_degree = math.inf
            for node, neighbors in moral_graph.net.items():
                fill_num = moral_graph.count_fill(node)
                degree = len(moral_graph.net[node])
                if fill_num < low:
                    min_fill_node = node
                    low = fill_num
                elif fill_num == low:
                    if degree < min_degree:
                        min_fill_node = node
                        min_degree = degree
            moral_graph.remove(min_fill_node)
            prefix.append(min_fill_node)
        return prefix

    def count_fill(self, node):
        """
        count the fill in edges if eliminate node
        Input:
            node: string, the name of the node to be eliminate
        Return:
            int: fill-in edge count
        """
        neighbors = self.net[node]
        neighbor_num = len(neighbors)
        before = 0
        for neighbor in neighbors:
            for neighbor_s_neighbor in self.net[neighbor]:
                if neighbor_s_neighbor in neighbors:
                    before += 1
        before //= 2
        after = neighbor_num*(neighbor_num-1)//2
        return after - before

    def moralize(self):
        """
        moralize the graph
        return:
            a new moral graph
        """
        new_graph = copy.deepcopy(self)
        graphT = self.transposeGraph(new_graph.net)
        new_graph.net = self.make_undirected(new_graph.net)
        for parents in graphT.values():
            new_graph.connect_all(parents)
        return new_graph

    def connect_all(self, nodes):
        """
        connect every node in nodes to every other node
        """
        for father in nodes:
            for child in nodes:
                if father != child:
                    self.connect(father, child)

    def show_moral_graph(self):
        moral_graph = self.moralize()
        dot = Graph()
        dot.attr(overlap="False", splines="True")
        finished = []
        for node, children in moral_graph.net.items():
            dot.node(node)
            for child in children:
                if (child, node) not in finished:
                    dot.edge(node, child)
                    finished.append((node, child))
        return dot

    ####################################
    ######################################
    # Exact inference
    ########################################
    ##########################################
    @staticmethod
    def show_jointree(jointree):
        dot = Graph()
        dot.attr(overlap="False", splines="True")
        finished = []
        for cluster, neighbors in jointree.items():
            dot.node(', '.join(cluster))
            for neighbor in neighbors:
                if (neighbor, cluster) not in finished:
                    dot.edge(', '.join(cluster), ', '.join(neighbor))
                    finished.append((cluster, neighbor))
        return dot

    def build_jointree(self, order):
        """
        self must be a moral graph
        Args:
            order (list): elimination order
        """
        for node, neighbors in self.net.items():
            for neighbor in neighbors:
                assert node in self.net[neighbor], 'the graph is not moral'

        # 1. construct clusters
        clusters = []
        max_cluster_size = 0
        for node in order:
            cluster = set([node] + self.net[node])
            self.connect_all(self.net[node])
            self.remove(node)
            if len(cluster) > max_cluster_size:
                max_cluster_size = len(cluster)
            clusters.append(cluster)
        # 2. maitain RIP
        cluster_seq = [tuple(_) for _ in clusters]
        n = len(clusters)
        for cluster in reversed(clusters):
            if len(cluster) < max_cluster_size:
                i = cluster_seq.index(tuple(cluster))
                for pre in reversed(cluster_seq[:i]):
                    if cluster.issubset(pre):
                        cluster_seq.remove(tuple(cluster))
                        cluster_seq.insert(i, pre)
                        cluster_seq.remove(pre)
                        break
        # 3. assembly
        jointree = dict()
        jointree[cluster_seq[-1]] = []
        n = len(cluster_seq)
        for i in range(n-2, -1, -1):
            jointree[cluster_seq[i]] = []
            edge = set(cluster_seq[i+1]).union(
                *[set(_) for _ in cluster_seq[i+2:]]) & set(cluster_seq[i])
            for other in cluster_seq[i+1:]:
                if edge.issubset(other):
                    jointree[cluster_seq[i]].append(other)
                    break
        return JoinTree(jointree)

    ####################################
    ######################################
    # Approximate inference
    ########################################
    ##########################################

    def gibbs_sampling(self, sample_num=100, chain_num=2, q_vars='all', **q_evis):
        """
        gibbs sampling the graph based on query, sample_num and chain_num specified by the user
        Input:
            sample_num: # of samples
            chain_num: # of chains
            q_vars: list of strings, the query variables, defalt is 'all', which means all variables in the graph other than query evidences
            q_evis: dictionary, the query evidences
        Return:
            samples: a list of dictionaries, each one is a sample contains the node and its value in query
        """
        if q_vars == 'all':
            q_vars = [_ for _ in self.net.keys() if _ not in q_evis]
        prunned_graph = self.prune(q_vars, **q_evis)
        chains = prunned_graph.burn_in(chain_num, **q_evis)
        samples = []
        # fisrt sample
        sample = dict()
        for var in q_vars:
            sample[var] = chains[0].node_value[var]
        samples.append(sample)
        curr = 1
        while curr < sample_num:
            sample = dict()
            for var in q_vars:
                chain = chains[np.random.choice(chain_num)]
                pre_value = samples[curr - 1][var]
                value = chain.sample_once(var)
                # A = chain.get_acceptance(var, pre_value, value)
                # sample[var] = np.random.choice(
                #     [value, pre_value], 1, p=[A, 1-A])[0]
                sample[var] = value
            samples.append(sample)
            curr += 1
        return samples

    def get_acceptance(self, node, pre, curr):
        """
        compute the acceptande probability of this sample
        Inputs:
            node: string, the node waiting to be asigned
            pre: string, the previous value assigned to this node
            curr: string, the current value waiting to be assigned to this node
        Return:
            accpt_prob: float, the acceptande probability
        """
        dom = self.factors[node]['dom']
        parents = dom[: -1]
        parents_value = [self.node_value[parent] for parent in parents]
        ppre = self.factors[node]['table'][tuple(parents_value + [pre])]
        pcurr = self.factors[node]['table'][tuple(parents_value + [curr])]
        return min(1, pcurr/ppre)

    def burn_in(self, chain_num, window_size=100, **evidences):
        """
        generate chains and keep sampling until mixed
        Inputs:
            chain_num: int, # of chains
            window_size: int, # of samples used to test if chains are mixed, defalt is 100
            evidences: dictionary, the evidences of the query
        Return:
            chains: list of GraphicalModel objects, the list of mixed chains
        """
        assert chain_num > 1, 'chain num is at least 2'
        chains = []
        chains_non_evis = []
        for seed in range(chain_num):
            np.random.seed(seed)
            chain = copy.deepcopy(self)
            # 1. fix evidence
            chain.node_value = evidences.copy()
            # 2: Initialize other variables
            non_evis = dict()
            for node, domain in self.outcomeSpace.items():
                if node not in evidences:
                    value = np.random.choice(domain, 1)[0]
                    chain.node_value[node] = value
                    non_evis[node] = [value]
            chains.append(chain)
            chains_non_evis.append(non_evis)

        sample_count = 1
        while True:
            if sample_count >= window_size:
                if self.mixed(chains_non_evis, self.outcomeSpace):
                    break
                # clear the chains_non_evis
                chains_non_evis = [{
                    node: []
                    for node in chains_non_evis[i].keys()
                } for i in range(chain_num)]
                sample_count = 0

            # 3: Choose a variable ordering O
            O = np.random.permutation(list(chains_non_evis[0].keys()))
            # 4: Repeat sample non_evis in the order O
            for var in O:
                for i, chain in enumerate(chains):
                    value = chain.sample_once(var)
                    chain.node_value[var] = value
                    chains_non_evis[i][var].append(value)
                sample_count += 1
        return chains

    def sample_once(self, node):
        """
        sample once for a particular node
        Input:
            node: string, name of the node to sample
        Return:
            a string, a value from this node's outcomeSpace
        """
        dom = self.factors[node]['dom']
        parents = dom[: -1]
        parents_value = [self.node_value[parent] for parent in parents]
        combinations = [tuple(parents_value + [node_value])
                        for node_value in self.outcomeSpace[node]]
        prob_list = [self.factors[node]['table'][combination]
                     for combination in combinations]
        return np.random.choice(self.outcomeSpace[node], 1, p=prob_list)[0]

    @staticmethod
    def convert(list_of_dict, outcomeSpace):
        """
        convert the outcome string value from the outcomespace into float value between 0 and 1
        Input:
            list_of_dic: list of dictionary, each key in the dictionary is a variable and corresponding value is the history of its sample value
            outcomeSpace: dictionary, the outcome space of all nodes
        Return:
            list_of_dic, converted list_of_dict
        """
        mapping = dict()
        for node, values in outcomeSpace.items():
            mapping[node] = dict()
            for value in values:
                mapping[node][value] = (values.index(value)+1) / len(values)
        for i, record in enumerate(list_of_dict):
            list_of_dict[i] = {key: [mapping[key][value]
                                     for value in item] for key, item in record.items()}
        return list_of_dict

    def mixed(self, chain_vars, outcomeSpace):
        """
        to judge whether chain_vars are mixed up
        Inputs:
            chain_vars = [
                {a:[...], b:[...] ...},
                {a:[...], b:[...] ...}]
                the history of samples' value
            outcomeSpace: dictionary, the outcome space of all nodes
        Return:
            bool, whether chain_vars are mixed up
        """
        # covert text value into num like value
        chain_vars = self.convert(chain_vars, outcomeSpace)

        parameters = list(chain_vars[0].keys())
        P_hat = []
        df_list = [pd.DataFrame(var_dic) for var_dic in chain_vars]
        concat_df = pd.concat(df_list, ignore_index=True)
        for parm in parameters:
            W = np.mean([df[parm].var() for df in df_list])
            B = concat_df[parm].var()
            p_hat = ((B+0.01)/(W+0.01)) ** 0.5
            P_hat.append(p_hat)
        return all([_ < 1.1 for _ in P_hat])

    def answer_from_samples(self, samples):
        """
        Answer to the query based on the samples
        Input:
            samples: list of dictionary, where keys are variables, value are node's value
        Return:
            pandas dataframe, the probability of every occured combination
        """
        samples = pd.DataFrame(samples)
        answer = samples.groupby(samples.columns.tolist()).size(
        ).reset_index().rename(columns={0: 'prob'})
        answer.prob = answer.prob/answer.prob.sum()
        return answer.sort_values(by=list(answer.columns[:-1]), ascending=False).reset_index(drop=True)


class JoinTree:
    def __init__(self, clusters=dict()):
        """
        cluster: key: list of node in this cluster, value: neighbor clusters
        """
        self.clusters = clusters
        self.factors = dict()
        self.outcomeSpace = dict()

    def show_jointree(self):
        dot = Graph()
        dot.attr(overlap="False", splines="True")
        finished = []
        for cluster, neighbors in self.clusters.items():
            dot.node(', '.join(cluster))
            for neighbor in neighbors:
                if (neighbor, cluster) not in finished:
                    dot.edge(', '.join(cluster), ', '.join(neighbor))
                    finished.append((cluster, neighbor))
        return dot

    def add_var(self, var, cluster):
        """add variabel to a cluster, at least one neighbor of the cluster should have this variable

        Args:
            var (string): variable name
            cluster (tuple): cluster
        """
        assert any([var in neighbor for neighbor in self.clusters[cluster]]
                   ), 'at least one neighbor of the cluster should have this variable'
        if var not in cluster:
            old = cluster
            cluster = list(cluster)
            cluster.append(var)
            cluster = tuple(cluster)
            self.clusters[cluster] = self.clusters.pop(old)
            for nb in self.clusters[cluster]:
                self.clusters[nb].remove(old)
                self.clusters[nb].append(cluster)

    def merge(self, cluster1, cluster2):
        """merge two neighboring clusters

        Args:
            cluster1 (list of nodes): cluster1
            cluster2 (list of nodes): cluster2
        """
        assert cluster2 in self.clusters[cluster1] and cluster1 in self.clusters[cluster2], 'clusters are not neighbors'
        assert cluster1 != cluster2
        newcluster = tuple(set(cluster1) | set(cluster2))
        self.clusters[newcluster] = list(
            set(self.clusters.pop(cluster1)) | set(self.clusters.pop(cluster2)))
        self.clusters[newcluster].remove(cluster1)
        self.clusters[newcluster].remove(cluster2)
        for nb in self.clusters[newcluster]:
            print(nb)
            self.clusters[nb].remove(cluster1)
            self.clusters[nb].remove(cluster2)
            self.clusters[nb].append(newcluster)

    def add_cluster(self, cluster, next_to=None):
        """add cluster to an existing cluster

        Args:
            cluster (list of nodes): new added cluster
            next_to (cluster): an existing cluster
        """
        if next_to:
            assert next_to in self.clusters, 'no such cluster'
            assert set(cluster).issubset(
                next_to), 'new added cluster should be a subset of the existing cluster'
        if cluster not in self.clusters:
            self.clusters[cluster] = []
            if next_to:
                self.clusters[cluster].append(next_to)
                self.clusters[next_to].append(cluster)

    def remove(self, cluster):
        """remove a cluster from the jointree

        Args:
            cluster (list of nodes): cluster need to be removed
        """
        if cluster in self.clusters:
            for nb in self.clusters[cluster]:
                self.clusters[nb].remove(cluster)
            self.clusters.pop(cluster)
