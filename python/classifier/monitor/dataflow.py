import re
from collections import defaultdict

from ..utils import TypedStr, ranges_str

__all__ = ['DfColumns', 'Source', 'Target']


class Source(TypedStr):
    ...


class Target(TypedStr):
    ...


class DataFlow:
    _style = {
        '': {
            'shape': 'box',
        },
        'intermediate': {
            'color': '#92E7F5',
        },
        'source': {
            'color': '#64F656',
        },
        'target': {
            'color': '#7F8FF5',
        },
        'unused': {
            'color': '#F5E46A',
        },
        'missing': {
            'color': '#F56C5E',
        },
        'isolated': {
            'color': '#F5B04E',
        },
    }

    _graph: defaultdict[str, set]
    _pattern: re.Pattern

    def __init_subclass__(cls):
        cls._graph = defaultdict(set)
        if isinstance(cls._pattern, str):
            cls._pattern = re.compile(cls._pattern)

    @staticmethod
    def _node_type(node: str | Source | Target) -> str:
        if isinstance(node, str):
            return 'intermediate'
        else:
            return type(node).__name__.lower()

    @classmethod
    def _merge_nodes(cls, *nodes: str):
        keys = defaultdict(set)
        for key in nodes:
            matched = cls._pattern.match(key)
            if matched:
                idx = int(matched.groupdict()['index'])
                span = matched.span('index')
                keys[key[:span[0]] + key[span[1]:]].add(idx)
            else:
                keys[key]
        labels = []
        for k in sorted(keys):
            v = keys[k]
            if not v:
                labels.append(k)
            else:
                labels.append(f'{k} [{",".join(ranges_str(sorted(v)))}]')
        return '\n'.join(labels)

    @classmethod
    @property
    def graph(cls):
        import networkx as nx
        from pyvis.network import Network
        from pyvis.options import Layout
        graph = nx.DiGraph()
        nodes_cat = defaultdict(set)
        nodes_map = {}
        # add nodes
        for node in cls._graph:
            nodes_cat[cls._node_type(node)].add(node)
            graph.add_node(node)
        # add edges
        for start, ends in cls._graph.items():
            for end in ends:
                graph.add_edge(start, end)
        # check connectivity
        for column in list(nodes_cat['intermediate']):
            from_source = False
            to_target = False
            for source in nodes_cat['source']:
                if nx.node_connectivity(graph, column, source) > 0:
                    from_source = True
                    break
            for target in nodes_cat['target']:
                if nx.node_connectivity(graph, target, column) > 0:
                    to_target = True
                    break
            if not from_source or not to_target:
                nodes_cat['intermediate'].remove(column)
            if not from_source:
                if not to_target:
                    nodes_cat['isolated'].add(column)
                else:
                    nodes_cat['missing'].add(column)
            elif not to_target:
                nodes_cat['unused'].add(column)
        # merge equivalent nodes
        equiv: defaultdict[
            str, defaultdict[
                tuple[frozenset, frozenset], set
            ]] = defaultdict(lambda: defaultdict(set))
        for node_type in ['intermediate', 'missing', 'unused', 'isolated']:
            for column in nodes_cat[node_type]:
                in_edges = frozenset(
                    node for node, _ in graph.in_edges(column))
                out_edges = frozenset(
                    node for _, node in graph.out_edges(column))
                equiv[node_type][(in_edges, out_edges)].add(column)
            merged_nodes = equiv[node_type]
            for k in list(merged_nodes):
                v = merged_nodes[k]
                merged = cls._merge_nodes(*v)
                merged_nodes[k] = merged
                for node in v:
                    nodes_map[node] = merged
                    nodes_cat[node_type].remove(node)

        def merged_node(node):
            if node in nodes_map:
                return nodes_map[node]
            return node
        # make new graph
        graph = nx.DiGraph()

        def add_node(node, node_type):
            node = str(node)
            graph.add_node(
                node,
                label=node,
                type=node_type,
                **cls._style[node_type],
                **cls._style[''])
        # add nodes
        for node_type, nodes in nodes_cat.items():
            for node in nodes:
                add_node(node, node_type)
        for node_type, merged_nodes in equiv.items():
            for node in merged_nodes.values():
                add_node(node, node_type)
        # add edges
        for node_type, nodes in nodes_cat.items():
            for node in nodes:
                for edge in cls._graph[node]:
                    graph.add_edge(str(merged_node(edge)), str(node))
        for merged_nodes in equiv.values():
            for (in_edges, out_edges), node in merged_nodes.items():
                for start in in_edges:
                    graph.add_edge(node, str(merged_node(start)))
                for end in out_edges:
                    graph.add_edge(str(merged_node(end)), node)
        # visualize
        gvis = Network(
            directed=True,
            cdn_resources='remote',
            layout=Layout(randomSeed=0))
        gvis.options.physics.enabled = False
        gvis.options.layout.hierarchical.enabled = True
        gvis.options.layout.hierarchical.sortMethod = 'directed'
        gvis.from_nx(graph)
        return gvis

    @classmethod
    def add(cls, *columns: str):
        for column in columns:
            cls._graph[column]

    @classmethod
    def add_derived(cls, derived: str, *columns: str):
        cls.add(*columns, derived)
        cls._graph[derived].update(columns)


class DfColumns(DataFlow):
    _pattern = r'(.*?)(?P<index>[0-9]+)_(.*)'
