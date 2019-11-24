import uuid
from collections import defaultdict
from typing import Dict, List

from remat.core.dfgraph import DFGraph
from remat.core.utils.definitions import EdgeList, AdjList
from remat.core.utils.dfgraph_utils import toposort


class GraphBuilder:
    def __init__(self):
        self.nodes: Dict[str, uuid.UUID] = {}  # map of user-facing names to internal uuid
        self.backward_nodes: Dict[uuid.UUID, bool] = {}  # map of nodes to if they are backwards nodes
        self.arguments: Dict[uuid.UUID, List[uuid.UUID]] = {}  # map of internal node uuid to its list of arguments (uuids)
        self.costs_cpu: Dict[uuid.UUID, int] = {}  # map of per-node CPU costs
        self.costs_ram: Dict[uuid.UUID, int] = {}  # map of per-node RAM costs

    def _name_to_uuid(self, name: str) -> uuid.UUID:
        if name not in self.nodes.keys():
            self.nodes[name] = uuid.uuid1()
        return self.nodes[name]

    def add_node(self, name: str, cpu_cost: int, ram_cost: int, backward: bool = False) -> "GraphBuilder":
        """
        Add a node to the graph with associated graph properties
        :param name: A name for the node that is added to the graph
        :param cpu_cost: Cost to evaluate this node in runtime or in FLOPs
        :param ram_cost: RAM cost needed to store the output of this node along with the temporary state like workspace
        :param backward: Boolean denoting whether this node is a backward node
        """
        # TODO (paras) support flag denoting do-not-recompute
        uuid = self._name_to_uuid(name)
        self.backward_nodes[uuid] = backward
        self.costs_cpu[uuid] = cpu_cost
        self.costs_ram[uuid] = ram_cost
        self.arguments[uuid] = []
        return self

    def set_deps(self, dest_node: str, *source_nodes: str) -> "GraphBuilder":
        dest_node_uuid = self._name_to_uuid(dest_node)
        if source_nodes is None or len(source_nodes) != 0:
            self.arguments[dest_node_uuid] = []
        else:
            self.arguments[dest_node_uuid] = list(map(self._name_to_uuid, source_nodes))
        return self

    def make_graph(self) -> DFGraph:
        # step 1 -- toposort graph and allocate node positions as a dict({0, ..., n} -> UUID)
        edge_list: EdgeList = [(source.int, dest.int) for dest, sources in self.arguments.items() for source in sources]
        uuid2topo = dict(reversed(pair) for pair in enumerate(uuid.UUID(int=x) for x in toposort(edge_list)))

        # step 2 -- map builder data-structures to node position indexed data-structures
        #           we need (1) list of vertices, (2) per-vertex cpu/ram costs, (3) arg-list and (4) node-name mapping
        vertex_list = list(uuid2topo.values())
        cost_cpu = dict((uuid2topo[idx], self.costs_cpu[idx]) for idx in self.costs_cpu.keys())
        cost_ram = dict((uuid2topo[idx], self.costs_ram[idx]) for idx in self.costs_ram.keys())
        arg_list: AdjList = {uuid2topo[key]: [uuid2topo[arg] for arg in args] for key, args in self.arguments.items()}
        names = {uuid2topo[idx]: name for idx, name in self.nodes.items()}

        # step 3 -- make DFGraph
        pass


def gen_linear_graph(forward_node_count, **kwargs):
    """
    gen_linear_graph will generate linear-style graphs like VGG and AlexNet.
    Method returns forward and backward graphs. Pass cost_ram and cost_cpu as kwargs.
    :param forward_node_count: number of forward (not backward nodes)
    :return: Graph object containing linear graph
    """
    args = defaultdict(list)
    vfwd_map = {}
    loss_node_idx = forward_node_count
    for i in range(forward_node_count * 2):
        args[i + 1].append(i)
        if i < forward_node_count:
            corresponding_bwd = (forward_node_count * 2) - i
            args[corresponding_bwd].append(i)
            vfwd_map[i] = corresponding_bwd
    v = list(vfwd_map.keys()) + list(vfwd_map.values()) + [loss_node_idx]
    return DFGraph(args=args, v=v, vfwd_map=vfwd_map, vloss=loss_node_idx, **kwargs)