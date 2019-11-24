import uuid
from typing import Dict, List
from uuid import UUID

from remat.core.dfgraph import DFGraph


class GraphBuilder:
    def __init__(self):
        self.nodes: Dict[str, UUID] = {}  # map of user-facing names to internal uuid
        self.backward_nodes: Dict[UUID, bool] = {}  # map of nodes to if they are backwards nodes
        self.arguments: Dict[UUID, List[UUID]] = {}  # map of internal node uuid to its list of arguments (uuids)
        self.costs_cpu: Dict[UUID, int] = {}  # map of per-node CPU costs
        self.costs_ram: Dict[UUID, int] = {}  # map of per-node RAM costs

    def _name_to_uuid(self, name: str) -> UUID:
        if name not in self.nodes.keys():
            self.nodes[name] = uuid.uuid1()
        return self.nodes[name]

    def add_node(self, name: str, cpu_cost: int, ram_cost: int, backward: bool = False) -> "GraphBuilder":
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
        # step 1 -- toposort graph and allocate node positions from {0, ..., n}
        # step 2 -- map builder data-structures to node position indexed data-structures
        # step 3 -- make DFGraph
        pass
