import numpy as np
import utils.math
from copy import copy
from framework.planner import Planner
from framework.workspace import Workspace
from framework.workspace import Node as WorkspaceNode

class Node:
    def __init__(self, node: WorkspaceNode, finish: np.ndarray, base=None) -> None:
        self.node = node
        self.priority = utils.math.distance(finish, self.node.pos)
        if base is None:
            self.cost = 0
            self.path = [self.node.pos]
        else:
            self.cost = np.inf
            self.compare_and_rebase(base)

    def compare_and_rebase(self, base) -> None:
        new_cost = base.cost + utils.math.distance(base.node.pos, self.node.pos)
        if self.cost > new_cost:
            self.cost = new_cost
            self.path = copy(base.path)
            self.path.append(base.node.pos)

def a_star(workspace: Workspace, start: np.ndarray, finish: np.ndarray) -> list[np.ndarray]:
    start_wsnode = workspace.nearest_node(start)
    finish_wsnode = workspace.nearest_node(finish)

    if len(start_wsnode.neighbours) == 0 or len(finish_wsnode.neighbours) == 0:
        return None

    start_node = Node(start_wsnode, finish_wsnode.pos)
    next_nodes = {start_node.node: start_node}
    removed_nodes = {}

    while len(next_nodes) > 0:
        # Find a node to remove.
        opt_node = None
        min_cost = np.inf
        for node in next_nodes.values():
            cost = node.cost + node.priority
            if cost < min_cost:
                min_cost = cost
                opt_node = node
        
        # Remove node.
        del next_nodes[opt_node.node]
        removed_nodes[opt_node.node] = None

        # Add neighbours.
        for wsnode in opt_node.node.neighbours:
            if wsnode in removed_nodes:
                continue
            if wsnode in next_nodes:
                next_nodes[wsnode].compare_and_rebase(opt_node)
            else:
                new_node = Node(wsnode, finish_wsnode.pos, base=opt_node)
                next_nodes[wsnode] = new_node
                if wsnode is finish_wsnode:
                    return new_node.path

    return None

class AStarPlanner (Planner):
    def __init__(self, sim, iks) -> None:
        super().__init__(sim, iks)
    
    def reach(self, pos: np.ndarray) -> bool:
        state = self.sim.state()
        path = a_star(self.sim.workspace, state.achieved, pos)
        if path is None:
            return False
        path.append(pos)
        for pt in path:
            self._reach(pt)
        return True