import numpy as np
import utils.math
from framework.planner import Planner
from framework.workspace import Workspace
from framework.workspace import Node as WorkspaceNode
from sortedcontainers import SortedKeyList

class Node:
    def __init__(self, wsnode: WorkspaceNode) -> None:
        self.wsnode = wsnode
        self.from_start = 0
        self.to_finish = 0
        self.parent = None
        
    @property
    def priority(self) -> float:
        return self.from_start + self.to_finish

def a_star(workspace: Workspace, joint_position: np.ndarray, target_pos: np.ndarray) -> list[np.ndarray] | None:
    start_wsnode = workspace.nearest_joint_position(joint_position)
    finish_wsnodes = set(workspace.find_positions(target_pos, 0.05))
    if start_wsnode is None or len(finish_wsnodes) == 0: return None
    start_node = Node(start_wsnode)
    next_node_ordered = SortedKeyList([start_node], key=lambda n: n.priority)
    next_node_map = {start_node.wsnode: start_node}

    while len(next_node_ordered) > 0:
        # Get the least cost node to remove.
        opt_node: Node = next_node_ordered.pop(0)
        del next_node_map[opt_node.wsnode]

        # Merge neighbours.
        for wsnode in opt_node.wsnode.neighbours:
            neighbour = next_node_map.get(wsnode)
            from_start = opt_node.from_start + utils.math.distance(
                opt_node.wsnode.joint_position, wsnode.joint_position)

            if neighbour is None:
                if wsnode in finish_wsnodes:
                    # Solution found.
                    path = [wsnode.joint_position]
                    parent = opt_node
                    while not parent is None:
                        path.insert(0, parent.wsnode.joint_position)
                        parent = parent.parent
                    return path
                neighbour = Node(wsnode)
                neighbour.to_finish = min([utils.math.distance(
                    n.joint_position, wsnode.joint_position) for n in finish_wsnodes])
                next_node_map[wsnode] = neighbour
            else:
                if from_start >= neighbour.from_start:
                    continue
                next_node_ordered.remove(neighbour)
            neighbour.from_start = from_start
            neighbour.parent = opt_node
            next_node_ordered.add(neighbour)
    return None

class AStarPlanner(Planner):
    def __init__(self, sim) -> None:
        super().__init__(sim)
    
    def reach(self, pos: np.ndarray) -> bool:
        state = self.sim.state()
        path = a_star(self.sim.workspace, state.joint_position, pos)
        if path is None:
            return False
        for joint_position in path:
            if not self._reach(joint_position):
                return False
        return True