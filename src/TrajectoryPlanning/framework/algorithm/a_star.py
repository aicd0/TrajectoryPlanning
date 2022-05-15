import numpy as np
import utils.math
from framework.planner import Planner
from framework.workspace import Workspace
from framework.workspace import Node as WorkspaceNode
from sortedcontainers import SortedKeyList
from typing import Generator

class Node:
    def __init__(self, wsnode: WorkspaceNode, from_start: float=None, to_finish: float=None) -> None:
        self.wsnode = wsnode
        self.from_start = from_start
        self.to_finish = to_finish
        self.last_node = None
        self.next_node = None
        self.closed = False
        
    @property
    def priority(self) -> float:
        return self.from_start + self.to_finish

def bidirectional_a_star(workspace: Workspace, joint_position: np.ndarray,
                         target_position: np.ndarray) -> list[np.ndarray] | None:
    '''Bidirectional A* implementation.'''
    start_wsnodes = set([workspace.nearest_node_from_joint_position_2(joint_position)])
    assert len(start_wsnodes) > 0
    finish_wsnodes = workspace.nearest_nodes_from_position_2(target_position, max_d=0.05)
    assert len(finish_wsnodes) > 0

    estimator = lambda nfrom, nto: min([utils.math.distance(
        n.joint_position, nfrom.joint_position) for n in nto])
    nodes_map = {n: Node(n, from_start=estimator(n, start_wsnodes),
        to_finish=estimator(n, finish_wsnodes)) for n in set.union(start_wsnodes, finish_wsnodes)}

    start_nodes = [nodes_map[n] for n in start_wsnodes]
    finish_nodes = [nodes_map[n] for n in finish_wsnodes]
    
    forward_nodes = SortedKeyList(start_nodes, key=lambda n: n.priority)
    backward_nodes = SortedKeyList(finish_nodes, key=lambda n: n.priority)

    ans_node: Node = None

    while len(forward_nodes) > 0 and len(backward_nodes) > 0:
        # Get the least cost node from forward list to remove.
        opt_node: Node = forward_nodes.pop(0)
        opt_node.closed = True

        # Solution found.
        if opt_node in backward_nodes:
            ans_node = opt_node
            break

        # Merge neighbours.
        for wsnode in opt_node.wsnode.neighbours:
            neighbour = nodes_map.get(wsnode)
            if not neighbour is None and neighbour.closed:
                continue
            from_start = opt_node.from_start + utils.math.distance(
                opt_node.wsnode.joint_position, wsnode.joint_position)
            in_backward = False
            if neighbour is None:
                neighbour = Node(wsnode, to_finish=estimator(wsnode, finish_wsnodes))
                nodes_map[wsnode] = neighbour
            else:
                if neighbour in forward_nodes:
                    if from_start >= neighbour.from_start:
                        continue
                    forward_nodes.remove(neighbour)
                in_backward = neighbour in backward_nodes
                if in_backward:
                    backward_nodes.remove(neighbour)
            neighbour.from_start = from_start
            neighbour.last_node = opt_node
            forward_nodes.add(neighbour)
            if in_backward:
                backward_nodes.add(neighbour)

        # Get the least cost node from backward list to remove.
        opt_node: Node = backward_nodes.pop(0)
        opt_node.closed = True

        # Solution found.
        if opt_node in forward_nodes:
            ans_node = opt_node
            break

        # Merge neighbours.
        for wsnode in opt_node.wsnode.neighbours:
            neighbour = nodes_map.get(wsnode)
            if not neighbour is None and neighbour.closed:
                continue
            to_finish = opt_node.to_finish + utils.math.distance(
                opt_node.wsnode.joint_position, wsnode.joint_position)
            in_forward = False
            if neighbour is None:
                neighbour = Node(wsnode, from_start=estimator(wsnode, start_wsnodes))
                nodes_map[wsnode] = neighbour
            else:
                if neighbour in backward_nodes:
                    if to_finish >= neighbour.to_finish:
                        continue
                    backward_nodes.remove(neighbour)
                in_forward = neighbour in forward_nodes
                if in_forward:
                    forward_nodes.remove(neighbour)
            neighbour.to_finish = to_finish
            neighbour.next_node = opt_node
            backward_nodes.add(neighbour)
            if in_forward:
                forward_nodes.add(neighbour)

    if ans_node is None:
        return None

    track = []
    last_node = ans_node
    while not last_node is None:
        track.append(last_node.wsnode.joint_position)
        last_node = last_node.last_node
    track.reverse()
    next_node = ans_node.next_node
    while not next_node is None:
        track.append(next_node.wsnode.joint_position)
        next_node = next_node.next_node
    return track

class AStarPlanner(Planner):
    def __init__(self, sim, **kwarg) -> None:
        super().__init__(sim, **kwarg)
    
    def _plan(self, position: np.ndarray) -> Generator[np.ndarray | None, None, None]:
        state = self.sim.state()
        track = bidirectional_a_star(self.sim.workspace, state.joint_position, position)
        if track is None:
            yield None
        else:
            for joint_position in track:
                yield joint_position