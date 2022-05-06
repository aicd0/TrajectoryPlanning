import bisect
import config
import math
import numpy as np
import os
import random
import time
import utils.fileio
import utils.math
import utils.print
from copy import copy
from framework.geometry import Geometry
from framework.robot import Robot
from sortedcontainers import SortedKeyList
from typing import Any

class NodeExportable:
    def __init__(self, joint_position: np.ndarray, position: np.ndarray, neighbours: list[int]) -> None:
        self.joint_position = joint_position
        self.position = position
        self.neighbours = neighbours

class Node:
    def __init__(self, joint_position: np.ndarray, position: np.ndarray, neighbours: list) -> None:
        self.idx = -1
        self.joint_position = joint_position
        self.position = position
        self.neighbours = neighbours

class Workspace:
    def __init__(self, name: str) -> None:
        self.name = name
        self.save_dir = config.Workspace.SaveDir
        self.nodes: list[Node] = []
        self.meta: dict[str, dict[int, Any]] = {}
        self.cartesian_space: list[Node] = []
        self.obstacles: list[Geometry] = []
        self.sorted = {}
        self.r = None

    def clear(self):
        self.nodes.clear()
        self.meta.clear()
        self.cartesian_space.clear()
        self.obstacles.clear()
        self.sorted.clear()
        self.r = None

    def save(self) -> None:
        assert len(self.nodes) > 0

        # Export nodes.
        nodes = [NodeExportable(n.joint_position, n.position, [m.idx for m in n.neighbours]) for n in self.nodes]
        cartesian_space = [n.idx for n in self.cartesian_space]

        utils.fileio.mktree(self.save_dir)
        np.savez(self.save_dir + self.name,
            nodes=np.array(nodes, dtype=object),
            meta=np.array(self.meta, dtype=object),
            cartesian_space=np.array(cartesian_space, dtype=object),
            obstacles=np.array(self.obstacles, dtype=object),
            r=self.r)

    def load(self) -> bool:
        filepath = self.save_dir + self.name + '.npz'
        if not os.path.exists(filepath):
            return False
        checkpoint = np.load(filepath, allow_pickle=True)
        nodes = checkpoint['nodes'].tolist()
        self.meta = checkpoint['meta'].tolist()
        cartesian_space = checkpoint['cartesian_space'].tolist()
        self.obstacles = checkpoint['obstacles'].tolist()
        self.r = float(checkpoint['r'])

        # Import nodes.
        self.nodes = [Node(n.joint_position, n.position, []) for n in nodes]
        for i, node in enumerate(self.nodes):
            node.neighbours = [self.nodes[j] for j in nodes[i].neighbours]
        self.cartesian_space = [self.nodes[i] for i in cartesian_space]
        self.__setup()

        utils.print.put('Workspace loaded (' + self.name + ')')
        return True

    def make(self, robot: Robot, joint_positions: float, min_r: float, obstacles: list[Geometry]=[]) -> None:
        self.clear()
        self.obstacles = obstacles
        self.__make_nodes(robot, joint_positions)
        fragments = self.__fragments()
        utils.print.put('Workspace fragments: %d (%d%%)' % (fragments, int(fragments / len(self.nodes) * 100)))
        self.__make_cartesian_space(min_r)
        self.__setup()

    # Nodes service
    def nearest_joint_position(self, joint_position: np.ndarray) -> Node:
        assert len(self.nodes) > 0
        min_d2 = np.inf
        for node in self.nodes:
            d2 = utils.math.distance2(node.joint_position, joint_position)
            if d2 < min_d2:
                min_d2 = d2
                ans = node
        return ans

    def nearest_positions(self, pos: np.ndarray) -> list[Node]:
        def _nearest_positions(max_d: float):
            pool: set[Node] = None
            for i in range(self.dim_position):
                subset: set[Node] = set()
                node_sorted = self.sorted['pos'][i]
                j = bisect.bisect_right(node_sorted, pos[i] - max_d, key=lambda n: n.position[i])
                while j < len(node_sorted) and  node_sorted[j].position[i] <= pos[i] + max_d:
                    subset.add(node_sorted[j])
                    j += 1
                if pool is None:
                    pool = subset
                else:
                    pool = pool.intersection(subset)

            ans = set()
            max_d2 = max_d * max_d
            for node in pool:
                d2 = utils.math.distance2(pos, node.position)
                if d2 < max_d2:
                    ans.add(node)
            return ans
        max_d = 0.05
        ans = []
        while len(ans) <= 0:
            ans = _nearest_positions(max_d)
            max_d += 0.05
        return ans

    # Meta service
    def get_meta(self, meta: str, node: Node) -> Any | None:
        nodes_meta = self.meta.get(meta)
        if nodes_meta is None:
            return None
        return nodes_meta.get(node.idx)

    def set_meta(self, meta: str, node: Node, value: Any) -> None:
        nodes_meta = self.meta.get(meta)
        if nodes_meta is None:
            nodes_meta = self.meta[meta] = {}
        nodes_meta[node.idx] = value

    def clear_meta(self, meta: str) -> None:
        if meta in self.meta:
            del self.meta[meta]

    # Sampling
    def sample_cartesian_space(self) -> np.ndarray:
        assert len(self.cartesian_space) > 0
        center = random.sample(self.cartesian_space, 1)[0].position
        error = utils.math.random_point_in_hypersphere(self.dim_position, high=self.r)
        return center + error

    # Internals
    def __setup(self) -> None:
        for i, node in enumerate(self.nodes):
            node.idx = i
        self.dim_position = self.nodes[0].position.shape[0]
        self.dim_joint_position = self.nodes[0].joint_position.shape[0]
        self.sorted['pos'] = [
            sorted(self.nodes, key=lambda n: n.position[i]) for i in range(self.dim_position)]
        self.sorted['joint_pos'] = [
            sorted(self.nodes, key=lambda n: n.joint_position[i]) for i in range(self.dim_joint_position)]

    def __make_nodes(self, robot: Robot, joint_positions: float) -> None:
        state_count = math.prod([len(p) for p in joint_positions])
        assert state_count > 0
        max_depth = len(joint_positions)

        def _find_neighbours(dst: list[Node], partial_joint_space: list, base_idx: list[int],
                             pos: np.ndarray, current_idx: list[int], depth: int) -> None:
            for delta in [-1, 0, 1]:
                idx = base_idx[depth] + delta
                if not 0 <= idx < len(partial_joint_space):
                    continue
                current_idx[depth] = idx
                if depth >= max_depth - 1:
                    node: Node = partial_joint_space[idx]
                    dst.append(node)
                else:
                    _find_neighbours(dst, partial_joint_space[idx], base_idx, pos, current_idx, depth + 1)

        def find_neighbours(joint_space: list, idx: list[int], pos: np.ndarray) -> list[Node]:
            neighbours = []
            current_idx = [None for _ in range(max_depth)]
            _find_neighbours(neighbours, joint_space, idx, pos, current_idx, 0)
            return neighbours

        def try_create_node(pool: list[Node], joint_space: list,
                            joint_position: np.ndarray, idx: list[int]) -> Node | None:
            # Check collisions.
            origins = robot.collision_points(joint_position)
            for origin in origins:
                if origin[2] < 0:
                    return None
                for obj in self.obstacles:
                    if obj.contain(origin):
                        return None

            # Lookup neighbours.
            neighbours = find_neighbours(joint_space, idx, origins[-1])

            # Generate a new node.
            new_node = Node(copy(joint_position), origins[-1], neighbours)
            for node in neighbours:
                node.neighbours.append(new_node)
            pool.append(new_node)
            return new_node

        node_processed = 0
        def _make_joint_space(pool: list[Node], joint_space: list, partial_joint_space: list,
                              current_state: list[np.ndarray], current_idx: list[int], depth: int):
            nonlocal last_update_time, node_processed

            for i, state in enumerate(joint_positions[depth]):
                current_state[depth] = state
                current_idx[depth] = i
                if depth >= max_depth - 1:
                    new_node = try_create_node(pool, joint_space, current_state, current_idx)
                    if not new_node is None:
                        partial_joint_space.append(new_node)
                    
                    # Progress.
                    node_processed += 1
                    if time.time() - last_update_time > 1:
                        utils.print.put('Generating nodes (%d - %d/%d - %d%%)' %
                            (len(pool), node_processed, state_count, int(node_processed / state_count * 100)),
                            same_line=True)
                        last_update_time = time.time()
                else:
                    new_states = []
                    partial_joint_space.append(new_states)
                    _make_joint_space(pool, joint_space, new_states, current_state, current_idx, depth + 1)

        def make_joint_space() -> list[Node]:
            pool = []
            joint_space = []
            current_state = np.zeros(max_depth)
            current_idx = [None for _ in range(max_depth)]
            _make_joint_space(pool, joint_space, joint_space, current_state, current_idx, 0)
            return pool
        
        last_update_time = time.time()
        self.nodes = make_joint_space()

    def __make_cartesian_space(self, r: float) -> None:
        self.r = r
        pos_sorted = [
            SortedKeyList(self.nodes, lambda n: float(n.position[0])),
            SortedKeyList(self.nodes, lambda n: float(n.position[1])),
            SortedKeyList(self.nodes, lambda n: float(n.position[2])),
        ]
        pool = set(self.nodes)
        r2 = r * r
        last_update_time = time.time()

        while len(pool) > 0:
            new_node = pool.pop()
            for node_list in pos_sorted:
                node_list.remove(new_node)

            # Search possible neighbours
            possible_neighbours: set[Node] = None
            for i in range(3):
                subset: set[Node] = set()
                it = pos_sorted[i].irange_key(new_node.position[i] - r, new_node.position[i] + r)
                for node in it:
                    subset.add(node)
                if possible_neighbours is None:
                    possible_neighbours = subset
                else:
                    possible_neighbours = possible_neighbours.intersection(subset)
            
            # Remove neighbours from lists.
            for node in possible_neighbours:
                if utils.math.distance2(new_node.position, node.position) < r2:
                    pool.remove(node)
                    for node_list in pos_sorted:
                        node_list.remove(node)

            # Add to cartesian_space.
            self.cartesian_space.append(new_node)

            # Progress.
            if time.time() - last_update_time > 1:
                node_processed = len(self.nodes) - len(pool)
                utils.print.put('Generating nodes in Cartesian Space (%d - %d/%d - %d%%)' %
                    (len(self.cartesian_space), node_processed, len(self.nodes),
                    int(node_processed / len(self.nodes) * 100)), same_line=True)
                last_update_time = time.time()

    def __fragments(self) -> int:
        pool = set(self.nodes)
        fragments = 0
        while len(pool) > 0:
            fragments += 1
            nodes = [pool.pop()]
            while len(nodes):
                neighbours = []
                for node in nodes:
                    for neighbour in node.neighbours:
                        if neighbour in pool:
                            pool.remove(neighbour)
                            neighbours.append(neighbour)
                nodes = neighbours
        return fragments