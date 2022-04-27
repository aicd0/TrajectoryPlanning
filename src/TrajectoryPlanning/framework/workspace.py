import config
import numpy as np
import os
import random
import time
import utils.fileio
import utils.math
import utils.print
from typing import Callable

class NodeExportable:
    def __init__(self, pos: np.ndarray, neighbours: list[int]) -> None:
        self.pos = pos
        self.neighbours = neighbours

class Node:
    def __init__(self, pos: np.ndarray, neighbours: list) -> None:
        self.pos = pos
        self.neighbours = neighbours
        self.idx = -1

    def to_exportable(self) -> NodeExportable:
        return NodeExportable(self.pos, [n.idx for n in self.neighbours])

class Workspace:
    def __init__(self) -> None:
        self.save_dir = config.Workspace.SaveDir
        self.workspace: list[Node] = []
        self.d = None

    def fill(self, gen_func: Callable, min_d: float, max_link_length: float, max_retry: int, append=False) -> None:
        assert min_d < max_link_length
        if not append:
            self.workspace.clear()
        self.d = min_d
        retry = 0
        last_log_time = time.time()

        while retry <= max_retry:
            pos = gen_func()
            neighbours: list[Node] = []
            valid = True

            for node in self.workspace:
                d = utils.math.distance(pos, node.pos)
                if d < min_d:
                    valid = False
                    break
                if d <= max_link_length:
                    neighbours.append(node)

            if valid:
                new_node = Node(pos, neighbours)
                for node in neighbours:
                    node.neighbours.append(new_node)
                self.workspace.append(new_node)
                retry = 0
            else:
                retry += 1

            if time.time() - last_log_time > 1:
                utils.print.put('%d nodes generated.' % len(self.workspace), same_line=True)
                last_log_time = time.time()

    def save(self, name: str) -> None:
        assert len(self.workspace) > 0

        # Export nodes.
        for i, node in enumerate(self.workspace):
            node.idx = i
        obj = [n.to_exportable() for n in self.workspace]

        utils.fileio.mktree(self.save_dir)
        np.savez(self.save_dir + name,
            workspace=np.array(obj, dtype=object),
            d=self.d)

    def load(self, name: str) -> bool:
        filepath = self.save_dir + name + '.npz'
        if not os.path.exists(filepath):
            return False
        checkpoint = np.load(filepath, allow_pickle=True)
        obj = checkpoint['workspace'].tolist()
        self.d = float(checkpoint['d'])

        # Import nodes.
        self.workspace = [Node(n.pos, []) for n in obj]
        for i, node in enumerate(self.workspace):
            node.neighbours = [self.workspace[j] for j in obj[i].neighbours]

        utils.print.put('Workspace loaded (' + name + ')')
        return True

    def sample(self) -> np.ndarray:
        assert len(self.workspace) > 0
        center = random.sample(self.workspace, 1)[0].pos
        error = utils.math.random_point_in_hypersphere(3, high=self.d)
        return center + error

    def nearest_node(self, pos: np.ndarray) -> Node:
        assert len(self.workspace) > 0
        ans = None
        min_d = np.inf
        for node in self.workspace:
            d = utils.math.distance(pos, node.pos)
            if d < min_d:
                min_d = d
                ans = node
        return ans