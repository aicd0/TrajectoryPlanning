import config
import numpy as np
import utils.fileio
from copy import copy

def make_rect(filename: str, track: list[np.ndarray], velocity: float):
    assert len(track) > 0
    assert velocity > 0

    current_joint_pos = copy(track[0])
    track_export = [copy(current_joint_pos)]
    for joint_pos in track[1:]:
        while True:
            action = joint_pos - current_joint_pos
            if np.max(np.abs(action)) < 1e-5:
                break
            action = action.clip(-velocity, velocity)
            current_joint_pos += action
            track_export.append(copy(current_joint_pos))
        
    save_path = config.Export.SaveDir
    utils.fileio.mktree(save_path)
    filepath = save_path + filename + '.rect'

    with open(filepath, 'wb') as f:
        for joint_pos in track_export:
            line = ', '.join(['%.6f' % p for p in joint_pos]) + '\n'
            f.write(line.encode('utf-8'))