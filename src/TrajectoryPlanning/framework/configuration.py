import config
import json
import os
import utils.fileio
import utils.print
import utils.string_utils
from typing import Any, Tuple

class Configuration:
    def __init__(self, name: str) -> None:
        self.__configs = {}
        self.__name = name
        self.__file_path = config.Common.ConfigDir + self.__name + '.txt'
        self.__load()

    def get(self, field: Tuple) -> Any:
        key = field[0]
        default = field[1]
        if key in self.__configs:
            return self.__configs[key]
        self.__configs[key] = default
        self.__save()
        return default

    def __save(self) -> None:
        path = utils.string_utils.to_parent_path(self.__file_path)
        utils.fileio.mktree(path)
        with open(self.__file_path, 'w') as f:
            json.dump(self.__configs, f, separators=(',\n', ': '))

    def __load(self) -> bool:
        if not os.path.isfile(self.__file_path):
            return False
        with open(self.__file_path, 'r') as f:
            self.__configs = json.load(f)
        utils.print.put("Configs loaded (" + self.__name + ")")
        return True

global_configs = Configuration('global')