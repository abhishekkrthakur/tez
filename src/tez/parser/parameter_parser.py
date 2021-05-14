import copy
from dataclasses import dataclass
from typing import Dict, List, Union

from sklearn.model_selection import ParameterGrid


@dataclass
class TezParameterParser:
    parameters: Dict[str, Union[int, str, float, List[Union[int, str, float]]]]

    def parse_params(self):
        for param_name in self.parameters:
            if not isinstance(self.parameters[param_name], list):
                self.parameters[param_name] = [self.parameters[param_name]]

        list_params = list(ParameterGrid(self.parameters))
        return list_params
