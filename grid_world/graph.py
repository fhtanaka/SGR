from node import Node
from typing import Dict
from arg_parser import Parameters

class Graph:
    def __init__(self) -> None:
        self.d_nodes: Dict[str, Node] = {}
         