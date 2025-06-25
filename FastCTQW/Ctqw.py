import numpy as np
import networkx as nx
from .Operator import Operator
from .State import State

class Ctqw:
    def __init__(
            self,
            data:  np.ndarray | int,
            initState:State, # amp向量 
            gamma: float = 1,
            laplacian: bool = False,
            dtype: np.complex64 | np.complex128 = np.complex64,
            device: str = "cpu",
            ctqwId: str = "Undef") -> None:


        if isinstance(data, np.ndarray):
                
            self._n = data.shape[0]
            self._device = device
            self.ctqwId = ctqwId
            self._initStator = initState
            self._finalStator = State.getSingleNodeState(self._n, 0) # |10..000\rangle 作为最终态的初始化
            self._Operator = Operator.from_numpy_array(data, gamma, laplacian, device)
            
        elif isinstance(data, int):
            self._n = data
            self.ctqwId = ctqwId
            self._device = device
            self._initStator = initState
            self._finalStator = State.getSingleNodeState(self._n, 0) # |10..000\rangle 作为最终态的初始化
            self._Operator = Operator.from_num_nodes(data,dtype)

    @classmethod
    def from_numpy_array(cls, 
                         A: np.ndarray , 
                         initState:State, 
                         gamma: float = 1, 
                         laplacian: bool = False, 
                         ctqwId: str = "Undef") -> "Ctqw":
        return cls(A, initState, gamma, laplacian, ctqwId)
        

    @classmethod
    def from_networkx_graph(cls, 
                         graph: nx.Graph, 
                         initState:State, 
                         gamma: float = 1, 
                         laplacian: bool = False, 
                         ctqwId: str = "Undef",
                         dtype = np.float32) -> "Ctqw":
        
        if graph.is_directed():
            raise TypeError("graph is a directed graph, but Ctqw recive a undirected graph")
        
        return cls(nx.to_numpy_array(graph, dtype = dtype), initState, gamma, laplacian, ctqwId)
    
    @classmethod
    def from_adjacency_list(cls, 
                         adj_list: nx.Graph, 
                         initState:State, 
                         time: float = 1, 
                         gamma: float = 1, 
                         laplacian: bool = False, 
                         ctqwId: str = "Undef",
                         dtype = np.float32) -> "Ctqw":

        ...
        
    
    @classmethod
    def from_num_nodes(cls, n: int, dtype: np.complex64 | np.complex128 = np.complex64, device = "cpu") -> "Ctqw":
        return cls(n, dtype, device = "cpu")
    
    def reset_numpy_array(self, A: np.ndarray,
            initState:State, # amp向量 
            gamma: float = 1,
            laplacian: bool = False):
        
        self._n = A.shape[0]
        self._Operator.reset_numpy_array(A, gamma, laplacian)
        self._initStator = initState
        self._finalStator = State.getSingleNodeState(self._n, 0) # |10..000\rangle 作为最终态的初始化

    def reset_networkx_graph(self, nx_grpah: nx.Graph,initState:State, gamma: float, laplacian: bool = False):
        A = nx.to_numpy_array(nx_grpah)
        self.reset_numpy_array(A,initState, gamma, laplacian)

    def runWalk(
            self,
            time: float = 0):
        self._Operator.buildExpmOperator(time)
        self._finalStator.setState(self._Operator.getOperator() @ self._initStator.getState())
        
    def getFinalState(self,):
        return self._finalStator
    
    