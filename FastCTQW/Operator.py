import numpy as np
import networkx as nx
from numpy.typing import NDArray
from .fastexpm import ExpMatComplex64, ExpMatComplex128
from scipy.linalg import expm
class Operator:
    """
    该类的目标是获取_expmOperator, 供CTQW进行 GEMV
    """
    def __init__(
        self,
        data: np.ndarray | int,
        gamma: float = 1,
        laplacian: bool = False,
        device: str = "cpu",
        dtype: np.complex64 | np.complex128 = np.complex64
    ) -> None:
        if isinstance(data, np.ndarray):
            if not self._is_adjacency_matrix(data):
                raise ValueError("A is not a adj matrix")
            
            self._n = data.shape[0]
            
            self._gamma = gamma
            self._laplacian = laplacian
            self._device = device
            if data.dtype in (np.float32, np.complex64):
                self._dtype = np.complex64
            elif data.dtype in (np.float64, np.complex128):
                self._dtype = np.complex128
            else:
                raise TypeError(f"A.dtype is error : {data.dtype}")
            
            self._hamiltonian = self._buildHamiltonian(data, gamma)
            
            if not self._is_hermitian(self._hamiltonian):
                raise ValueError("_hamiltonian is not a hamiltonian matrix")
            
            
            self._expMator = self._initExpMator(self._n, dtype)

        elif isinstance(data, int):
            self._n = data
            self._device = device
            self._expMator = self._initExpMator(data, dtype)
    def free(self):
        self._expMator.free()

    def _buildHamiltonian(self, A: np.ndarray, gamma: float) -> NDArray[np.complex64 | np.complex128]:
        if self._laplacian:
            return -gamma * np.asarray((np.diag(A.sum(axis = 1)) - A),dtype = self._dtype)
        else:
            return -gamma * np.asarray(A, self._dtype)
        
    def _buildUnitory(self, time):
        return np.asarray(-1j * self._hamiltonian * time, dtype = self._dtype)
    
    def _initExpMator(self, n: int,  dtype: np.complex64 | np.complex128 = np.complex64) -> None:
        if self._device == "cpu":
            return
        elif self._device == "cuda":
            n = self._n if n is None else n
            if dtype == np.complex64:
                return ExpMatComplex64(self._n)
            elif dtype == np.complex128:
                return ExpMatComplex128(self._n)
            else:
                raise TypeError(f"self._phaseFactor.dtype: {dtype} is error")

    def buildExpmOperator(self, time: float) -> None:
        """
        得到$e^{-iHt}矩阵$
        """
        if self._device == "cpu":
            
            self._expmMat =  expm(self._buildUnitory(time))
        elif self._device == "cuda":
            self._expmMat = self._expMator.run(self._buildUnitory(time))
    
    def _is_adjacency_matrix(self, A: np.ndarray) -> bool:
        if A.ndim != 2:
            print(f"Error: Matrix is not 2D. Dimensions: {A.ndim}")
            return False
        if A.shape[0] != A.shape[1]:
            print(f"Error: Matrix is not square. Shape: {A.shape}")
            return False
        if not np.array_equal(A, A.T):
            print("Error: Matrix is not symmetric (expected for undirected graph).")
            return False
        return True
    
    def _is_hermitian(self, hamiltonian) -> bool:
        """Checks if the adjacency matrix is Hermitian.
        Parameters
        ----------
        hamiltonian : np.ndarray
            Adjacency matrix.
        Returns
        -------
        bool
            True if Hermitian, False otherwise.
        """
        return np.allclose(hamiltonian, hamiltonian.conj().T)

    @classmethod
    def from_networkx_graph(cls, nx_graph: nx.Graph, gamma:float = 1,  laplacian: bool = False, device: str = "cpu") -> "Operator":
        A = nx.to_numpy_array(nx_graph)
        return cls(A, gamma, laplacian, device)
    
    @classmethod
    def from_numpy_array(cls, A: np.ndarray, gamma:float = 1,  laplacian: bool = False, device: str = "cpu") -> "Operator":
        return cls(A, gamma, laplacian, device)
    
    @classmethod
    def from_adjacency_list(cls, adj_list: list, gamma:float = 1,  laplacian: bool = False, device: str = "cpu"):
        ...
        
    @classmethod
    def from_num_nodes(cls, n: int, dtype: np.complex64 | np.complex128 = np.complex64):
        return cls(n, dtype)
    def reset_numpy_array(self, A: np.ndarray, gamma:float = 1, laplacian: bool = False):
        """
        
        """
        if A.dtype in (np.float32, np.complex64):
            self._dtype = np.complex64
        elif A.dtype in (np.float64, np.complex128):
            self._dtype = np.complex128
        else:
            raise TypeError(f"A.dtype is error : {A.dtype}")
        
        self._gamma = gamma
        self._laplacian = laplacian
        self._hamiltonian = self._buildHamiltonian(A, gamma)
        
        if not self._is_hermitian(self._hamiltonian):
            raise ValueError("_hamiltonian is not a hamiltonian matrix")
        
        
        if self._expMator is not None:
           self._expMator = self._initExpMator(self._n, self._dtype)

        
    def reset_networkx_graph(self, nx_grpah: nx.Graph, gamma: float, laplacian: bool = False, device: str = "cpu"):
        A = nx.to_numpy_array(nx_grpah)
        self.reset_numpy_array(A, gamma, laplacian, device)
        
    def reset_adjList(self, adj_list, gamma: float, time: float, laplacian: bool = False, device: str = "cpu") :
        ...
        

    def setHamiltonian(self, hamiltonian:NDArray):
        """Sets the hamiltonian for the walk.

        Parameters
        ----------
        hamiltonian : np.ndarray
            Hamiltonian of the graph.
        """
        self._hamiltonian = hamiltonian
        
    def setexpmOperator(self, newexpmMat):
        self._expmMat = newexpmMat
    
    def getOperator(self):
        return self._expmMat