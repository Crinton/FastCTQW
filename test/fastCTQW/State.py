import numpy as np
from numpy.typing import NDArray
type QuantumState =  NDArray[np.complex64 | np.complex128]

class State:
    def __init__(
            self,
            initState: QuantumState
    ):
        if not self._isState(initState):
            raise ValueError("initState is not a quantum state")
        
        self._state = initState 
    
    def _isState(self, state: QuantumState, tol = 1e-5) -> bool:
        """
        检查是否是量子态
        """
        if state.ndim != 1:
            print(f"警告: 输入不是一维数组。维度: {state.ndim}")
            return False
        sum_of_squares = np.sum(np.abs(state)**2)
        if not np.isclose(sum_of_squares, 1.0, atol=tol):
            print(f"警告: 量子态未归一化。模方和为 {sum_of_squares}，期望为 1.0。")
                
        return True
    @classmethod
    def getUniformSuperposition(cls, n: int) -> "State":
        """
        返回均匀叠加态的State
        """
        initial_state = np.full(n, 1.0 / np.sqrt(n), dtype=np.complex64)
        return cls(initial_state)
    
    @classmethod
    def getSingleNodeState(cls, n: int, m: int) -> "State":
        """
        振幅都在第m个节点
        """
        initial_state = np.zeros(n, dtype=np.complex64)
        initial_state[m] = 1.0
        return cls(initial_state)
    
    @classmethod
    def getLocalSuperposition(cls, n: int, markNodes: list):
        initial_state = np.zeros(n, dtype=np.complex64)
        for i in markNodes:
            initial_state[i] = 1.0
        initial_state = initial_state/np.linalg.norm(initial_state)
        return cls(initial_state)
    def getProbabilities(self) -> QuantumState:
        return np.abs(self._state)**2
    
    def setState(self, initState: QuantumState) -> None:
        if not self._isState(initState):
            raise ValueError("initState is not a quantum state")
        self._state = initState
    
    def getState(self) -> QuantumState:
        return self._state
    
    