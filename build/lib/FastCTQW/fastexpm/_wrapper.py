import numpy as np
from numpy.typing import NDArray
from typing import Annotated

Matrix = NDArray[np.floating | np.integer]
SquareMatrix = Annotated[Matrix, "A.shape[0] == A.shape[1]"]

from . import _fastexpm_core 
#  ExpMatFloat32, ExpMatFloat64, ExpMatComplex64, ExpMatComplex128 

class ExpMatFloat32:
    
    def __init__(self, N: int):
        self.N = N
        self._backend = _fastexpm_core.ExpMatFloat32(N)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.free()
        except Exception as cleanup_error:
            # 可选：你可以在这里记录清理失败的日志
            import warnings
            warnings.warn(f"[fastexpm] Failed to free resources in __exit__: {cleanup_error}")
        
        # 如果有异常发生，不吞掉它，保持向上传播
        return False
    
    def run(self, A: SquareMatrix) -> np.ndarray:
        if not isinstance(A, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if A.shape != (self.N, self.N): # Assuming N is stored in the C++ object
            raise ValueError(f"Input matrix must have shape ({self.N}, {self.N}).")
    
        return self._backend.run(A).reshape(self.N, self.N)
    
    def free(self):
        self._backend.free()
        
class ExpMatFloat64:
    
    def __init__(self, N: int):
        self.N = N
        self._backend = _fastexpm_core.ExpMatFloat64(N)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.free()
        except Exception as cleanup_error:
            # 可选：你可以在这里记录清理失败的日志
            import warnings
            warnings.warn(f"[fastexpm] Failed to free resources in __exit__: {cleanup_error}")
        
        # 如果有异常发生，不吞掉它，保持向上传播
        return False
    
    def run(self, A: SquareMatrix) -> np.ndarray:
        if not isinstance(A, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if A.shape != (self.N, self.N): # Assuming N is stored in the C++ object
            raise ValueError(f"Input matrix must have shape ({self.N}, {self.N}).")
    
        return self._backend.run(A).reshape(self.N, self.N)
    
    def free(self):
        self._backend.free()
class ExpMatComplex64:
    
    def __init__(self, N: int):
        self.N = N
        self._backend = _fastexpm_core.ExpMatComplex64(N)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.free()
        except Exception as cleanup_error:
            # 可选：你可以在这里记录清理失败的日志
            import warnings
            warnings.warn(f"[fastexpm] Failed to free resources in __exit__: {cleanup_error}")
        
        # 如果有异常发生，不吞掉它，保持向上传播
        return False
    
    def run(self, A: SquareMatrix) -> np.ndarray:
        if not isinstance(A, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if A.shape != (self.N, self.N): # Assuming N is stored in the C++ object
            raise ValueError(f"Input matrix must have shape ({self.N}, {self.N}).")
    
        return self._backend.run(A).reshape(self.N, self.N)
    
    def free(self):
        self._backend.free()
        
class ExpMatComplex128:
    
    def __init__(self, N: int):
        self.N = N
        self._backend = _fastexpm_core.ExpMatComplex128(N)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.free()
        except Exception as cleanup_error:
            # 可选：你可以在这里记录清理失败的日志
            import warnings
            warnings.warn(f"[fastexpm] Failed to free resources in __exit__: {cleanup_error}")
        
        # 如果有异常发生，不吞掉它，保持向上传播
        return False
    
    def run(self, A: SquareMatrix) -> np.ndarray:
        if not isinstance(A, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if A.shape != (self.N, self.N): # Assuming N is stored in the C++ object
            raise ValueError(f"Input matrix must have shape ({self.N}, {self.N}).")
    
        return self._backend.run(A).reshape(self.N, self.N)
    
    def free(self):
        self._backend.free()