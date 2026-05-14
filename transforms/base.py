from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseRelightTransform(ABC):
    
    @abstractmethod
    def __call__(self, image: np.ndarray, **kwargs):
        pass
    
    @abstractmethod
    def _apply_transform(self):
        pass