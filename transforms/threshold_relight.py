from .filter_relight import FilterRelight
import numpy as np
import cv2

class ThresholdRelight(FilterRelight):
    def __init__(self, image_sourc: str, image_target: str, threshold: float):
        super().__init__(image_sourc, image_target)
        self.threshold = threshold
            
    def __call__(self, **kwargs):
        return self._apply_transform(**kwargs)
    
    def _apply_transform(self, source_image: np.ndarray):
        lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        gain = FilterRelight.GAIN_FILTER
        mask = gain >= self.threshold   # pixels we keep

        # apply relighting where mask true
        lab[:, :, 0][mask] = lab[:, :, 0][mask] * gain[mask]

        # pixels below threshold → completely dark
        lab[:, :, 0][~mask] = 0

        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # force full black on masked pixels
        result[~mask] = 0

        return result
