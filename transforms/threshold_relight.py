from .filter_relight import FilterRelight
import numpy as np
import cv2

class ThresholdRelight(FilterRelight):

    def __init__(self, image_sourc: str, image_target: str, threshold: float, softness: float = 0.5):
        super().__init__(image_sourc, image_target)
        self.threshold = threshold
        self.softness = softness  # width of transition band

    def __call__(self, **kwargs):
        return self._apply_transform(**kwargs)
    
    def _apply_transform(self, source_image: np.ndarray):
        lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        gain = FilterRelight.GAIN_FILTER

        # soft mask (0–1)
        #  mask = 1 when gain >= threshold
        #  mask = 0 when gain <= threshold - softness
        #  linear ramp between
        mask = np.clip((gain - (self.threshold - self.softness)) / self.softness, 0.0, 1.0)

        # relight L channel
        relit_L = lab[:, :, 0] * gain

        # dark version (full black)
        dark_L = np.zeros_like(lab[:, :, 0])

        # blend soft
        lab[:, :, 0] = relit_L * mask + dark_L * (1 - mask)
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # apply soft blending to final RGB
        result = (result * mask[..., None]).astype(np.uint8)

        return result
