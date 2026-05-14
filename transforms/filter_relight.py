from .base import BaseRelightTransform
import numpy as np
import cv2
import os

class FilterRelight(BaseRelightTransform):
    GAIN_FILTER = None
    
    def __init__(self, image_sourc: str, image_target: str):    
        try:
            img_x = cv2.imread(image_sourc)
            img_y = cv2.imread(image_target)
            FilterRelight.GAIN_FILTER, _ = self.__create_lighting_filter(img_x, img_y)
            FilterRelight.GAIN_FILTER = self.__apply_darkening_from_gain(smooth_gain=FilterRelight.GAIN_FILTER, shadow_strength=0.7)
        except RuntimeError as error:
            print(f"{error}")
        except FileExistsError as error:
            print(f"{error}")
        
    
    def __call__(self, save_folder: str,  **kwargs):
        result: np.ndarray = self._apply_transform(**kwargs)
        cv2.imwrite(os.path.join(save_folder, "filter.jpg"), result)

    def _apply_transform(self, source_image: np.ndarray):
        lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = lab[:, :, 0] * FilterRelight.GAIN_FILTER
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
    
    
    def __create_lighting_filter(self, img_x: np.ndarray, img_y: np.ndarray, epsilon: float=1e-5):
        lab_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2LAB).astype(np.float32)

        lx = lab_x[:, :, 0]
        ly = lab_y[:, :, 0]

        gain_map = ly / (lx + epsilon)

        gain_map = np.clip(gain_map, 0.1, 10.0)
        radius = 100
        eps = 0.1 * (255 ** 2)
        smooth_gain = cv2.ximgproc.guidedFilter(guide=lx.astype(np.uint8), src=gain_map, radius=radius, eps=eps)
        return smooth_gain, lab_x
        


    def __apply_darkening_from_gain(self, smooth_gain: np.ndarray, shadow_strength: float = 0.4, curve_power: float = 1.5):
        """
        Darkens the lighting filter itself, without using the image (L-channel).
        Only modifies `smooth_gain` directly.
        """

        # Normalize gain to [0, 1]
        g = smooth_gain.astype(np.float32)
        g_norm = g / g.max()

        # Create a darkening curve entirely from the gain itself
        # High gain → brighter areas → get reduced less
        # Low gain → naturally darker → get reduced more
        darkening_curve = (1.0 - shadow_strength * (1.0 - g_norm) ** curve_power)

        # Final filter
        final_gain = g * darkening_curve

        return final_gain