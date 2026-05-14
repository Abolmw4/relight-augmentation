from transforms.nn_relight import NNRelight
from transforms.filter_relight import FilterRelight
from transforms.threshold_relight import ThresholdRelight
import cv2


def main():
    
    nn_relight = NNRelight()
    filter_relight = FilterRelight(image_sourc="data/obama.jpg", image_target="data/rotate_light_06.jpg")
    thresholdrelight = ThresholdRelight(image_sourc="data/obama.jpg", image_target="data/rotate_light_06.jpg", threshold=1.0)
    
    nn_relight(img="data/obama.jpg", model_folder="models", save_folder=".", light_configs="configs/base_config")
    filter_relight(save_folder=".", source_image=cv2.imread("data/obama.jpg"))
    result = thresholdrelight(source_image=cv2.imread("data/obama.jpg"))
    cv2.imwrite("./threshold.jpg", result)
    
if __name__:
    main()
    