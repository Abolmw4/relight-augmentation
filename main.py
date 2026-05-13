from transforms.nn_relight import NNRelight
from transforms.filter_relight import FilterRelight
import cv2


def main():
    # nn_relight = NNRelight()
    # nn_relight(img="data/obama.jpg", model_folder="models", save_folder=".", light_configs="configs/base_config")

    filter_relight = FilterRelight(image_sourc="data/obama.jpg", image_target="data/rotate_light_06.jpg")
    filter_relight(save_folder=".", source_image=cv2.imread("data/obama.jpg"))
    
if __name__:
    main()