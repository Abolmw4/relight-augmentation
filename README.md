# Relight Augmentation

A single-image portrait **relighting & data-augmentation toolkit**. Given one input portrait, it generates relit versions under novel illumination using three interchangeable strategies — a learned neural pipeline and two classical, filter-based approaches — making it easy to augment portrait datasets with diverse lighting conditions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [NN-Based Relighting](#1-nn-based-relighting-nnrelight)
  - [Filter-Based Relighting](#2-filter-based-relighting-filterrelight)
  - [Threshold Relighting](#3-threshold-relighting-thresholdrelight)
- [Light Configuration Files](#light-configuration-files)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
- [Limitations & Notes](#limitations--notes)
- [License](#license)

---

## Overview

Relighting is a powerful augmentation technique for training robust face-analysis models (recognition, landmark detection, attribute estimation, etc.) under varying illumination. This repository provides three complementary relighting strategies behind a common interface (`BaseRelightTransform`):

| Strategy | Type | Requires | Output |
|---|---|---|---|
| **NNRelight** | Learned (stacked-hourglass CNN) | Pretrained checkpoint + SH light configs + GPU | Relit images, one per light config |
| **FilterRelight** | Classical (gain-map filtering) | A reference image of the same subject under target lighting | One relit image (`filter.jpg`) |
| **ThresholdRelight** | Classical (extends FilterRelight) | Same as above + a gain threshold | One relit image with hard-light / low-key effect |

---

## Features

- **Neural relighting** — a stacked-hourglass network conditioned on 2nd-order spherical harmonics (SH) lighting vectors relights the L channel while preserving chroma.
- **Filter relight mapping** — computes a per-pixel luminance gain map from a source/target pair in LAB space, edge-aware smooths it with a **guided filter**, and reshapes it with a power-curve darkening pass for **smooth, natural shadows**.
- **Threshold relighting** — fades sub-threshold gain regions to black through a soft, configurable transition band for dramatic low-key effects.
- **Reusable gain filters** — a computed filter can be applied to any new image of the same subject.
- **Unified interface** — all strategies inherit from a common `BaseRelightTransform` abstract class.
- **7 preset light directions** — ready-to-use SH coefficient configs for rotating light around the subject.

---

## Project Structure

```
relight-augmentation/
├── main.py                          # Entry point / demo script
├── configs/
│   └── base_config/
│       ├── rotate_light_00.txt      # SH coefficients (9 per file) for 7 light directions
│       ├── ...
│       └── rotate_light_06.txt
├── data/
│   ├── obama.jpg                    # Sample input portrait
│   └── rotate_light_06.jpg          # Sample target-light reference
├── models/
│   └── trained_model_03.t7          # Pretrained checkpoint (NOT committed — see .gitignore)
├── transforms/
│   ├── __init__.py
│   ├── base.py                      # BaseRelightTransform (abstract base class)
│   ├── nn_relight.py                # NNRelight — learned relighting
│   ├── filter_relight.py            # FilterRelight — gain-map relighting
│   └── threshold_relight.py         # ThresholdRelight — threshold + fade-to-black
└── utils/
    ├── defineHourglass_512_gray_skip.py  # HourglassNet, HourglassBlock, lightingNet, BasicBlock
    └── utils_SH.py                       # SH basis & shading helpers (2nd-order, 9 coeffs)
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:Abolmw4/relight-augmentation.git
   cd relight-augmentation
   ```

2. **Create an environment and install dependencies**

   ```bash
   pip install torch numpy opencv-contrib-python
   ```

   > **Important:** `opencv-contrib-python` (not plain `opencv-python`) is required — the guided filter lives in `cv2.ximgproc`.

3. **Obtain the pretrained checkpoint**

   Place `trained_model_03.t7` in the `models/` directory. This file is intentionally excluded from version control (see `.gitignore`).

4. **GPU requirement**

   The NN pipeline calls `.cuda()` unconditionally — a CUDA-capable GPU with a CUDA build of PyTorch is required for `NNRelight`. The filter-based strategies run on CPU.

---

## Usage

### Quick Start

Run the demo, which executes all three pipelines on the sample portrait:

```bash
python main.py
```

Outputs are written to the project root:
- `rotate_light_00.jpg` … `rotate_light_06.jpg` — NN-relit images (one per light config)
- `filter.jpg` — filter-based relighting result
- `threshold.jpg` — threshold relighting result

### 1. NN-Based Relighting — `NNRelight`

A stacked-hourglass CNN takes the L channel of the input image plus a target SH lighting vector (9 coefficients, grayscale) and predicts the relit L channel. Chroma (a, b) is preserved; the result is converted back to BGR.

```python
from transforms.nn_relight import NNRelight

nn_relight = NNRelight()
nn_relight(
    img="data/obama.jpg",            # path to input portrait
    model_folder="models",           # folder containing trained_model_03.t7
    save_folder=".",                 # output directory
    light_configs="configs/base_config"  # folder of SH .txt configs
)
```

One output image is produced per `.txt` file in `light_configs`, named after the config (e.g. `rotate_light_03.jpg`). The input is resized to 512×512 internally and restored to its original resolution on output.

### 2. Filter-Based Relighting — `FilterRelight`

Computes a per-pixel LAB-luminance gain map `L_target / (L_source + ε)` from a source/target pair of the same subject, clips it to `[0.1, 10]`, smooths it with an edge-aware **guided filter**, and applies a power-curve darkening pass so low-gain regions fall off smoothly into shadow. The resulting filter is stored and can be applied to any new image of the same subject.

```python
import cv2
from transforms.filter_relight import FilterRelight

filter_relight = FilterRelight(
    image_sourc="data/obama.jpg",            # source lighting
    image_target="data/rotate_light_06.jpg"  # desired lighting
)
filter_relight(save_folder=".", source_image=cv2.imread("data/obama.jpg"))
# writes ./filter.jpg
```

### 3. Threshold Relighting — `ThresholdRelight`

Extends `FilterRelight` with a soft threshold on the gain map: regions whose gain falls below `threshold` are faded to black over a `softness`-wide linear ramp, producing a dramatic hard-light / low-key look.

```python
import cv2
from transforms.threshold_relight import ThresholdRelight

threshold_relight = ThresholdRelight(
    image_sourc="data/obama.jpg",
    image_target="data/rotate_light_06.jpg",
    threshold=1.0,   # gains below this are faded out
    softness=0.5     # width of the transition band (default: 0.5)
)
result = threshold_relight(source_image=cv2.imread("data/obama.jpg"))
cv2.imwrite("./threshold.jpg", result)
```

---

## Light Configuration Files

Each `.txt` file in `configs/base_config/` contains **9 spherical-harmonics coefficients** (one per line) describing a 2nd-order SH lighting environment:

```
1.0841...
-0.4642...
...
```

The seven provided configs (`rotate_light_00` – `rotate_light_06`) encode seven light directions rotating around the subject. Add your own `.txt` files in the same format to generate additional directions.

---

## Pipeline Architecture

```
main.py
 ├── NNRelight ──────────────► HourglassNet (stacked hourglass, grayscale L channel)
 │      │                         ├── BasicBlock
 │      │                         ├── HourglassBlock (recursive encoder–decoder)
 │      │                         └── lightingNet (SH-conditioned bottleneck)
 │      └── utils_SH.py ────► SH_basis / get_shading (2nd-order SH shading)
 ├── FilterRelight ──────────► gain map → guided filter → shadow curve → apply
 └── ThresholdRelight ───────► extends FilterRelight with soft-threshold masking
```

All three classes inherit from `BaseRelightTransform` (`transforms/base.py`), which defines a common interface:

```python
class BaseRelightTransform(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, **kwargs): ...
    @abstractmethod
    def _apply_transform(self): ...
```

---

## Requirements

| Dependency | Notes |
|---|---|
| Python ≥ 3.7 | |
| PyTorch (CUDA build) | NN pipeline requires GPU |
| NumPy | |
| opencv-contrib-python | Required for `cv2.ximgproc.guidedFilter` |
| `models/trained_model_03.t7` | Pretrained checkpoint — not committed to the repo |

---

## Limitations & Notes

- **GPU only (NN path):** `NNRelight` calls `.cuda()` unconditionally; there is no CPU fallback.
- **Class-level gain filter:** `FilterRelight.GAIN_FILTER` is a class attribute, so all instances share the most recently computed filter. Construct instances sequentially if you work with multiple subjects.
- **`main.py` guard:** the demo uses `if __name__:` (always truthy) rather than the conventional `if __name__ == "__main__":`.
- **Portrait-oriented:** the pretrained model and the SH hemisphere assumption target frontal face portraits; results on arbitrary scenes are not guaranteed.

---

## License

This project is released under the terms in the [LICENSE](LICENSE) file. If no license file is present, please contact the repository owner before use.
