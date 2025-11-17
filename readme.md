## Introduction

The demo to extract the visual related voice from the given video. The input is a cropped face video with overlap speech, the output is the same video with the extracted voice.

## Requirements

Install the environment with the packages in requirements, the torch version ought to match your cuda version

```
pip install -r requirement.txt
```

## Code sstructure

```
    # se_demo
    # ├── data (folder for the given raw video)
    # │   ├── 001.mp4
    # │   ├── ...
    # ├── exps (folder for the outputs)
    # │   ├── 001.wav (overlapped speech)
    # │   ├── 001_res.wav (extracted speech)
    # │   ├── 001_res.avi (result video with the extracted speech)
    # │   ├── ...
    # │── pretrain_model (folder for the trained model)
    # │   ├── frontend.pt (visual frontend is used to extract features)
    # │   ├── backend.pt (backend is used to extract the sound)
    # main.py
    # model.py (Contain the entire model structure)
    # modules.py (Contain the related classes/functions in model)
    # tools.py (Functions to load/save the audio/video)
    # requirement.txt
```

## Usage:

`python main.py`, then check the results in **exps** folder

Can put the different video in the data folder, change the name in main file.

## Notice: 

The input videos are cropped face videos, other kinds of videos need more preprocessing (face detection + tracking)

**001.mp4** is an easy case so the performance is good. Some videos might perform bad.