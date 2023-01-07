### How to use:

clone this repo and cd into it
```bash
git clone --recursive git@github.com:scott306lr/yoloMOT_Streaming.git
cd yoloMOT_Streaming
```

create new conda environment and activate it
```bash
conda create --name yoloMOT
conda activate yoloMOT
```

install requirements
```bash
pip install -r requirements.txt
```

(optional) If not cloned recursively:\
clone submodule (yolov5)
```bash
git submodule update --init --recursive
```


Run
```bash
python track.py --source 0
```
