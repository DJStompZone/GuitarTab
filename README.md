# GuitarTab

## Installation
```
conda create --name guitar-tab python==3.10
conda activate guitar-tab
pip install -r requirement.txt
```

## Check [docs/UPDATES.md](docs/UPDATES.md) for new documentation

## Check [docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md) for output token format

## Codebase
```
Project
|-- DadaGP-v1.1/  ---> raw dataset
|-- Dataset/      ---> select dataset
|-- Dataset_midi/ ---> MIDI dataset
|-- gp2jams/
|-- jams2midi/ 
|-- get_dataset.py
|-- requirement.txt
|-- README.md
```
`Dataset/` contains the selection gp files after execute data selection and `Dataset_midi/` contains the midi files after execute the pre-processing.

## Dataset Selection
```
python get_dataset.py
```
Use `get_dataset.py` to select only one-track and 6-string guitar gp files. 

Total select 5185 gp files.

## Pre-processing 
- gp2jams
- jams2midi
 
First stage: Convert gp files to jams files
```
cd gp2jams
python process_guitrapro.py
```
Change the `gpro_dir` parameter in `process_guitrapro.py` for your download DadaGP dataset folder name.

Second stage: Convert jams files to midi files
```
cd jams2midi
bash jams2midi.sh
```
Change the dir name you set in bash `python JAMS-to-MIDI.py <targetDir> <outputDir> <keyswitch_config>`

Final get the 14751 midi files.

## Train
```
bash scripts/train.sh
```
The training log and checkpoint will save in outputs folder.

## Inference & Visualize
```
python inference.py +checkpoint_path=./outputs/<exp_dir>/best_model.pt
```
- Inference the best checkpoint model generate the tablatures and calculate the metrics scores below.
  - Token accuracy
  - Pitch accuracy
  - Tab accuacy
  - Difficulty
- After that will visualize few sample for generate tab & groundtruth tab.


## Plot Loss Curve
```
python plot_loss.py
```
change `json_path` in plot_loss.py for your training log folder path. The loss_curve picture will save in your training log folder.

## DadaGP Dataset
### Token
```
artist:unknown_artist
downtune:0
tempo:100
start
new_measure
clean0:note:s4:f0
nfx:let_ring
wait:240
clean0:note:s4:f0
nfx:tie
wait:240
...
```
- new_measure: Means bar
- clean0: Guitar sound
- note:s4:f0: Note evnet on string 4 at fret 0
- nfx: play technique
- wait: duration in ticks (480 ticks = quarter note)


## Citation
DadaGP:
```
@inproceedings{dadagp2021,
  author = {Sarmento, Pedro and Kumar, Adarsh and Carr, CJ and Zukowski, Zack and Barthet, Mathieu and Yang, Yi-Hsuan},
  booktitle = {Proceedings of the 22nd International Society for Music Information Retrieval Conference},
  title = {{DadaGP: a Dataset of Tokenized GuitarPro Songs for Sequence Models}},
  url = {https://archives.ismir.net/ismir2021/paper/000076.pdf},
  year = {2021}
}
```
SynthTab:
```
@inproceedings{synthtab2024,
  title={SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription},
  author={Zang, Yongyi and Zhong, Yi and Cwitkowitz, Frank and Duan, Zhiyao}
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```
