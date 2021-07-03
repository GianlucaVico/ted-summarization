# BuildDataset

Script to download the data and prepare the files

## Content

**Files**:

* Dataset_integration.ipynb : notebook to merge Amara, TED and MuST-C. Inital filtering
* EvalData.ipynb : notebook to filter the dataset and compute some statistic (can contain old/incorrect code)
* prepare_kenlm.py : generate files to train a KenLM n-gram model (not used in the final project)
* Rasample.ipynb : old notebook used to resample the talks
* split_audio.py : retrieve and split the talks, generate pickle files that pairs the fragments and the transcripts
* split_audio_job.sh : script to run split_audio.py
* split_on_silence.py : retrieve and split the talks on silence, generate pickle files that pairs the fragments and the transcripts
* split_silence.sh : script to run split_on_silence.py
* ted_align.sh : run align.py
