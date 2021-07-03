# TED Talk Teaser Generation With Pre-Trained Models

## Content

The root folder contains the base files needed to run the project.

In general, the *.py files do the work, while the *.sh run the *.py on the Aachen cluster.

The *.sh files also load the libraries, set the environment variables, and pass the arguments to the Python scripts.

Sensitive information (e.g. mail addresses, ids, ...) has been removed from the code.

**Files**:
* tools.py: base class for the model, methods to create the pipelines and convert the transcripts
* text_tools.py: model for the summarization part, methods to get the transcripts and summaries, pipeline steps for the transcripts
* audio_tools.py: model for the ASR, methods to get the recordings and the fragments, method to split the recordings
* cascade_tools.py: model for the cascade
* decoders.py: various decoders for wav2vec 2.0

**Folders**:
* BuildDataset: folder with the notebooks and the scripts to download the dataset. It is not well maintained, so there can be changes
    in *_tools.py files that can break part of the notebook. This also contains the zip files with the transcripts and the pickle files with
    the prepared documents.
* Models: notebooks and scripts to train the models (summariation, ASR and cascade)

## Requirements

When not specified the version is not relevant.
Other Python packages are installed automatically to fulfil these requirements.

**Python requirements**:
* beautifulsoup4
* ctcdecode=1.0.2 : CTC decoder from DeepSpeech
* flashlight=1.0.0 (optional) : used in decoders.py, but not used in the paper.
* jiwer : package to compute WER
* joblib : for compressed pickle files (it should be a standard package)
* jupyter : to run jupyter notebooks
* matplotlib
* neuspell=1.0.0 : spell correction toolkit (see [https://github.com/neuspell/neuspell](https://github.com/neuspell/neuspell) for the installation)
* newsroom=0.1 : used to compute the extractive fragments metrics (see [https://github.com/lil-lab/newsroom](https://github.com/lil-lab/newsroom) for the installation)
* nltk
* numpy
* pandas
* pydub=0.25.1 : used to play and split audio (pip3 install pydub)
* python-Levenshtein=0.12.2 : Levenshtein distance to compute CER
* PyYAML=5.4.1 : read yaml files in the MuST-C dataset
* rouge=1.0.0 : used to compute the ROUGE scores (pip3 install rouge)
* scikit-learn : not used
* scipy=1.3.1 : dependency for neuspell
* seaborn
* sentencepiece : dependency for Pegasus' tokenizer
* spacy=3.0.5 : dependency for neuspell
* tokenizers=0.10.2 : included in transformers, originally the project used an older version and there is still some legacy code
* torch=1.7.0
* torchaudio=0.7.0
* transformers=4.6.0 : for the Hugging Face models, originally the project used an older version and there is still some legacy code
* Unidecode : used to clean the transcripts

**Other requirements**:
* gcc 9
* python 3.8.7
* cmake
* ffmpeg : used to downsample the recordings, it has to be in the PATH variable
* CUDA 11.0
* CuDNN 8.0.5
* KenLM : compiled to support 20-grams models (default is up to 6-grams)
    * For the python binding this parameter has to be passed to setup.py
* Intel MKL 2020 : required to compile flashlight (not used in the final work)
* flashlight python binding : compiled with CUDA, MKL and KenLM support, KENLM_ROOT is the base folder of KenLM (not the build folder), not used in the final work
    * It has other requirements already fulfilled on the Aachen cluster.

**Data and models**:

* The fine-tuned models are momentaneary available [here](https://drive.google.com/drive/folders/1HBFSfHxpBoII7FlhbYcuHDCGhONAe8xD?usp=sharing)
* TED is not longer using Amara (since end of 2020), therefore, the talks from Amara might no longer be available
* MuST-C is can be downloaded from their [site](https://ict.fbk.eu/must-c/)
* The talks from TED are downloaded directly from their [website](https://www.ted.com/) by using the script contained here

## Copyright information

TED talks are copyrighted by TED Conference LLC and licensed under a Creative Commons Attribution-NonCommercial-NoDerivs 4.0 
(see [https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy](https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy)) 


## Author
Gianluca Vico [gianlucavico99@gmail.com](mailto:gianluca.vico99@gmail.com)

Supervisor: Jan Niehues
