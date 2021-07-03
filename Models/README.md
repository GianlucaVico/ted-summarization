# Models

Scripts to train and test the models (ASR, summarization and cascade)

The train_*.sh files can also evaluate the model. Comment/Uncomment the line "--eval" to change this behaviour.

## Content

**Files**:

* Pegasus_training.ipynb : evaluate the training of Pegasus
* test_cascade.py : script to test the cascade model (HACK to avoid running it in one go)
* test_cascade_job.sh : run test_cascade.py; uncomment the lines to change the ASR and summarization models
* train_pegasus.py : train Pegasus
* train_pegasus_job.sh : run train_pegasus.py
* train_pegasus_job_[dropout|embeddings|encoder|no_embeddings].sh : run train_pegasus.py with an higher dropout or with freezing part of the model
* train_pegasus_job_real.sh : train Pegasus on the ASR transcripts
* train_wav2vec2.py : train, test or transcribe wav2vec2
* train_wav2vec2_job.sh : run train_wav2vec2.py on the TED+MuST-C dataset
* train_wav2vec2_job_mustc.sh : run train_wav2vec2.py on the MuST-C dataset
* transcribe_wav2vec2_job.sh : run train_wav2vec2.py for transcription
* Wav2Vec2_training.ipynb : evaluate the training of wav2vec 2.0
