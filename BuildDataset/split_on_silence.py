#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import audio_tools
import text_tools

audio_tools.MUST_C_PATH = "/work/aachen_id/MUST-C/en-cs/data"
audio_tools.AMARA_PATH = "/work/aachen_id/AMARA"
audio_tools.AMARA_DATA_PATH = "/work/aachen_id/AMARA"
audio_tools.TED_PATH = "/home/aachen_id/Documents/BSc-Thesis-AudioSumm/BuildDataset/TED"
audio_tools.TED_DATA_PATH = "/work/aachen_id/TED/Data"

text_tools.MUST_C_PATH = "/work/aachen_id/MUST-C/en-cs/data"
text_tools.AMARA_PATH = "/work/aachen_id/AMARA"
text_tools.AMARA_DATA_PATH = "/work/aachen_id/AMARA"
text_tools.TED_PATH = "/home/aachen_id/Documents/BSc-Thesis-AudioSumm/BuildDataset/TED"
text_tools.TED_DATA_PATH = "/work/aachen_id/TED/Data"

dataset = "/home/aachen_id/Documents/BSc-Thesis-AudioSumm/BuildDataset/integrated_data.csv"


def do_work(test_data, name, files):

    def save_(audio_data, lengths, tar, ids, name, save):
        skip = lambda x: x[0] is not None and x[1] is not None and x[2] is not None and x[3] is not None

        try:
            audio_data, lengths, tar, ids = zip(*filter(skip, zip(audio_data, lengths, tar, ids)))
        except ValueError:
            print("Skip saving: empty")
        else:
            extended_ids = [id_ for id_, n in zip(ids, lengths) for i in range(n)]
            audio_data = [np.array(j) for i in audio_data for j in i]

            #print("Fragments:", len(audio_data))
            #print("Avg length:", sum([len(i) for i in audio_data]) / len(audio_data))
            #input()
            # Save
            print(f"Saving {save}...")
            with open(f"{name}_audio_{save}.pkl", "wb")as f:
                df = pd.DataFrame({"id":extended_ids, "fragment": audio_data})
                joblib.dump(df, f, compress=True)
            with open(f"{name}_summ_{save}.pkl", "wb")as f:
                df = pd.DataFrame({"id":ids, "summary": tar})
                joblib.dump(df, f, compress=True)

    audio_data = ["" for i in range(test_data.shape[0])] # store the fragments
    tar = ["" for i in range(test_data.shape[0])] # store the summaries
    ids = [None for i in range(test_data.shape[0])] # store the ids (aligned with tar)
    lengths = [0 for i in range(test_data.shape[0])] # store the length of the framgents
    save = 9 #450: done
    group_size = 50
    for i, id_ in enumerate(test_data.index):
        # One set for the audio and one set for the transcript
        # MustC doesn't have the transcript
        print(i)
        if i < (group_size * save):
            pass
        else:
            set_transcript = None
            set_audio = None
            if dataset.loc[id_]["must_c"]:
                set_audio = text_tools.DATASETS.Mustc
            elif dataset.loc[id_]["ted"]:
                set_audio = text_tools.DATASETS.Ted

            if dataset.loc[id_]["ted"]:
                set_transcript = text_tools.DATASETS.Ted
            elif dataset.loc[id_]["amara"]:
                set_transcript = text_tools.DATASETS.Amara

            if set_transcript is not None and set_audio is not None:
                path = audio_tools.get_audio_path(id_, set_audio, **files)
                if path == "":
                    print(i, "-", id_, "has not audio path")
                    audio_data[i] = None
                    lengths[i] = None
                    tar[i] = None
                    ids[i] = None
                else:
                    audio_data[i] = audio_tools.get_fragments(path)
                    lengths[i] = len(audio_data[i])
                    tar[i] = text_tools.get_transcript(id_, set_transcript, **files)[2] # only the description/summary
                    ids[i] = id_
            else:
                print(i, "-", id_, "missing")
                audio_data[i] = None
                lengths[i] = None
                tar[i] = None
                ids[i] = None

            if ((i+1)% group_size)==0:
                print(f"At i=={i}")
                save_(audio_data, lengths, tar, ids, name, save)
                audio_data = ["" for i in range(test_data.shape[0])] # store the fragments
                tar = ["" for i in range(test_data.shape[0])] # store the summaries
                ids = [None for i in range(test_data.shape[0])] # store the ids (aligned with tar)
                lengths = [0 for i in range(test_data.shape[0])] # store the length of the framgents
                save += 1
        #else:
        #    print("Not saving")

    #audio_data, lengths, tar, ids = zip(*filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None and x[3] is not None, zip(audio_data, lengths, tar, ids)))

    #extended_ids = [id_ for id_, n in zip(ids, lengths) for i in range(n)]
    #audio_data = [np.array(j) for i in audio_data for j in i]

    #print("Fragments:", len(audio_data))
    #print("Avg length:", sum([len(i) for i in audio_data]) / len(audio_data))
    #input()
    # Save
    #print(f"Saving {save}...")
    #with open(f"{name}_audio_{save}.pkl", "wb")as f:
    #    df = pd.DataFrame({"id":extended_ids, "fragment": audio_data})
    #    joblib.dump(df, f, compress=True)
    #with open(f"{name}_summ_{save}.pkl", "wb")as f:
    #    df = pd.DataFrame({"id":ids, "summary": tar})
    #    joblib.dump(df, f, compress=True)
    save_(audio_data, lengths, tar, ids, name, save)
    print("--done")

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = pd.read_csv(dataset, index_col="id")
    dataset = dataset[dataset.index >= 0]

    test_data = dataset[(dataset["test"] == True) & (dataset["drop"] == False)]

    print("--done")

    print("Loading files...")
    files = text_tools.get_files()
    print("--done")

    print("Making test dataset:", test_data.shape[0])
    do_work(test_data, "/hpcwork/aachen_id/cascade/test", files)

    test_data = test_data[test_data["must_c"] == True]
    print("Making test dataset:", test_data.shape[0])
    do_work(test_data, "/hpcwork/aachen_id/cascade_mustc/test", files)
