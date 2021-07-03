#!/usr/bin/env python3

import joblib
import pandas as pd
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


def do_work(data, name, files):
    fragm_list = [None for i in range(data.shape[0])]
    transcript_list = [None for i in range(data.shape[0])]
    id_list = [None for i in range(data.shape[0])]
    save = 0
    for i in range(0,data.shape[0]):
        print(f"{i+1}/{data.shape[0]}")
        id_ = data.iloc[i]["id"]
        set_ = None
        if data.iloc[i]["must_c"]:
            set_ = text_tools.DATASETS.Mustc
        elif data.iloc[i]["ted"]:
            set_ = text_tools.DATASETS.Ted
        # elif data.iloc[i]["amara"]:
        #    set_ = text_tools.DATASETS.Amara
        if set_ is not None:
            path = audio_tools.get_audio_path(id_, set_, **files)
            #print("Audio:", path)
            if path == "" and id_ > 0:
                print(f"Missing {id_} from {set_}")
            else:
                transcript, _, _ = text_tools.get_transcript(id_, set_, False, **files)
                fragm = audio_tools.split_talk(path, transcript, rate=16000, original_rate=0)
                transcript = transcript["transcript"].tolist()
                id_ = [id_ for j in transcript]
                fragm_list[i] = fragm
                transcript_list[i] = transcript
                id_list[i] = id_

        if i % 100 == 0 and i != 0:
            # To list
            fragm_list = [i for i in fragm_list if i is not None]
            transcript_list = [i for i in transcript_list if i is not None]
            id_list = [i for i in id_list if i is not None]

            fragm_list = [j.numpy() for i in fragm_list for j in i]
            transcript_list = [j for i in transcript_list for j in i]
            id_list = [j for i in id_list for j in i]

            # Save
            print("Writing", save)
            with open(f"{name}_{save}.pkl", "wb")as f:
                tmp = [id_list, fragm_list, transcript_list]
                joblib.dump(tmp, f, compress=True)
            save += 1
            del tmp

            # Restart
            fragm_list = [None for i in range(data.shape[0])]
            transcript_list = [None for i in range(data.shape[0])]
            id_list = [None for i in range(data.shape[0])]


    fragm_list = [i for i in fragm_list if i is not None]
    transcript_list = [i for i in transcript_list if i is not None]
    id_list = [i for i in id_list if i is not None]

    fragm_list = [j.numpy() for i in fragm_list for j in i]
    transcript_list = [j for i in transcript_list for j in i]
    id_list = [j for i in id_list for j in i]

    # Save
    with open(f"{name}_{save}.pkl", "wb")as f:
        tmp = [id_list, fragm_list, transcript_list]
        joblib.dump(tmp, f, compress=True)
    save += 1
    del tmp


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv(dataset)
    df = df[df["id"] >= 0]

    test_data = df[(df["test"]==True) & (df["drop"]==False)]
    train_data = df[(df["train"]==True) & (df["drop"] == False)]

    mustc_test_data = df[(df["test"]==True) & (df["drop"]==False) & (df["must_c"]==True)]
    mustc_train_data = df[(df["train"]==True) & (df["drop"] == False) & (df["must_c"]==True)]
    print("--done")

    print("Loading files...")
    files = text_tools.get_files()
    print("--done")

    print("Making test dataset:", test_data.shape[0])
    do_work(test_data, "/hpcwork/aachen_id/test_audio/audio", files)
    do_work(mustc_test_data, "/hpcwork/aachen_id/mustc_test_audio/audio", files)
    #do_work(mustc_test_data, ".", files)

    print("Making train dataset:", train_data.shape[0])
    do_work(train_data, "/hpcwork/aachen_id/train_audio/audio", files)
    do_work(mustc_train_data, "/hpcwork/aachen_id/mustc_train_audio/audio", files)
