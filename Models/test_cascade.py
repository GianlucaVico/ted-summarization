import sys

sys.path.append("../")

import cascade_tools
import text_tools
import audio_tools
import argparse
import pandas as pd
import joblib
import tools
import glob

def get_model(device, asr_model=None, summ_model=None, decoder="greedy", spell=False):
    summ = text_tools.SimpleSummarizationModel(
        device,
        text_tools.MODELS.Pegasus,
        model=summ_model,
    )

    asr = audio_tools.SimpleASRModel(
        device=device, model=asr_model, decoder=decoder, spell=spell
    )

    return cascade_tools.SimpleCascadeModel(asr, summ)

def test_from_files(args, target_pipeline, model):
    files = text_tools.get_files()

    dataset = pd.read_csv(args.dataset, index_col="id")
    dataset = dataset[dataset.index >= 0]

    test_data = dataset[(dataset["test"] == True) & (dataset["drop"] == False)]
    if args.mustc:
        test_data = test_data[test_data["must_c"] == True]
    audio_data = ["" for i in range(test_data.shape[0])]
    tar = ["" for i in range(test_data.shape[0])]
    for i, id_ in enumerate(test_data.index):
        set_ = None
        if dataset.loc[id_]["must_c"]:
            set_ = text_tools.DATASETS.Mustc
        elif dataset.loc[id_]["ted"]:
            set_ = text_tools.DATASETS.Ted
        if set_ is not None:
            audio_data[i] = audio_tools.get_audio_path(id_, set_, **files)
            tar[i] = text_tools.get_transcript(id_, set_, **files)[2] # only the description/summary

    tar = target_pipeline(tar)
    print("--done")

    #print("Loading model...")
    #model = get_model(device, args.asr_model, args.summ_model)
    #print("--done")
    print("Starting test...")
    metrics = model.evaluate(audio_data, tar)
    print("--done")
    print("Metrics:", metrics)

def test_from_saved_data(args, target_pipeline, model):
    # same ordering
    audio_parts = sorted(glob.glob(args.saved_audio + "/*_audio_*.pkl"))#[9:]
    summ_parts = sorted(glob.glob(args.saved_summ + "/*_summ_*.pkl"))#[9:]
    print("Audio parts:", len(audio_parts))
    print("Summ parts:", len(summ_parts))
    assert len(audio_parts) == len(summ_parts), "Missing parts in the saved dataset"

    tot = 0 # used to compute the avg
    r1 = 0 # partial results
    r2 = 0
    rl = 0
    for n, (a, t) in enumerate(zip(audio_parts, summ_parts)):
        print(f"Part {n+1}/{len(audio_parts)}")
        print(a)
        print(t)
        audio = joblib.load(a)
        text = joblib.load(t)
        # make a list of summaries and of list of tensors
        # same indexes corresponds to the same talk
        # summ[i]: str -> audio[i]: List[np.array]
        tar = [None for i in range(text.shape[0])]
        audio_data = [None for i in range(text.shape[0])]
        for i in range(len(tar)):
            id_ = text.iloc[i]["id"]
            tar[i] = text.iloc[i]["summary"]
            audio_data[i] = audio[audio["id"] == id_]["fragment"].tolist()
        tar = target_pipeline(tar)
        metrics = model.evaluate(audio_data, tar)
        tot += len(tar)
        r1 += metrics["rouge-1"]["f"] * len(tar)
        r2 += metrics["rouge-2"]["f"] * len(tar)
        rl += metrics["rouge-l"]["f"] * len(tar)
        print("Partial rouge-1:", r1)
        print("Partial rouge-2:", r2)
        print("Partial rouge-l:", rl)
        print("Samples:", tot)
    r1 /= tot
    r2 /= tot
    rl /= tot
    tmp = {"rouge-1": r1, "rouge-2": r2, "rouge-l": rl}
    print("Metrics:", tmp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cascade")
    parser.add_argument(
        "-t",
        "--ted",
        type=str,
        help="Path to the folder with the transcripts and audio from ted",
    )
    parser.add_argument(
        "-a",
        "--amara",
        type=str,
        help="Path to the folder with the transcripts and audio from amara",
    )
    parser.add_argument(
        "-mc",
        "--must-c",
        type=str,
        help="Path to the folder with the transcripts and audio from MuSt-C",
    )
    parser.add_argument(
        "-td", "--ted-data", type=str, help="Path to data_urls.json file for ted"
    )
    parser.add_argument(
        "-ad", "--amara-data", type=str, help="Path to data_urls.json file for amara"
    )
    parser.add_argument("-ds", "--dataset", type=str, help="Dataset csv file")
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        choices=["audio", "cpu", "cuda"],
        type=str,
        help="Device to useg (values: cuda, cpu, auto (default))",
    )
    parser.add_argument(
        "--asr-model", default=None, type=str, help="ASR model"
    )
    parser.add_argument(
        "--summ-model", default=None, type=str, help="Summarization model"
    )
    parser.add_argument("--saved-audio", default=None, type=str, help="Path to the audio files already split into fragments.\
     It should be a pickle object of a pandas.DataFrame with the columns 'id' and 'fragment'")
    parser.add_argument("--saved-summ", default=None, type=str, help="Path to the summaries.\
     It should be a pickle object of a pandas.DataFrame with the columns 'id' and 'summary'. The id is assumed to be unique.")

    parser.add_argument("--mustc", action="store_true", help="Use only the talks from MuST-C")
    args = parser.parse_args()

    audio_tools.MUST_C_PATH = args.must_c
    audio_tools.AMARA_PATH = args.amara
    audio_tools.AMARA_DATA_PATH = args.amara_data
    audio_tools.TED_PATH = args.ted
    audio_tools.TED_DATA_PATH = args.ted_data

    text_tools.MUST_C_PATH = args.must_c
    text_tools.AMARA_PATH = args.amara
    text_tools.AMARA_DATA_PATH = args.amara_data
    text_tools.TED_PATH = args.ted
    text_tools.TED_DATA_PATH = args.ted_data

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" or args.device == "cpu":
        device = args.device
    else:
        raise ValueError(f"Invalid device:{args.device}")

    target_pipeline = tools.get_pipeline(
        text_tools.preprocess_numbers,
        text_tools.preprocess_spaces,
        text_tools.preprocess_clean,
    )
    print("Summarization:", args.summ_model)
    print("ASR:", args.asr_model)
    print("Loading model...")
    model = get_model(device, args.asr_model, args.summ_model)
    print("--done")

    # print("Loading files...")
    if args.saved_summ is not None and args.saved_audio is not None:
        test_from_saved_data(args, target_pipeline, model)
    else:
        test_from_files(args, target_pipeline, model)
