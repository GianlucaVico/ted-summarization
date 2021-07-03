print("Importing torch...")
import torch

lock = torch.tensor([1]).to("cuda")  # lock the gpu
import sys

sys.path.append("../")

import pickle
import joblib
import glob
import argparse
import time
import datetime
import jiwer
import os

print("Importing numpy...")
import pandas as pd
import numpy as np

print("Importing audio tools...")
import audio_tools

print("Importing tools...")
import tools

print("Importing text tools...")
import text_tools

from torch.utils.data.dataset import Dataset, IterableDataset
import torch.nn.functional as F

print("Importing transformers...")
from transformers import AdamW  # , Adafactor

print("Packages imported")
# torch.backends.cudnn.benchmark = True
###############################################################################
class PresavedSet(Dataset):
    def __init__(self, path, pipeline, tokenizer, device, max_len=400000):
        """
        Args:
            path: path to the compressed files
            pipeline: used to process the text
            tokenizer: wav2vec2 preprocessor

        Returns:
            None.

        """
        flatten = lambda x: [j for i in x for j in i]
        no_none = lambda x: [i for i in x if i is not None]

        self.max_len = max_len
        self.tokenizer = tokenizer
        parts = glob.glob(f"{path}/audio_*")
        if len(parts) == 0:
            print("Empty dataset")

        # self.ids = [None for i in parts]
        self.audio = [None for i in parts]
        self.transcripts = [None for i in parts]
        for i, p in enumerate(parts):
            print(f"Loading part {i+1}/{len(parts)}")
            # self.ids[i], self.audio[i], self.transcripts[i] = joblib.load(p)
            _, self.audio[i], self.transcripts[i] = joblib.load(p)

        # flatten
        # self.ids = flatten(no_none(self.ids))
        self.audio = flatten(no_none(self.audio))#[:10]  # [4895*16:4950*16]
        self.transcripts = flatten(no_none(self.transcripts))#[:10]
        self.transcripts = pipeline(self.transcripts)
        # self.transcripts = pipeline(flatten(no_none(self.transcripts)))#[4895*16:4950*16]

        self.audio, self.transcripts = zip(
            *[
                (i, j)
                for i, j in zip(self.audio, self.transcripts)
                if len(j.strip()) >= 5 and len(i) < self.max_len
            ]
        )

        #self.audio = self.audio[:160]
        #self.transcripts = self.transcripts[:160]
        self.device = device

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        item = {}
        a = self.audio[idx]
        # print(idx, len(a))
        item["input_values"] = np.pad(a, (0, max(0, 1000 - len(a))))[: self.max_len]
        item["labels"] = self.transcripts[idx].strip().upper()
        return item


class AsPresavedDataset(Dataset):
    """
    Use an IterDataset as a PresavedDataset

    __getitem__ return the next item independently of the index
    """

    def __init__(
        self, path, pipeline, tokenizer, device, len_, batch_size=1, max_len=320000
    ):
        self.path = path
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.device = device
        self.len_ = len_
        self.batch_size = batch_size
        self.max_len = max_len
        self.set = self.renew()

    def renew(self):
        return IterDataset(
            self.path,
            self.pipeline,
            self.tokenizer,
            self.device,
            self.len_,
            self.batch_size,
            self.max_len,
        )

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        try:
            item = next(self.set)
        except StopIteration:
            self.set = self.renew()
            item = self[idx]
        return item


class IterDataset(IterableDataset):
    # TODO remove batch_size
    def __init__(
        self, path, pipeline, tokenizer, device, len_, batch_size=1, max_len=400000
    ):
        """
        Args:
            path: path to the compressed files
            pipeline: used to process the text
            tokenizer: wav2vec2 preprocessor

        Returns:
            None.

        """
        self.tokenizer = tokenizer
        self.parts = glob.glob(f"{path}/audio_*")
        self.p = 0  # part counter
        self.i = -batch_size  # item counter
        self.pipeline = pipeline
        self._len = len_
        self.max_len = max_len
        if len(self.parts) == 0:
            print("Empty dataset")
            # self.ids = []
            self.audio = []
            self.transcripts = []
        else:
            _, self.audio, self.transcripts = joblib.load(self.parts[self.p])
            self.transcripts = self.pipeline(self.transcripts)
            print("Part size:", len(self.audio))
        # self.audio = self.audio[:10]
        # self.transcripts = self.transcripts[:10]
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        return self

    def __len__(self):
        # return 1830732
        # return 10
        return self._len

    def __next__(self):
        # update counters and cache
        self.i += self.batch_size
        if self.i >= len(self.audio):
            self.i = 0
            self.p += 1
            if self.p >= len(self.parts):
                raise StopIteration  # no items left
            else:
                print(f"Part: {self.p+1}/{len(self.parts)}")
                # load next file
                _, self.audio, self.transcripts = joblib.load(self.parts[self.p])
                self.transcripts = self.pipeline(self.transcripts)
                print("Part size:", len(self.audio))

        # if len(self.transcripts[self.i].strip()) <= 5:
        #    return self.__next__()

        # make batch
        item = {}
        a = self.audio[self.i]
        # print(len(a))
        if len(a) > self.max_len or len(self.transcripts[self.i].strip()) <= 5:
            # print("Skip", self.p, self.i, ":", len(a))
            # print(self.transcripts[self.i])
            return next(self)
        item["input_values"] = np.pad(a, (0, max(0, 1000 - len(a))))[: self.max_len]
        item["labels"] = self.transcripts[self.i].strip().upper()
        return item

    """def _make_batch(self, x):
        l = len(x)
        x_len = [0 for i in range(l)]
        x_ = [None for i in range(l)]

        for i in range(l):
            x_len[i] = x[i].shape[1]
        x_pad = max(x_len)

        for i in range(l):
            x_[i] = torch.nn.functional.pad(x[i], (0, x_pad - x[i].shape[1]))

        return torch.cat(x_)"""


###############################################################################
def do_test(model, pipeline, dataset, delta, files):
    # test = dataset[(dataset["test"] == True) & (dataset["drop"]==False)]
    # train = dataset[(dataset["train"] == True) & (dataset["drop"] == False)]

    def make_batch(data, batch_n):
        fragms = []
        targets = []
        ids = []
        data = data.iloc[batch_n : (batch_n + 1)]
        for i in range(data.shape[0]):
            id_ = data.iloc[i].name
            set_ = None
            if data.iloc[i]["must_c"]:
                set_ = text_tools.DATASETS.Mustc
            elif data.iloc[i]["ted"]:
                set_ = text_tools.DATASETS.Ted
            # elif test.iloc[i]["amara"]:
            #    set_ = text_tools.DATASETS.Amara
            else:
                print("CANNOT FIND THE SET:", id_)
                # pass
            if set_ is not None:
                path = audio_tools.get_audio_path(id_, set_, **files)
                if path == "" and id_ > 0:
                    print(f"Missing {id_} from {set_}")
                elif id_ > 0:
                    transcript, _, _ = text_tools.get_transcript(
                        id_, set_, False, **files
                    )
                    if set_ != text_tools.DATASETS.Mustc:
                        transcript["offset"] = transcript["offset"] - delta
                        transcript["duration"] = transcript["duration"] + 2 * delta
                    audio = audio_tools.split_talk(path, transcript, resampler=None)
                    # print(transcript.head())
                    transcript = transcript["transcript"].tolist()

                    fragms.extend(audio)
                    targets.extend(transcript)
                    ids.extend([id_ for j in transcript])
        # print(len(fragms))
        return fragms, targets, ids

    print("Starting testing...")
    transcripts = []
    results = pd.DataFrame(columns=["ID", "WER", "transcript", "target"])

    # batch size == 1
    batches = len(dataset)

    for b in range(batches):
        print(f"Batch {b+1}/{batches}")
        fragms, targets, ids = make_batch(dataset, b)
        targets = pipeline(targets)
        transcripts = []
        # print("Batch", b)
        for i, f in enumerate(fragms):
            if len(f) >= 1000000:
                print(len(f))
                f = f[:1000000]
            if targets == "" or len(f) >= 1000000:
                transcripts.append("")
            else:
                if len(f) < 1000:  # empty fragments (smaller -> wav2vec fails)
                    f = F.pad(f, (0, 1000 - len(f)))
                transcripts.extend(model.predict(f))
        # print(ids)
        #print(targets)
        #print(transcripts)
        errs = [
            None if t.strip() == "" else audio_tools.evaluate([t], [h])[0] # ignore empty, only wer
            for i, (t, h) in enumerate(zip(targets, transcripts))
        ]
        results = results.append(
            pd.DataFrame(
                {"ID": ids, "WER": errs, "transcript": transcripts, "target": targets}
            )
        )
    # This is not correct -> different length of the transcripts
    wer = results["WER"].dropna().mean()

    print("--done")
    return wer, results


def do_transcribe(
    model,
    pipeline,
    output,
    train_data,
    test_data,
    dataset,
    delta,
    files,
    test_=False,
    output_test=".",
):
    test_wer, test_res = do_test(model, pipeline, test_data, delta, files)
    train_wer, train_res = do_test(model, pipeline, train_data, delta, files)
    print("Train WER:", train_wer)
    print("Test WER:", test_wer)

    if test_:
        test_res.to_csv(f"{output_test}/test_wer.csv")
        train_res.to_csv(f"{output_test}/train_wer.csv")

    test_doc = test_res.groupby("ID")["transcript"].agg(" \n ".join)
    train_doc = train_res.groupby("ID")["transcript"].agg(" \n ".join)
    mask_test = [True for i in range(test_doc.shape[0])]
    mask_train = [True for i in range(train_doc.shape[0])]
    test_targets = []  # summaries
    train_targets = []  # summaries

    for i in range(test_doc.shape[0]):
        id_ = test_doc.index[i]
        set_ = None
        if dataset.iloc[i]["must_c"]:
            set_ = text_tools.DATASETS.Mustc
        elif dataset.iloc[i]["ted"]:
            set_ = text_tools.DATASETS.Ted
        if set_ is None:
            mask_test[i] = False
        else:
            _, _, descr = text_tools.get_transcript(id_, set_, True, **files)
            test_targets.append(descr)

    for i in range(train_doc.shape[0]):
        id_ = train_doc.index[i]
        set_ = None
        if dataset.iloc[i]["must_c"]:
            set_ = text_tools.DATASETS.Mustc
        elif dataset.iloc[i]["ted"]:
            set_ = text_tools.DATASETS.Ted
        if set_ is None:
            mask_train[i] = False
        else:
            _, _, descr = text_tools.get_transcript(id_, set_, True, **files)
            train_targets.append(descr)
    #print(test_doc[mask_test].tolist())
    #print(test_targets)
    with open(f"{output}/test_transcript_documents.pkl", "wb") as f:
        pickle.dump(test_doc[mask_test].tolist(), f)
    with open(f"{output}/test_transcript_targets.pkl", "wb") as f:
        pickle.dump(test_targets, f)
    with open(f"{output}/train_transcript_documents.pkl", "wb") as f:
        pickle.dump(train_doc[mask_train].tolist(), f)
    with open(f"{output}/train_transcript_targets.pkl", "wb") as f:
        pickle.dump(train_targets, f)


###############################################################################
def main(
    epochs,
    batch,
    test_batch,
    log,
    output,
    warmup,
    weight_decay,
    dataset,
    device,
    test,
    transcribe,
    evaluate,
    output_test,
    delta,
    resume,
    freeze_encoder,
    debug,
    saved_train,
    saved_test,
    length,
    model,
    decoder,
    spell,
    lm,
):
    print("Debug:", debug)
    print("Device:", device)
    files = {}
    if test or transcribe:
        files = text_tools.get_files()
    # Load model
    print("Loading model...")
    model = audio_tools.SimpleASRModel(
        device=device, model=model, decoder=decoder, spell=spell, lm=lm
    )
    pipeline = tools.get_pipeline(
        text_tools.preprocess_numbers,
        text_tools.preprocess_clean,
        text_tools.preprocess_no_punctuation,
        lambda x: text_tools.preprocess_case_normalization(x, False),
        text_tools.preprocess_spaces,
    )
    # resampler = audio_tools.Resampler([44100])
    print("--done")

    # load data
    print(f"Loading data... {dataset}")
    dataset = pd.read_csv(dataset, index_col="id")
    dataset = dataset[dataset.index >= 0]
    if debug:
        dataset = dataset.iloc[:10]
        epochs = 2
    train_data = []
    test_data = []
    if test or transcribe:
        test_data = dataset[(dataset["test"] == True) & (dataset["drop"] == False)]
        train_data = dataset[(dataset["train"] == True) & (dataset["drop"] == False)]
    else:
        test_data = PresavedSet(saved_test, pipeline, model.tokenizer, device)
        # train_data = PresavedSet(saved_train, pipeline, model.tokenizer, device)
        if evaluate:
            train_data = []
        else:
            train_data = IterDataset(
                saved_train, pipeline, model.tokenizer, device, length, 1
            )
        # train_data = AsPresavedDataset(saved_train, pipeline, model.tokenizer, device, length, 1)

        # test_data = IterDataset(saved_test, pipeline, model.tokenizer, device, 233700, batch)
        # train_data = IterDataset(saved_train, pipeline, model.tokenizer, device, 545320, 1) -> 128162 for mustc

    print("Train data:", len(train_data))
    print("Test data:", len(test_data))
    print("--done")

    # run
    if transcribe:
        do_transcribe(
            model,
            pipeline,
            output,
            train_data,
            test_data,
            dataset,
            delta,
            files,
            test_=test,
            output_test=output_test,
        )
    elif test:
        wer, res = do_test(model, pipeline, test_data, delta, files)
        print("WER:", wer)
        res.to_csv(f"{output_test}/wer.csv")
    else:
        model.setup_trainer(
            train_data,
            test_data,
            epochs,
            output,
            batch,
            test_batch,
            warmup,
            weight_decay,
            log,
            freeze_encoder,
            resume,
        )
        if evaluate:
            print("Evaluating...")
            model.evaluate()
        else:
            print("Training...")
            model.train()


###############################################################################
if __name__ == "__main__":
    # Add evaluation, language model
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(description="Train Wav2Vec2")

    group = parser.add_argument_group("Model")
    group.add_argument(
        "-d",
        "--device",
        default="auto",
        choices=["audio", "cpu", "cuda"],
        type=str,
        help="Device to useg (values: cuda, cpu, auto (default))",
    )
    group.add_argument(
        "-m", "--model", default=None, type=str, help="Trained model to fine-tune"
    )
    group.add_argument(
        "--no-freeze", action="store_false", help="Don't free the encoder"
    )
    group.add_argument(
        "--decoder",
        default="greedy",
        choices=["greedy", "kenlm", "viterbi", "kenlm_lf"],
        help="Decoding methods. Values are 'greedy', 'kenlm', 'viterbi' and 'kenlm_lf'.\
         For 'kenlm' and 'kenlm_lf' both the lm and the lexicon are hard coded. \
         'kenlm_lf' is lexicon free.",
    )
    group.add_argument(
        "-lm", "--lm", default=None, type=str, help="Path to a kenlm language model."
    )
    group.add_argument(
        "--spell", action="store_true", help="Use spell correction after the decoding"
    )

    group = parser.add_argument_group("Trainer")
    group.add_argument(
        "-e", "--epochs", default=2000, type=int, help="Number of training epochs"
    )
    group.add_argument(
        "-b", "--batch", default=16, type=int, help="Batch size during the training"
    )
    group.add_argument(
        "-tb",
        "--test-batch",
        default=64,
        type=int,
        help="Batch size during the testing",
    )
    group.add_argument(
        "-o",
        "--output",
        default="./results",
        type=str,
        help="Folder to store the trained models",
    )
    group.add_argument(
        "-l", "--log", default="./logs", type=str, help="Folder for logs"
    )
    group.add_argument("-w", "--warmup", default=100, type=int, help="")
    group.add_argument(
        "-wd",
        "--weight",
        default=0.01,
        type=float,
        help="Weight decay for AdamW optimizer",
    )
    group.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )

    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "-t",
        "--ted",
        type=str,
        help="Path to the folder with the transcripts and audio from ted",
    )
    group.add_argument(
        "-a",
        "--amara",
        type=str,
        help="Path to the folder with the transcripts and audio from amara",
    )
    group.add_argument(
        "-mc",
        "--must-c",
        type=str,
        help="Path to the folder with the transcripts and audio from MuSt-C",
    )
    group.add_argument(
        "-td", "--ted-data", type=str, help="Path to data_urls.json file for ted"
    )
    group.add_argument(
        "-ad", "--amara-data", type=str, help="Path to data_urls.json file for amara"
    )
    group.add_argument("-ds", "--dataset", type=str, help="Dataset csv file")
    group.add_argument(
        "-f",
        "--fragment",
        type=float,
        default=0,
        help="Make the fragments larger (in seconds) -- doesn't work with presaved dataset (pkl files)",
    )
    group.add_argument("--saved-train", type=str, help="Folder with train data")
    group.add_argument("--saved-test", type=str, help="Folder with test data")
    group.add_argument("--length", type=int, help="Size of the training set")

    group = parser.add_argument_group("Job")
    group.add_argument(
        "--test",
        action="store_true",
        help="Test the model on the test set and save the WER for each talk",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Test the model on the test set. Use the transformers framework and the presave sets.\
         Print WER and CER on the test set",
    )
    group.add_argument(
        "--transcribe",
        action="store_true",
        help="Transcribe both the train an the test set (ignored if --test)",
    )
    group.add_argument(
        "-to",
        "--test-output",
        default="results.csv",
        type=str,
        help="Where to store the test results",
    )

    group = parser.add_argument_group("Others")
    group.add_argument("--debug", action="store_true", help="Test run of the script")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" or args.device == "cpu":
        device = args.device
    else:
        raise ValueError(f"Invalid device:{args.device}")
    print("--done")
    print("Setting paths")
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
    print(args)
    print("Starting...")
    main(
        epochs=args.epochs,
        batch=args.batch,
        test_batch=args.test_batch,
        log=args.log,
        output=args.output,
        warmup=args.warmup,
        weight_decay=args.weight,
        dataset=args.dataset,
        device=device,
        test=args.test,
        transcribe=args.transcribe,
        evaluate=args.evaluate,
        output_test=args.test_output,
        delta=args.fragment,
        resume=args.resume,
        freeze_encoder=args.no_freeze,
        debug=args.debug,
        saved_train=args.saved_train,
        saved_test=args.saved_test,
        length=args.length,
        model=args.model,
        decoder=args.decoder,
        spell=args.spell,
        lm=args.lm
    )
    print("--done")
