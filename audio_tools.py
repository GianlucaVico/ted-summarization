"""
ASR model and helper methods

The following gloabal variables defines where the different datasets are stored:
- MUST_C_PATH: points to the folder "data" from MuST-C
- AMARA_PATH: root folder for the TED set
- AMARA_DATA_PATH: folder where the audio files and transcripts are stored
- TED_PATH: root folder for the TED set
- TED_DATA_PATH: folder where the audio files and transcripts are stored

"""
import IPython
import torchaudio
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from jiwer import wer, compute_measures
from text_tools import preprocess_case_normalization, preprocess_spaces
import pandas as pd
import yaml
from tools import pysrt_to_pandas, make_name, Model
import text_tools
import pysrt
import numpy as np
import glob
import pydub
import Levenshtein
import random
import decoders
import dataclasses

from typing import List, Union, Callable, Iterable, Dict, Any, Optional

MUST_C_PATH: str = ""
AMARA_PATH: str = " "  # path to the amara folder
AMARA_DATA_PATH: str = ""  # path to the folder with the transcripts / audio
TED_PATH: str = ""
TED_DATA_PATH: str = ""
DATASETS = text_tools.DATASETS


def play(audio: Union[np.array, str, bytes, List[float]], rate: int = 16000) -> None:
    """
    Display a widget to play some audio

    Args:
        audio: numpy array or list with the content of the audio
        rate: sampling rate of the audio
    Returns:
        None
    """
    return IPython.display.Audio(audio, rate=rate)


class SimpleASRModel(Model):
    """
    Wrapper class for Wav2Vec2ForCTC

    Args:
        device: where the model should be stored, "cpu" or "cuda"
        model: load trained model. If None, Wav2Vec2-base-960h is used
        decoder: type of decoder used
        spell: use spell correction
    """

    def __init__(
        self,
        device: str,
        model: Optional[str] = None,
        decoder: str = "greedy",
        spell: bool = False,
        lm: Optional[str] = None
    ) -> None:
        super().__init__(device)
        if model == None:
            model = "facebook/wav2vec2-base-960h"
        #self.tokenizer = Wav2Vec2Processor.from_pretrained(model, do_normalize=True)
        self.tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", do_normalize=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model,
            gradient_checkpointing=True,
            ctc_loss_reduction="mean",
            pad_token_id=self.tokenizer.tokenizer.pad_token_id,
            activation_dropout=0.2,
            ctc_zero_infinity=True,
        ).to(device)
        self.model.eval()

        self.decoder = None
        if decoders == "kenlm_lf":
            args = decoders.DEFAULT_NO_LEXICON
        else:
            args = decoders.DEFAULT

        if lm is not None:
            args = dataclasses.replace(args, **{"kenlm_model": lm})

        if decoder == "greedy":
            self.decoder = decoders.W2V2GreedyDecoder(args, self.tokenizer)
        elif decoder == "viterbi":
            self.decoder = decoders.W2V2ViterbiDecoder(args, self.tokenizer)
        elif decoder == "kenlm":
            self.decoder = decoders.W2V2KenLMDecoder(args, self.tokenizer)
        elif decoder == "kenlm_lf":
            self.decoder = decoders.W2V2KenLMDecoder(
                args, self.tokenizer
            )
        else:
            raise ValueError(
                f"Invalid decoder: {decoder}. Values accepted are 'greedy', 'viterbi', 'kenlm' and 'kenlm_lf'"
            )

        if spell:
            self.decoder = decoders.W2V2NeuSpellDecoder(
                args, self.tokenizer, self.decoder
            )

        print("Pad token:", self.tokenizer.tokenizer.pad_token_id)
        print("Model max input size:", self.tokenizer.tokenizer.max_model_input_sizes)
        print("Vocab:", self.tokenizer.tokenizer.get_vocab())

    def setup_trainer(
        self,
        train: Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset],
        test: torch.utils.data.Dataset,
        epochs: Union[int, float],
        output_dir: str = "./results",
        batch_size: int = 16,
        test_batch_size: int = 64,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        log: str = "./logs",
        freeze_encoder: bool = True,
        resume: Optional[bool] = None,
    ) -> None:
        """
        Fine tune the model. Transformers framework is used.
        The optimizer is AdamW

        Args:
            train: dataset for training, # TODO specify columns
            test: dataset for testing, # TODO specify columns
            epochs: number epochs for the training # TODO swith to number of steps
            output_dir: where to store the trained model
            batch_size: batch size for the training
            test_batch_size: batch size for testing
            warmup_steps: for AdamW
            weight_decay: for AdamW
            log: where to write the logs (not used)
            freeze_encoder: whether to freeze the encoder (Wav2Vec2 without the CTC head)
            resume: whether to resume the training. The interrupted model has to be in the `output_dir`
        """
        if freeze_encoder:
            self.model.freeze_feature_extractor()

        print("Setting trainer....")
        training_args = TrainingArguments(
            output_dir=output_dir,
            # num_train_epochs=epochs,
            max_steps=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=test_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=log,
            no_cuda=(self.device != "cuda"),
            # no_cuda=True,
            # save_strategy="epoch",
            save_steps=1000,
            eval_steps=200,
            evaluation_strategy="steps",
            # evaluation_strategy="epoch",
            logging_steps=200,
            learning_rate=5e-5,  # 5e-5
            save_total_limit=10,
            group_by_length=False,
            # dataloader_pin_memory=False,
            fp16=(self.device == "cuda"),
            # eval_accumulation_steps=2000,
            # ignore_data_skip=True, # for debug
            disable_tqdm=True
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            eval_dataset=test,
            tokenizer=self.tokenizer.feature_extractor,
            data_collator=self.data_collator_str,
            compute_metrics=self.train_metric,
        )
        self.resume = resume
        print("--done")

    def predict(self, audio: Union[torch.Tensor, np.array, List[int]]) -> List[str]:
        """
        Transcribe a batch of audio recordings

        Args:
            audio: data to transcribe. THe shape is (number of recordings, length). The padding is done internally
        Returns:
            List of transcriptions
        """
        input_values = self.tokenizer(
            audio, sampling_rate=16000, return_tensors="pt", padding="longest"
        ).input_values.to(
            self.device
        )  # Batch size 1
        logits = self.model(input_values).logits

        return self.decoder.batch_decode(logits.cpu())

    def data_collator_str(
        self, x: List[Dict[str, Union[np.array, str]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process the audio fragments and make a batch.
        This method is used by a transformers.Trainer

        Args:
            x: List of items from the dataset. Each element must have the keys "input_values" and "labels".
                The "input_values" are numpy.array, while the "labels" are str.
        Return:
            The batch as a dictionary. It has the same keys of the input, but the values are torch.Tensor
        """
        inputs = [i["input_values"] for i in x]
        labels = [i["labels"] for i in x]
        batch = {}
        # Pad the audio fragments
        batch = self.tokenizer(
            inputs,
            padding=True,
            pad_to_multiple_of=32,
            return_tensors="pt",
            sampling_rate=16000,
        )

        # Tokenize and pad the labels
        with self.tokenizer.as_target_processor():
            labels_batch = self.tokenizer(
                labels,
                padding=True,
                pad_to_multiple_of=32,
                return_tensors="pt",
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

    def train_metric(self, pred) -> Dict[str, float]:
        """
        Evaluate the model during the training
        This method is used by a transformers.Trainer

        Args:
            pred: predictions of the model (the entire test set)
        Returns:
            Dictionary with the metrics "wer" and "cer" as float
        """
        # Decode the preditions
        pred_logits = pred.predictions
        # pred_ids = np.argmax(pred_logits, axis=-1)
        # pred_str = self.tokenizer.batch_decode(pred_ids)
        print("Decoding")
        # Decode the labels
        pred.label_ids[pred.label_ids == -100] = self.tokenizer.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        pred_str = self.decoder.batch_decode(pred_logits)

        # HACK This is to split the evaluation in multiple runs
        """pred_str = []
        b_size = 900
        for i in range(len(pred_logits)//b_size + 1):
            start = b_size * i
            end = start + b_size
            print(f"Decoding {i+1}/{len(pred_logits)//b_size + 1}")
            pred_str.extend(self.decoder.batch_decode(pred_logits[start:end]))
            tmp = compute_measures(
                preprocess_spaces(preprocess_case_normalization(label_str[start:end])),
                preprocess_spaces(preprocess_case_normalization(pred_str[start:end])),
            )
            print(f"Hits ({i+1}): {tmp['hits']}")
            print(f"Substitutions ({i+1}): {tmp['substitutions']}")
            print(f"Deletions ({i+1}): {tmp['deletions']}")
            print(f"Insertions ({i+1}): {tmp['insertions']}")
            dist = sum([Levenshtein.distance(i.lower(), j.lower()) for i, j in zip(label_str[start:end], pred_str[start:end])])
            ref_len = sum([len(i) for i in label_str[start:end]])
            print(f"Dist ({i+1}): {dist}")
            print(f"Len ({i+1}): {ref_len}")
        """

        # Print some examples
        for i in range(5):
            print("---------------------")
            print("Prediction:\t", pred_str[i].replace(" ", "|"))
            print("Label:\t\t", label_str[i].replace(" ", "|"))
            print("---------------------")
        for i in range(5):
            print("---------------------")
            print("Prediction:\t", pred_str[len(pred_str) - i - 1].replace(" ", "|"))
            print("Label:\t\t", label_str[len(pred_str) - i - 1].replace(" ", "|"))
            print("---------------------")
        for i in range(5):
            rand_i = random.randint(0, len(pred_str) - 1)
            print("---------------------")
            print(f"Prediction ({rand_i}): ", pred_str[rand_i].replace(" ", "|"))
            print(f"Label ({rand_i}):\t", label_str[rand_i].replace(" ", "|"))
            print("---------------------")
        # Compute and print the metrics
        wer_, cer_ = evaluate(label_str, pred_str)
        #cer_ = cer(label_str, pred_str)
        print("Results:")
        print("WER:", wer_)
        print("CER:", cer_)
        print("\n\n")
        return {"wer": wer_, "cer": cer_}


class Resampler:
    """
    Change the sample rates of audio files to a specific sample rate

    Args:
        sample_rates: list of sample rates of the input audio files
        out_sample_rate: downsample to this sample rate
    """

    def __init__(
        self, sample_rates: List[int], out_sample_rate: int = 16000
    ) -> None:
        self.group = {
            i: torchaudio.transforms.Resample(i, out_sample_rate) for i in sample_rates
        }
        self._out = out_sample_rate

    def resample(self, t: torch.Tensor, s: int) -> torch.Tensor:
        """
        Resample the torch sensor t  with original sample rate s

        Args:
            t: torch tensor
            s: original sample rate
        Return:
            Resampled torch tensor
        """
        r = self.group.get(s)  # resample object
        if r is None:
            r = torchaudio.transforms.Resample(s, self._out)
            self.group[s] = r
        return r(t)


def split_talk(
    audio: Union[str, torch.Tensor],
    split: Union[str, pd.DataFrame],
    rate: int = 16000,
    original_rate: int = 0,
    resampler: Optional[Resampler] = None,
) -> List[torch.Tensor]:
    """
    Split the audio file given a file describing how to split it

    Args:
        audio (str or torch.Tensor): path to the audio file or its content
        split (str or pandas.DataFrame): srt file with the time annotations or the yaml file from Must-c, if it is a dataframe it must have the columms "duration" and "offset"
    Returns:
        list of tensors with the audio fragments
    """

    if isinstance(split, str):
        if split.endswith("yaml"):
            split = pd.DataFrame(yaml.full_load(split))
        elif split.endswith("srt"):
            split = pysrt_to_pandas(pysrt.open(split))
        else:
            print("Invalid path")

    if isinstance(audio, str):
        audio = open_audio([audio], resampler)[0]
    elif resampler is not None:
        audio = resampler.resample(audio, original_rate)

    length = len(audio)
    fragm = [None for i in range(split.shape[0])]

    for i in range(len(fragm)):
        start = min(seconds_to_frame(split.iloc[i]["offset"], rate, np.floor), length)
        end = min(
            start + seconds_to_frame(split.iloc[i]["duration"], rate, np.ceil), length
        )
        fragm[i] = audio[start:end]
    return fragm


def evaluate(target: List[str], hypothesis: List[str]) -> float:
    """
    Evaluate the output of the ASR model.

    Args:
        target: list of target strings
        hypothesis: list of outputs from the model
    Returns:
        Word error rate (in [0, 1])
    """
    target = preprocess_spaces(preprocess_case_normalization(target))
    hypothesis = preprocess_spaces(preprocess_case_normalization(hypothesis))

    return wer(target, hypothesis), cer(target, hypothesis)


def open_audio(
    paths: List[str], resampler: Optional[Resampler] = None, mono: bool = True
) -> List[torch.Tensor]:
    """
    Read a list of audio files

    Args:
        paths: list of paths to the files
        resampler: resampler object used to change the sample rate of the audio. If None, the original sample rate is used
        mono: whether to force the read file to have a single audio channel
    Returns:
        list of tensors with the content of the files
    """
    ret_list = [None for i in paths]
    samples = [0 for i in paths]
    for i, path in enumerate(paths):
        ret_list[i], samples[i] = torchaudio.load(path)

    if mono:
        ret_list = [i.mean(axis=0) for i in ret_list]
    if resampler is not None:
        ret_list = [resampler.resample(i, j) for i, j in zip(ret_list, samples)]
    return ret_list


def seconds_to_frame(
    seconds: float, rate: int, rounding: Callable[[float], float] = np.round
) -> int:
    """
    Find the frame corresponding to the given offset in seconds

    Args:
        seconds (float): offset in seconds from the start
        rate: frame rate
        rounding: function used to round the index of the frame
    Returns:
        index of the frame
    """
    return int(rounding(seconds * rate))


def frame_to_seconds(frame: int, rate: int) -> float:
    """
    Args:
        frame: index of the frame
        rate: sample rate of the audio (16000 Hz for the processed audio)
    Returns:
        Time in seconds given the index of the frame
    """
    return frame / rate


def get_audio_path(id_: int, set_, **kargs) -> str:
    """
    Get the path to an audio file given its id and the set (MuST-C, TED or AMARA) it is into

    Args:
        id_: talk id_
        set_: which dataset
        **kargs: files used to retrieve the talks (speed-up the process)
            amara_talk_id
            ted_talk_id
    Returns:
        path to the audio
    """
    try:
        if set_ == DATASETS.Amara:
            return _get_amara_audio(id_, **kargs)
        elif set_ == DATASETS.Ted:
            return _get_ted_audio(id_, **kargs)
        elif set_ == DATASETS.Mustc:
            return _get_must_c_audio(id_, **kargs)
        else:
            raise ValueError("Invalid set")
    except KeyError:
        print(f"Can't find {id_} in {set_}")
        return ""


def _get_amara_audio(id_, **kargs):
    if "amara_talk_id" in kargs:
        tid = kargs["amara_talk_id"][["amara", "id"]]
    else:
        tid = pd.read_csv(f"{AMARA_PATH}/talk_id.csv")[["amara", "id"]]
    tid = tid.set_index("id")
    url = tid.loc[id_]["amara"]
    name = f"{make_name(url, True)}.wav"
    paths = glob.glob(f"{AMARA_DATA_PATH}/*/{name}")
    return _select_path(id_, paths)


def _get_ted_audio(id_, **kargs):
    """
    Args:
        **kargs: ted_talk_id file
    Returns:
        path
    """
    tid = None
    if "ted_talk_id" in kargs:
        tid = kargs["ted_talk_id"][["ted", "id"]]
    else:
        tid = pd.read_csv(f"{TED_PATH}/talk_id.csv")[["ted", "id"]]
    tid = tid.set_index("id")
    url = tid.loc[id_]["ted"]
    name = f"{make_name(url)}.wav"
    paths = glob.glob(f"{TED_DATA_PATH}/{name}")
    return _select_path(id_, paths)


def _get_must_c_audio(id_, **kargs):
    paths = glob.glob(f"{MUST_C_PATH}/*/wav/ted_{id_}.wav")
    return _select_path(id_, paths)


def _select_path(id_, paths):
    if len(paths) == 1:
        return paths[0]
    elif len(paths) > 1:
        print("Ambiguous id:", id_)
        return paths[0]
    else:
        # print("Missing id:", id_)
        return ""


def get_fragments(
    audio: str, max_len: int = 800000, batch_size: int = 1, min_size: int = 500
) -> torch.Tensor:
    """
    Args:
        audio: path to the audio file
        max_len: maximum length of the fragments, if larger they are split in half
    Returns:
        list of tensors with the fragments (batch of size 1)
    """
    # read
    try:
        a = pydub.AudioSegment.from_file(audio).set_frame_rate(16000).set_channels(1)
    except FileNotFoundError:
        print("Can't find audio:", audio)
        return []

    # split
    #print("Splitting...")
    #print(a.dBFS)
    split = pydub.silence.split_on_silence(a, silence_thresh=-27)
    #print("--done")
    # count large fragments
    extra = 0
    for i in split:
        extra += i.frame_count() // max_len

    # split large fragments
    if extra != 0:
        tmp = [None for i in range(len(split) + int(extra))]
        j = 0  # position in tmp
        for i in split:
            if i.frame_count() < max_len:
                tmp[j] = i
                j += 1
            else:
                l = i.frame_count()
    #            # where to split (e.g. max_len is 50, split 120 -> 0, 40, 80, 120, 3 fragments)
                cuts = np.linspace(0, l, int(l // max_len + 2))
                for c in range(1, len(cuts)):
                    tmp[j] = i[c - 1 : c]
                    j += 1
        split = tmp

    # to tensor
    # split = [torch.tensor(i.get_array_of_samples()) for i in split]
    for i, s in enumerate(split):
        tmp = torch.tensor(s.get_array_of_samples())
        if len(tmp) < min_size:
            tmp = F.pad(tmp, (0, min_size - len(tmp)))
        split[i] = tmp
    #print("Split in", len(split))
    #print(split[0] is None)
    # make batches
    # if batch_size != 1:
    #     empty = 0  # empty fragments to append to make the batch
    #     if len(split) % batch_size != 0:
    #         empty = batch_size - (len(split) % batch_size)
    #     split.extend([torch.zeros((min_size,)) for i in range(empty)])
    #
    #     batches = [None for i in range(int(len(split) / batch_size))]
    #     for b, _ in enumerate(batches):
    #         batch = split[b : b + batch_size]
    #         lens = [len(i) for i in batch]
    #         l = max(lens)
    #         batch = torch.vstack([F.pad(i, (0, l - len(i))) for i in batch])
    #         batches[b] = batch
    if len(split) == 0 or split[0] is None:
        print("Can't get audio: empty")
    return split


def cer(
    ref: Union[List[str], str],
    hyp: Union[List[str], str],
    ignore_space: bool = False,
    keep_list: bool = False,
) -> Union[List[float], float]:
    """
    Compute the character error rate between two strings.

    Note: this method automaticatly apply case normalization

    Args:
        ref: str or list of reference strings
        hyp: str or list of hypothesis
        ignore_space: whether to keep or ignore the spaces
        keep_list: return the total CER or a list of CERs
    Returns:
        Total CER or list of CER for each sample
    """
    assert type(ref) == type(
        hyp
    ), "The reference and the hypothesis must be of the same type"
    assert type(ref) == str or (
        type(ref) == list and len(ref) == len(hyp)
    ), "The reference and the hypothesis must be lists with the same length or strings"

    if type(ref) == str:
        ref = [ref.strip()]
        hyp = [hyp.strip()]

    if ignore_space:
        ref = [i.replace(" ", "").strip() for i in ref]
        hyp = [i.replace(" ", "").strip() for i in hyp]

    dist = [Levenshtein.distance(i.lower(), j.lower()) for i, j in zip(ref, hyp)]
    ref_len = [len(i) for i in ref]

    if keep_list:
        return [i / j for i, j in zip(dist, ref_len)]
    else:
        return sum(dist) / sum(ref_len)
