"""
Class for the summarization model and other helper methods

The following global variables are defined.
- Dataset locations:
    - MUST_C_PATH: points to the folder "data" from MuST-C
    - AMARA_PATH: root folder for the TED set
    - AMARA_DATA_PATH: folder where the audio files and transcripts are stored
    - TED_PATH: root folder for the TED set
    - TED_DATA_PATH: folder where the audio files and transcripts are stored
- Enumerators:
    - AMARA_FOLDERS: folders in the Amara set
    - MUSTC_FOLDERS: folders in the MuST-C set
    - DATASETS: set
    - MODELS: summarization models
    - SELECT_TYPE: subdivisions levels of the text, used to select part of the text
- Utilities
    - engine: used to conver numbers to text
"""
from tools import pysrt_time_to_float, pysrt_to_pandas, make_name, Model
import pandas as pd
import collections
import yaml
import glob
import torch
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizerFast,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import inflect
import string
import re
import rouge
import json
import pysrt

# from nltk.tokenize import sent_tokenize, word_tokenize
from unidecode import unidecode

from typing import List, Union, Callable, Iterable, Dict, Any, Optional

MUST_C_PATH: str = ""
AMARA_PATH: str = ""  # path to the amara folder
AMARA_DATA_PATH: str = ""  # path to the folder with the transcripts / audio
TED_PATH: str = ""
TED_DATA_PATH: str = ""
AMARA_FOLDERS: List[str] = ["TED", "TEDx", "TED-ED", "TED-Series", "TED-Translator"]
MUSTC_FOLDERS: List[str] = ["dev", "train", "tst-COMMON", "tst-HE"]

DATASETS = collections.namedtuple("Datasets", ["Amara", "Ted", "Mustc"])(
    "Amara", "Ted", "Mustc"
)
MODELS = collections.namedtuple("Models", ["Pegasus", "T5"])("Pegasus", "T5")
SELECT_TYPE = collections.namedtuple("Select", ["Sentence", "Word", "Character"])(
    "Sentence", "Word", "Character"
)

engine: inflect.engine = inflect.engine()


class SimpleSummarizationModel(Model):
    """
    Wrapper class for a summarization model (Pegasus or T5)
    Pegasus by default.
    Most of the arguments are specific for Pegasus

    Args:
        device: where the model should be stored, "cpu" or "cuda"
        type_: architecture of the model, "Pegasus" or "T5"
        model: path to a saved model
        beams: number of beams for the beam seach (if 1, it is greedy search) (not used)
        return_sequences: return the n most probable sequences
        max_len: maximum length of the predicted sequence
        len_penalty: force the model to produce longer/shoerter sequences (> 1: longer sqeuences, = 1: no penalty, < 1: shorter sequences)
        dropout: dropout probability of the fully connected layers
        do_sample: during the decoding, use sampling instead of greedy search
        temperature: temperature to module the probability of the next token
        top_k: top-k filtering, sample only the k most probable tokens
        top_p: top-p filtering, sample only the p most probable tokens whose probability sum up to p (p < 1)
        max_pos_emb: size of the positional embedding, Pegasus will use only the first max_pos_emb tokens
    """

    pegasus: str = "google/pegasus-xsum"
    t5: str = "t5-base"

    def __init__(
        self,
        device: str,
        type_: str = MODELS.Pegasus,
        model: str = None,
        beams: int = 1,
        return_sequences: int = 1,
        max_len: int = 1000,
        len_penalty: float = 1,
        dropout: float = 0.1,
        do_sample: bool = False,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
        max_pos_emb: int = 512,
    ) -> None:
        super().__init__(device)
        self.beams: int = beams
        self.return_sequences: int = return_sequences
        self.rouge: rouge.rouge.Rouge = rouge.Rouge()

        if type_ == MODELS.Pegasus:
            if model is None:
                model = self.pegasus

            print("Pos emb:", max_pos_emb)
            self.tokenizer: PegasusTokenizerFast = (
                PegasusTokenizerFast.from_pretrained(
                    model, model_max_length=max_pos_emb
                )
            )
            self.tokenizer.model_max_length = max_pos_emb

            self.model: PegasusForConditionalGeneration = PegasusForConditionalGeneration.from_pretrained(
                model,
                dropout=dropout,
                gradient_checkpointing=True,
                # repetition_penalty=10.0,
                # max_length=max_len,
                length_penalty=len_penalty,  # longer: >1
                min_length=5,
                do_sample=do_sample,
                # temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_position_embeddings=max_pos_emb,
                # no_repeat_ngram_size=3,
            ).to(
                self.device
            )
        elif type_ == MODELS.T5:
            if model is None:
                model = self.t5
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model)
            self.model: T5ForConditionalGeneration = (
                T5ForConditionalGeneration.from_pretrained(model).to(self.device)
            )
        self.model.eval()
        print("Vocab size:", self.tokenizer.vocab_size)
        print("Tokenizer max length:", self.tokenizer.model_max_length)
        # print("Model max length:", self.tokenizer.model_max_length)

    def setup_trainer(
        self,
        train: Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset],
        test: torch.utils.data.Dataset,
        epochs: Union[float, int],
        output_dir: str = "./results",
        batch_size: int = 1,
        test_batch_size: int = 1,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        log: str = "./logs",
        resume: Optional[bool] = None,
    ) -> None:
        """
        Fine tune the model. Transformers framework is used.
        The optimizer is Adafactor

        Args:
            train: dataset for training, # TODO specify columns
            test: dataset for testing, # TODO specify columns
            epochs: number epochs for the training # TODO swith to number of steps
            output_dir: where to store the trained model
            batch_size: batch size for the training
            test_batch_size: batch size for testing
            warmup_steps: for Adafactor
            weight_decay: for Adafactor
            log: where to write the logs (not used)
            resume: whether to resume the training. The interrupted model has to be in the `output_dir`

        """
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=test_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=log,
            no_cuda=(self.device != "cuda"),
            logging_steps=100,
            save_steps=300,
            #save_strategy="epoch",
            save_total_limit=60,
            evaluation_strategy="steps",
            eval_steps=300,  # 446,
            # evaluation_strategy="epoch",
            # dataloader_pin_memory=True,
            predict_with_generate=True,
            label_smoothing_factor=0,
            adafactor=True,
            logging_first_step=True,
            learning_rate=5e-5,  # 1e-3 / 5e-5
            #disable_tqdm=True
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            eval_dataset=test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator_str,
            compute_metrics=self.train_metric,
        )
        self.resume = resume

    #def train(self):
    #    # self.trainer.evaluate()
    #    if self.resume:
    #        self.trainer.train(self.resume)
    #    else:
    #        self.trainer.train()

    #def evaluate(self):
    #    self.trainer.evaluate()

    def predict(self, batch: List[str]) -> List[str]:
        """
        Summarize the documents

        Args:
            batch: list of documents as strings
        Returns:
            List of summaries
        """
        tok_batch = self.tokenizer(
            batch, padding="longest", return_tensors="pt", truncation=True,
        ).to(self.device)
        summary = self.model.generate(
            **tok_batch
        )  # , num_beams=self.beams, num_return_sequences=self.return_sequences, max_length=self.max_len, length_penalty=self.penalty)
        return self.tokenizer.batch_decode(summary, skip_special_tokens=True)

    def data_collator_str(self, data: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Process the input documents and make a batch.
        This method is used by a transformers.Trainer

        Args:
            x: List of items from the dataset. Each element must have the keys "doc" (for the document) and "tar" (for the target summary).
                The values have type str.
        Return:
            The batch as a dictionary. The keys are "input_values", "attention_mask", "labels", "decoder_input_ids", the values are torch.Tensor
        """
        inputs = [i["doc"] for i in data]
        targets = [i["tar"] for i in data]

        # process the inputs
        model_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        # process the labels
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                padding=True,
                truncation=True,
                return_tensors="pt",
                pad_to_multiple_of=8,
            ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
            model_inputs["labels"]
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs

    def train_metric(self, pred) -> Dict[str, float]:
        """
        Evaluate the model during the training
        In case the method fails to comput the metricts, 0 is returned for all the metrics.

        This method is used by a transformers.Trainer

        Args:
            pred: predictions of the model (the entire test set)
        Returns:
            Dictionary with the metrics "rouge-1", "rouge-2" and "rouge-l" as float
        """
        target = pred.label_ids
        target[target == -100] = self.tokenizer.pad_token_id
        predict = pred.predictions

        predict_text = self.tokenizer.batch_decode(predict, skip_special_tokens=True)
        target_text = self.tokenizer.batch_decode(target, skip_special_tokens=True)

        for i in range(5):
            print("----------------------")
            print("Target:", target_text[i])
            print("Predicted:", predict_text[i])
            print("----------------------")

        # ignore predictions with only dots
        predict_text, target_text = zip(*[(i,j) for i,j in zip(predict_text, target_text) if not all([k=="." for k in i])])
        #for i in predict_text:
        #    print(i)
        try:
            # ignore_empty doesn't seem to work -> does not ignore dots
            scores = self.rouge.get_scores(predict_text, target_text, avg=True)#, ignore_empty=True)
        except Exception as e:
            print("Fail to compute metrics:", e.args[0])
            scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
        print(scores)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }

    def freeze_encoder(self, value: bool = False) -> None:
        """
        Freeze the encoder, including the input embeddings

        Args:
            value: whether to unfreeze (False: freeze, True: don't freeze)
        Returns:
            None
        """
        enc = self.model.model.encoder
        if enc is None:
            print("Encoder is None: skip freezing")
        else:
            for param in enc.parameters():
                param.requires_grad = value

    def freeze_decoder(self, value: bool = False) -> None:
        """
        Freeze the decoder, including the output embeddings

        Args:
            value: whether to unfreeze (False: freeze, True: don't freeze)
        Returns:
            None
        """
        dec = self.model.model.decoder
        if dec is None:
            print("Decoder is None: skip freezing")
        else:
            for param in dec.parameters():
                param.requires_grad = value

    def freeze_input_embeddings(self, value: bool = False) -> None:
        """
        Freeze only input embeddings

        Args:
            value: whether to unfreeze (False: freeze, True: don't freeze)
        Returns:
            None
        """
        ie = self.model.model.get_input_embeddings()
        if ie is None:
            print("Input embeddings are None: skip freezing")
        else:
            for param in ie.parameters():
                param.requires_grad = value

    def freeze_output_embeddings(self, value: bool = False) -> None:
        """
        Freeze only output embeddings

        Args:
            value: whether to unfreeze (False: freeze, True: don't freeze)
        Returns:
            None
        """
        oe = self.model.model.get_output_embeddings()
        if oe is None:
            print("Output embeddings are None: skip freezing")
        else:
            for param in oe.parameters():
                param.requires_grad = value

    def freeze_not_embeddings(self, value: bool = False) -> None:
        """
        Freeze evereything expect the embeddings (both input and output)

        Args:
            value: whether to unfreeze (False: freeze, True: don't freeze)
        Returns:
            None
        """
        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = value
        # Unfreeze embeddings
        self.freeze_input_embeddings(not value)
        self.freeze_output_embeddings(not value)


class Evaluator:
    """
    Wrapper class for rouge.rouge.Rouge
    Compute the rouge scores

    Args:
        f_score_only: whether to return only the F-score and ignore precision and recall
    """

    def __init__(self, f_score_only: bool = False) -> None:
        if f_score_only:
            self.evaluator = rouge.Rouge(stats=["f"])
        else:
            self.evaluator = rouge.Rouge()

    def evaluate(
        self, target: List[str], hypothesis: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute the rouge score. Number are converted to text (2 -> "two") and the case is normalized.

        Args:
            target: list of reference summaries
            hypothesis: list of summaries generated by the model
        Returns:
            Dictionaty with the metrics.
            e.g. {"rouge-1": {"f": 0.5, "p": 0.5, "r": 0.5}, "rouge-2": ...}
        """
        target = preprocess_case_normalization(target)
        target = preprocess_numbers(target)
        hypothesis = preprocess_case_normalization(hypothesis)
        return self.evaluator.get_scores(
            hypothesis, target, avg=True,# ignore_empty=True
        )


def get_transcript(
    id_: int, set_: str, text_only: bool = True, **kargs
) -> List:
    """
    Get the transcript and the description of a talk given its id and the set

    Args:
        id_: talk id
        set_: dataset name
        text_oly: wheter to return only the text or a DataFrame with the time and the text
        **kargs: content of some files used to retrieve data (speed up)
            For Amara:
                data_[AMARA_FOLDER]: dictionary with the data of the talks, for each Amara folder
                amara_talk_id: dataframe with talk_id.csv
            For TED:
                data_urls: dictionarity with titles, descriptions, video urls and ids of the talks
                ted_talk_id: dataframe with talk_id.csv
            For MUST-C:
                transcript: dataframe of lines
                yaml: dataframe with the content of the yaml file
    Returns:
        Transcript, title and description given the Talk id and the dataset. If text_only it returns a string, otherwise it returns a dataframe with "offset", "duration", "transcript"
    """
    if set_ == DATASETS.Amara:
        return _get_amara_transcript(id_, text_only, **kargs)
    elif set_ == DATASETS.Ted:
        return _get_ted_transcript(id_, text_only, **kargs)
    elif set_ == DATASETS.Mustc:
        return _get_must_c_transcript(id_, text_only, **kargs)
    else:
        raise ValueError("Invalid set")


def _get_amara_transcript(id_, text_only=True, **kargs):
    talk_id = None
    if "amara_talk_id" in kargs:
        tid = kargs["amara_talk_id"][["amara", "id"]]
    else:
        tid = pd.read_csv(f"{AMARA_PATH}/talk_id.csv")[["amara", "id"]]
    tid = tid.set_index("id")

    datas = []
    for i in AMARA_FOLDERS:
        if f"data_{i}" in kargs:
            datas.append(kargs[f"data_{i}"])
        else:
            datas.append(json.load(open(f"{AMARA_DATA_PATH}/{i}/data_urls.json")))

    srt = None
    if id_ in tid.index:
        url = tid.loc[id_]["amara"]
        name = make_name(url, True)  # BUG: THERE NAME IS ...\n.srt !!!
        path = glob.glob(f"{AMARA_DATA_PATH}/*/{name}.srt")
        srt = pysrt.open(path[0])
        title = ""
        descr = ""
        for i in datas:
            if url in i:
                title = i[url][0]
                descr = i[url][1]
                break

    if srt != None:
        if text_only:
            return [srt.text, title, descr]
        else:
            return [pysrt_to_pandas(srt), title, descr]
    else:
        if text_only:
            return ["", "", ""]
        else:
            return [pd.DataFrame(columns=["duration", "offset", "transcript"]), "", ""]


def _get_ted_transcript(id_, text_only=True, **kargs):
    talk_id = None
    if "ted_talk_id" in kargs:
        tid = kargs["ted_talk_id"][["ted", "id"]]
    else:
        tid = pd.read_csv(f"{TED_PATH}/talk_id.csv")[["ted", "id"]]
    data = None
    if "data_urls" in kargs:
        data = kargs["data_urls"]
    else:
        data = json.load(open(f"{TED_DATA_PATH}/data_urls.json"))
    tid = tid.set_index("id")
    srt = None
    if id_ in tid.index:
        url = tid.loc[id_]["ted"]
        name = make_name(
            url
        )  # NO (BUG: THERE NAME IS ...\n.srt !!!) -> audio has \n, not the transcript

        srt = pysrt.open(f"{TED_DATA_PATH}/{name.strip()}.srt")
        #srt = pysrt.open(f"{TED_DATA_PATH}/{name}.srt")
    if srt != None:
        if text_only:
            return [srt.text, data[name][0][0], data[name][0][1]]
        else:
            return [pysrt_to_pandas(srt), data[name][0][0], data[name][0][1]]
    else:
        if text_only:
            return ["", "", ""]
        else:
            return [pd.DataFrame(columns=["duration", "offset", "transcript"]), "", ""]


def _get_must_c_transcript(id_, text_only=True, **kargs):
    folder = None
    if "yaml" not in kargs or "transcript" not in kargs:
        p = glob.glob(f"{MUST_C_PATH}/*/wav/ted_{id_}.wav")
        if len(p) != 0:
            p = p[0]
            folder = p.replace(MUST_C_PATH + "/", "").split("/")[0]
    y = None
    if "yaml" in kargs:
        y = kargs["yaml"]
    else:
        with open(f"{MUST_C_PATH}/{folder}/txt/{folder}.yaml") as f:
            y = pd.DataFrame(yaml.full_load(f))
    t = None
    if "transcript" in kargs:
        t = kargs["transcript"]
    else:
        with open(f"{MUST_C_PATH}/{folder}/txt/{folder}.en") as f:
            t = pd.DataFrame(f.readlines())

    mask_lines = y["wav"].map(lambda x: x == f"ted_{id_}.wav")

    if text_only:
        return ["".join(t[mask_lines][0].to_list()), "", ""]
    else:
        out = pd.DataFrame(
            {"duration": y[mask_lines]["duration"], "offset": y[mask_lines]["offset"]}
        )
        out["transcript"] = t[mask_lines]
        return [out, "", ""]


def preprocess_clean(transcripts: List[str]) -> List[str]:
    """
    Remove parenthesis and special characters

    Args:
        transcripts: list where each item is a documents
    Return:
        List with the cleaned text
    """
    r = re.compile(
        r"\(.*\)|<.*>|<br>|<\br>|\[.*\]|\{.*\}"
    )  # remove parenthesis (Music), (Applauses), etc, ...
    return [unidecode(re.sub(r, "", i)) for i in transcripts]


def preprocess_no_punctuation(transcripts: List[str]) -> List[str]:
    """
    Remove the punctuation from the text.
    Note: ' is kept because it is in the Wav2Vec2 dictionary


    Args:
        transcripts: list where each item is a documents
    Return:
        List with the text without punctuation
    """
    punct = string.punctuation.replace("'", "")
    r = re.compile(f"[{punct}]+")
    return [re.sub(r, " ", i) for i in transcripts]


def preprocess_numbers(transcripts: List[str]) -> List[str]:
    """
    Convert numbers to text (2 -> "two").
    Pegasus can handle numbers, but Wav2Vec2 can't

    Args:
        transcripts: list where each item is a documents
    Return:
        List with the processed text
    """
    num_re = re.compile("\d+[\.[0-9]*]?")

    def job(t: str) -> str:
        search = re.search(num_re, t)
        while search is not None:
            num = engine.number_to_words(t[search.start() : search.end()])
            t = f"{t[:search.start()]}{num}{t[search.end():]}"  # cut and replace
            search = re.search(num_re, t)
        return t

    return [job(i) for i in transcripts]


def preprocess_case_normalization(
    transcripts: List[str], lower: bool = True
) -> List[str]:
    """
    Normalize the case of all the documents

    Args:
        transcripts: list where each item is a documents
        lower: everything lower case instead of upper case
    Return:
        List with the processed text
    """
    if lower:
        return [i.lower() for i in transcripts]
    else:
        return [i.upper() for i in transcripts]


def preprocess_spaces(transcripts: List[str]) -> List[str]:
    """
    Remove multiple white spaces

    Args:
        transcripts: list where each item is a documents
    Return:
        List with the processed text
    """
    r = re.compile("  +")
    return [re.sub(r, " ", i) for i in transcripts]


def preprocess_select(
    transcripts: List[str],
    start: int = 0,
    end: int = 0,
    level: str = SELECT_TYPE.Sentence,
) -> List[str]:
    """
    Select the first "start" and "end" of the selected items

    Args:
        transcripts: documents to preprocess
        start: first n items
        end:: last n items
        level: type of items, SELECT_TYPE.Sentence, SELECT_TYPE.Word or SELECT_TYPE.Character
    Returns:
        The reduced text
    """
    from nltk.tokenize import sent_tokenize, word_tokenize  # improve global performance

    def job(t: str) -> str:
        if level == SELECT_TYPE.Sentence:
            sentences = sent_tokenize(t)
            if len(sentences) <= (start + end):
                return t
            else:
                return "".join(sentences[:start] + sentences[end:])
        elif level == SELECT_TYPE.Word:
            words = word_tokenize(t)
            if len(words) <= (start + end):
                return t
            else:
                return "".join(words[:start] + words[end:])
        elif level == SELECT_TYPE.Character:
            if len(t) <= (start + end):
                return t
            else:
                return f"{t[:start]} {t[end:]}"

    return [job(i) for i in transcripts]


def preprocess_presumm(transcripts, model, group):
    """
    Summarize smaller parts of the text

    Args:
        transcripts: documents to preprocess
        model: model used for thes summarization
        group: how many sentences are summarized every time.
    Returns:
        Reduced text
    """

    def job(t: str) -> str:
        t = sent_tokenize(t)
        out = []
        i = 0
        while (i) * group < len(t):
            out.append(model.predict(["".join(t[i * group : (i + 1) * group])])[0])
            i += 1
        return "".join(out)

    return [job(i) for i in transcripts]


def get_files() -> Dict[str, Any]:
    """
    Read all the files needed to retrieve the audio files and transcripts.

    Keys of the output dictionary:
    - data_{AMARA_FOLDER}: data_urls.json for an Amara folder (TED, TEDx, etc, ...)
    - data_urls: data_urls.json for Ted
    - amara_talk_id: dataframe with id and url for Amara
    - ted_talk_id: dataframe with id and url for Ted
    - yaml: information for the audio fragments from MuST-C
    - transcipt: transcripts for the MuST-C audio fragments

    Returns:
        Dictionary of files used to get the transcripts
    """
    files = {}
    print("-loading Amara")
    # read the amara data_urls
    for i in AMARA_FOLDERS:
        with open(f"{AMARA_DATA_PATH}/{i}/data_urls.json", "r") as f:
            files[f"data_{i}"] = json.load(f)
    print("-loading Ted")
    with open(f"{TED_DATA_PATH}/data_urls.json") as f:
        files["data_urls"] = json.load(f)
    print("-loading ids")
    # read the talk id
    files["amara_talk_id"] = pd.read_csv(f"{AMARA_PATH}/talk_id.csv")
    files["ted_talk_id"] = pd.read_csv(f"{TED_PATH}/talk_id.csv")
    print("-loading yaml")
    # get the yaml file from the two test folders
    with open(f"{MUST_C_PATH}/tst-COMMON/txt/tst-COMMON.yaml", "r") as f1:
        with open(f"{MUST_C_PATH}/tst-HE/txt/tst-HE.yaml", "r") as f2:
            with open(f"{MUST_C_PATH}/train/txt/train.yaml", "r") as f3:
                with open(f"{MUST_C_PATH}/dev/txt/dev.yaml", "r") as f4:
                    files["yaml"] = (
                        pd.DataFrame(yaml.full_load(f1))
                        .append(pd.DataFrame(yaml.full_load(f2)))
                        .append(pd.DataFrame(yaml.full_load(f3)))
                        .append(pd.DataFrame(yaml.full_load(f4)))
                    )
                    # files["yaml"] = pd.DataFrame(yaml.full_load(f1)).append(pd.DataFrame(yaml.full_load(f3))).append(pd.DataFrame(yaml.full_load(f4)))
    print("-loading transcripts")
    with open(f"{MUST_C_PATH}/tst-COMMON/txt/tst-COMMON.en", "r") as f1:
        with open(f"{MUST_C_PATH}/tst-HE/txt/tst-HE.en", "r") as f2:
            with open(f"{MUST_C_PATH}/train/txt/train.en", "r") as f3:
                with open(f"{MUST_C_PATH}/dev/txt/dev.en", "r") as f4:
                    files["transcript"] = (
                        pd.DataFrame(f1.readlines())
                        .append(pd.DataFrame(f2.readlines()))
                        .append(pd.DataFrame(f3.readlines()))
                        .append(pd.DataFrame(f4.readlines()))
                    )
                    # files["transcript"] = pd.DataFrame(f1.readlines()).append(pd.DataFrame(f3.readlines())).append(pd.DataFrame(f4.readlines()))
    return files
