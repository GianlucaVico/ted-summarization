"""
Generic helper methods
"""

import os
import re
import pandas as pd
import pysrt
from typing import List, Callable, Any


def progress_bar(
    value: int, text: str, tot: int = 100, done_char: str = "|", bar_char="-", width=50
) -> None:
    """
    Display progress bar

    My prigress bar: [|||---------] 30%

    Args:
        value: fraction done
        text: text to print at the beginning of the bar
        tot: number of task to comple the bar
        done_char: character used for the filled bar
        bar_char: character used for the empty bar
        width: wdth in character of the bar
    """
    p = ("{:2.2f}").format(100 * (value / tot))  # percentage done
    filled = int(width * value // tot)  # chars of the filled part of the bar
    bar = "|" * filled + "-" * (width - filled)  # bar to display
    print(
        f"\r{text}: [{bar}] {p}% - {value}/{tot}", end="", flush=True
    )  # text: [bar] percentage, no new line, go back at the begginig
    if value >= tot:  # end
        print()

class Model:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.resume = False
        self.trainer = None

    def evaluate(self):
        self.trainer.evaluate()

    def train(self):
        #self.trainer.evaluate()
        print("Start training...")
        if self.resume:
            self.trainer.train(self.resume)
        else:
            self.trainer.train()
        print("--done")

    def setup_trainer(self, *args):
        raise NotImplementedError("The trainers must be setup by the child\
        class. If it is not implemented, the training might not be suppoerted\
        by the model.")

    def train_metric(self, pred):
        raise NotImplementedError()

    def predict(self, batch: List[str]) -> List[str]:
        raise NotImplementedError()

def make_name(url: str, remove_colon: bool = False) -> str:
    """
    Make the file name from the url

    Args:
        url: original url of the file
        remove_colon: whether to remove ":" from the url
    Returns:
        Name of the file

    Note: there could \n at the end of the name"
    """
    if remove_colon:
        url = url.replace(":", "_")
    return url.replace("/", "__")


def name_to_url(name: str, add_colon: bool = True) -> str:
    """
    Undo make_name: try to retrieve the url from the name

    Args:
        name: name of the file
        add_colon: whether to add ":" (for example in "https:...")
    Returns:
        Original url of the file
    """
    if add_colon:
        name = name.replace("_", ":", 1)
    return name.replace("__", "/")


def file_exists(file: str) -> bool:
    """
    Check if file exists

    Args:
        file: path to the file
    Returns:
        Whether the file exists or not
    """
    return os.path.isfile(file)


remove_re = [
    re.compile(r"\(.*\)"),  # remove parenthesis (Music), (Applauses), etc, ...
    re.compile("<.*>"),  # remove html tags
    re.compile("[^A-Za-z0-9 ]+"),  # non-english and special characters #?èé
]


def drop_transcript(text: str, min_words: int = 50) -> bool:
    """
    Whether document has to be discarded based of the number of words

    Args:
        text: text to check
        min_words: minimum number of words to keep the text
    Return:
        Whether to discard the text
    """
    for i in remove_re:
        text = re.sub(i, "", text)
    text = text.strip().split(" ")
    return len(text) < min_words


def pysrt_time_to_float(srt: pysrt.srttime.SubRipTime) -> float:
    """
    Convert pysrt time to seconds

    Args:
        srt: pysrt time to convert
    Returns:
        Amount of seconds
    """
    return srt.hours * 3600 + srt.minutes * 60 + srt.seconds + srt.milliseconds / 1000


def pysrt_to_pandas(srt: pysrt.srtfile.SubRipFile) -> pd.DataFrame:
    """
    Convert the pysrt transcript to a pandas DataFrame

    Args:
        srt: the file opened with pysrt
    Returns:
        A DataFrame with the content of the file.
        It has three columns:
        - offset: when the fragment start (in seconds)
        - duration: length of the fragment (in seconds)
        - transcript: transcript of the fragment (str)
    """
    df = pd.DataFrame(columns=["offset", "duration", "transcript"])
    for i in srt:
        start = pysrt_time_to_float(i.start)
        df = df.append(
            {
                "offset": start,
                "duration": pysrt_time_to_float(i.duration),
                "transcript": i.text,
            },
            True,
        )
    return df


def get_pipeline(*steps: List[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """
    Process a list of strings

    Args:
        *steps: functions to apply.
            The functions must take a single argument.
            The result of a function is passed to the next one
    Returns:
        A function that takes one argument and applies the given list of functions
    """

    def pipeline(data: List[str]) -> List[str]:
        for i in steps:
            data = i(data)
        return data

    return pipeline
