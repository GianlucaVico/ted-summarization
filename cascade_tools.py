"""
Class for the cascade model
"""
import tools
import audio_tools
import text_tools
import torch
from tools import progress_bar
from typing import Union, List, Dict

class SimpleCascadeModel:
    def __init__(self, asr, summ):
        """
        Cascade model for audio summarization

        Args:
            asr: pre-trainied asr model
            summ: pre-trained summarization model
        Returns:
            Cascade model
        """
        self.asr = asr
        self.summ = summ

        self.eval()

    def predict(self, data: Union[str, List[torch.Tensor]]) -> str:
        """
        Summarize an audio files.

        Note: it is not possible to set the parameters to split the audio if
            the talks are passed as strings.

        Args:
            data: path to the audio file or list of torch tensors
        Returns:
            Summarized audio
        """
        if isinstance(data, str):
            data = audio_tools.get_fragments(data)

        transcripts = ["" for i in range(len(data))]
        #print(f"Adding {len(data)} fragments")
        for i, d in enumerate(data):
            # FIX handle large fragments
            transcripts[i] = self.asr.predict(d[:800000])[0]
            #print(transcripts[i])
        doc = " ".join(transcripts)
        #print(doc)
        # wav2vec 2.0 return upper case text <-> pegasus is trained on lower case
        # bad initial design choices
        return self.summ.predict([doc.lower()])[0]

    def evaluate(self, x: List[Union[List[torch.Tensor], str]], y: List[str], verbose: bool=True) -> Dict[str,Dict[str, float]]:
        """
        Evaluate the model with the ROUGE score

        Args:
            x: list of speech. Each item is passet to SimpleASRModel.predic.
                So it should be a list paths to the audio files or a list of list of tensors.
            y: list of reference summaries
            verbose: whether to print the progress bar
        Returns:
            Dictionary with the metrics
        """
        assert len(x) == len(y), "The number of samples and of labels must be the same"
        #x = x[:5]
        #y = y[:5]
        if verbose:
            docs = ["" for i in x]
            #progress_bar(0, "Evaluating:", len(docs))
            for n, i in enumerate(x):
                docs[n] = self.predict(i)
                progress_bar(n+1, "Evaluating:", len(docs))
                #if n+1 % 10 == 0:
                #    print(f"Decoding {n+1}/{len(x)}")
        else:
            docs = [self.predict(i) for i in x]

        for i in range(5):
            print("--------------------------------------")
            print("Target:", y[i])
            print("Predicitons:", docs[i])
            print("--------------------------------------")
        ev = text_tools.Evaluator(True)
        docs, y = zip(*[(i,j) for i,j in zip(docs, y) if not all([k=="." for k in i])])
        return ev.evaluate(y, docs)

    def train(self):
        """
        Set the model to train mode
        """
        self.asr.model.train()
        self.summ.model.train()

    def eval(self):
        """
        Set the model to evaluation mode (default)
        """
        self.asr.model.eval()
        self.summ.model.eval()
