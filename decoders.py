"""
Decoders for Wav2Vec2

Decoder available:
- Greedy
- Viterbi
- KenLM with flashlight
- KenLM with CTCDecode from DeepSpeech

Post-decoding:
- NeuSpell

Default settings:
- DEFAULT
- DEFAULT_NO_LEXICON

Note: flashlight is needed only for W2V2KenLMDecoder_flashlight and W2V2ViterbiDecoder
    If they are not used it is not neccessary to install flashlight. Comment the import statements for flashlight
"""

import itertools as it
import math
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np
# compile kenlm with default gcc compiler and cmake -DKENLM_MAX_ORDER=25 ..
# For flashlight, use gcc/9 intelmkl/200 and the correct python, cuda and cudnn modulues
# export KENLM_ROOT=$HOME/kenlm
# python3 setup.py clean && USE_CUDA=1 USE_KENLM=1 USE_MKL=1 python3 setup.py install --user
from flashlight.lib.text.decoder import LexiconDecoder, LexiconDecoderOptions
from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions
from flashlight.lib.text.decoder import KenLM, CriterionType, Trie, SmearingMode
from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from flashlight.lib.text.dictionary import create_word_dict, load_words

import neuspell
from neuspell import available_checkers, BertChecker

from typing import List, Union, Type, Iterable
from tools import progress_bar
import re

import ctcdecode
import numpy as np

def to_torch(t: Iterable) -> torch.Tensor:
    """
    Ensure that the tensors are of type torch.Tensor

    Args:
        t: tensor
    Returns:
        Torch tensor
    """
    if isinstance(t, Iterable) and not isinstance(t, torch.Tensor):
        return torch.tensor(t)
    return t

@dataclass
class W2V2DecoderArgs:
    """
    Adapted https://github.com/huggingface/transformers/blob/9ee266fdf2138cda013a8fe9da434fa497aebfda/examples/research_projects/wav2vec2/run_wav2vec2_eval_with_lm.py
    By deepang17

    Arguments for W2V2Decoder
    """

    lexicon: Optional[str] = field(
        default=None, metadata={"help": "Specify the path of the lexicon file."}
    )
    lm_weight: Optional[float] = field(
        default=0.2,
        metadata={"help": "Weight for lm while interpolating with neural score."},
    )
    unit_lm: Optional[bool] = field(
        default=True, metadata={"help": "Whether using unit lm or not."}
    )
    beam: Optional[int] = field(
        default=200, metadata={"help": "Specify the size of the beam."}
    )
    beam_threshold: Optional[float] = field(
        default=25.0, metadata={"help": "Specify the threshold for beam."}
    )
    word_score: Optional[float] = field(
        default=1.0,
        metadata={"help": "Specify the score factor of a word while using lm."},
    )
    unk_weight: Optional[float] = field(
        default=-math.inf, metadata={"help": "Specify weight of unk token."}
        #default=0.1, metadata={"help": "Specify weight of unk token."}
    )
    sil_weight: Optional[float] = field(
        default=0.0, metadata={"help": "Specify the weight of sil."}
    )
    nbest: Optional[int] = field(
        default=2, metadata={"help": "Specify the number of beams to select from."}
    )
    kenlm_model: Optional[str] = field(
        default=None, metadata={"help": "Specify the path of the kenlm file."}
    )


DEFAULT: W2V2DecoderArgs = W2V2DecoderArgs(
    #kenlm_model="/hpcwork/zv653460/lm_librispeech_kenlm_word_4g_200kvocab.bin",
    kenlm_model = "/hpcwork/zv653460/lm_librispeech_kenlm_wp_10k_6gram_pruning_000012.bin",
    lexicon="/home/zv653460/decoder-unigram-10000-nbest10.lexicon",
)

DEFAULT_NO_LEXICON: W2V2DecoderArgs = W2V2DecoderArgs(
    #kenlm_model = "/hpcwork/zv653460/lm_wsj_kenlm_word_4g.bin",
    kenlm_model = "/hpcwork/zv653460/lm_wsj_kenlm_char_15g_pruned.bin",
)


def get_target_dict(processor) -> List[str]:
    """
    Get the target dictionary a Wav2Vec2Processor

    Args:
        processor(transformers.Wav2Vec2Processor): processor used by the model
    Returns:
        List of tokens
    """
    return [t.lower() for t in processor.tokenizer.get_vocab().keys()]


class W2V2Decoder(object):
    """
    Adapted https://github.com/huggingface/transformers/blob/9ee266fdf2138cda013a8fe9da434fa497aebfda/examples/research_projects/wav2vec2/run_wav2vec2_eval_with_lm.py
    By deepang17

    Base class for the decoders

    Args:
        args: arguments for the decoder. Unused arguments are ignored
        processor(transformers.Wav2Vec2Processor): processor used by the model
    Returns:
        W2V2Decoder
    """

    def __init__(self, args: W2V2DecoderArgs, processor) -> None:
        tgt_dict = get_target_dict(processor)
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<pad>") if "<pad>" in tgt_dict else tgt_dict.index("<s>")
        )
        if "<sep>" in tgt_dict:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.index("</s>")
        self.asg_transitions = None

    def get_prefix(self, idxs: torch.IntTensor) -> str:
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        prefix_answer = ""
        for i in list(idxs):
            prefix_answer += self.tgt_dict[i]
        return prefix_answer.replace("|", " ").strip().upper()

    def decode(self, logits: torch.Tensor) -> str:
        """
        Decode the output of the model

        Args:
            logits: logits generated by the model
        Returns:
            Decoded str

        Note: the derived class must implement `batch_decode`
        """
        #B, _, _ = logits.size()
        B = logits.shape[0]
        if B != 1:
            print("Decode received a batch instead of a single sample")
        # return [self.decode(i) for i in batch]
        return self.batch_decode(logits)[0]

    def batch_decode(self, logits: torch.Tensor) -> str:
        """
        Decode the output of the model (batch)

        Args:
            logits: logits generated by the model
        Returns:
            Decoded str

        Note: not implemented
        """
        raise NotImplementedError()


class W2V2GreedyDecoder(W2V2Decoder):
    """
    Default greedy decoder from Wav2Vec2Processor

    Args:
        args: arguments for the decoder (ignored)
        processor(transformers.Wav2Vec2Processor): processor used by the model
    """

    def __init__(self, args: W2V2DecoderArgs, processor) -> None:
        super().__init__(args, processor)
        self.processor = processor

    def _greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Find the index of the most probable symbols

        Args:
            logits: output logits from the model
        Returns:
            Tensor with the index of the symbols. It can be decoded by a Wav2Vec2Tokenizer
        """
        #return torch.argmax(logits, axis=-1)
        return logits.argmax(axis=-1)

    def decode(self, logits):
        """
        Decode the output of the model

        Args:
            logits: logits generated by the model
        Returns:
            Decoded str
        """
        return self.processor.decode(self._greedy(logits))

    def batch_decode(self, logits):
        """
        Decode the output of the model (batch)

        Args:
            logits: logits generated by the model
        Returns:
            Decoded str
        """
        #print(self._greedy(logits)[0])
        return self.processor.batch_decode(self._greedy(logits))


class W2V2ViterbiDecoder(W2V2Decoder):
    """
    Adapted https://github.com/huggingface/transformers/blob/9ee266fdf2138cda013a8fe9da434fa497aebfda/examples/research_projects/wav2vec2/run_wav2vec2_eval_with_lm.py
    By deepang17

    Viterbi algorithm to decode the output logits

    Args:
        args: arguments for the decoder (ignored)
        processor(transformers.Wav2Vec2Processor): processor used by the model
    """

    def __init__(self, args: W2V2DecoderArgs, processor) -> None:
        super().__init__(args, processor)

    def batch_decode(self, emissions: torch.Tensor) -> torch.Tensor:
        """
        Decode the output of the model (batch)

        Args:
            emissions: logits generated by the model
        Returns:
            Decoded str
        """
        emissions = to_torch(emissions)
        B, T, N = emissions.size()
        transitions = torch.FloatTensor(N, N).zero_()
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [self.get_prefix(viterbi_path[b].tolist()) for b in range(B)]


class W2V2KenLMDecoder_flashlight(W2V2Decoder):
    """
    Adapted https://github.com/huggingface/transformers/blob/9ee266fdf2138cda013a8fe9da434fa497aebfda/examples/research_projects/wav2vec2/run_wav2vec2_eval_with_lm.py
    By deepang17

    KenLM language model to decode the output logits

    Args:
        args: arguments for the decoder (ignored)
        processor(transformers.Wav2Vec2Processor): processor used by the model
    """

    def __init__(self, args: W2V2DecoderArgs, processor) -> None:
        super().__init__(args, processor)

        if args.lexicon:  # To be tested
            self.lexicon = load_words(args.lexicon)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    # spelling_idxs = [self.tgt_dict.index(token) for token in spelling]
                    if (
                        len(spelling) > 0
                        and all([len(i) == 1 for i in spelling])
                        and (not spelling[0].startswith("_")
                        or spelling[0] == "_")
                    ):  # adapt dictionary
                        spelling_idxs = [
                            self.tgt_dict.index(token)#.upper())
                            for token in spelling
                            if token != "_"
                        ]
                        assert (
                            self.tgt_dict.index("<unk>") not in spelling_idxs
                        ), f"{spelling} {spelling_idxs}"
                        self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(len(self.tgt_dict)),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            if self.asg_transitions is None:
                self.asg_transitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asg_transitions,
                args.unit_lm,
            )
        else:
            assert (
                args.unit_lm
            ), "lexicon free decoding can only be done with a unit language model"

            d = {w: [[w]] for w in self.tgt_dict}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(len(self.tgt_dict)),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=CriterionType.CTC,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def batch_decode(self, emissions: torch.Tensor) -> torch.Tensor:
        """
        Decode the output of the model (batch)

        Args:
            emissions: logits generated by the model
        Returns:
            Decoded str
        """
        emissions = to_torch(emissions)
        B, T, N = emissions.size()
        hypos = []
        progress_bar(0, "Decoding", B)
        for b in range(B):  # each item in the batch
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)
            nbest_results = results[: self.nbest]

            hypos.extend(
                [self.get_prefix(result.tokens) for result in nbest_results]
            )

            print("\n", hypos[-1])
            progress_bar(b+1, "Decoding", B)
        return hypos

class W2V2KenLMDecoder(W2V2Decoder):
    """
    KenLM language model to decode the output logits
    Adapted from the CTC package used by DeepSpeech

    Args:
        args: arguments for the decoder
        processor(transformers.Wav2Vec2Processor): processor used by the model
    """
    def __init__(self, args, processor):
        super().__init__(args, processor)
        #alpha = 2.5 # LM Weight
        alpha = args.lm_weight
        #beta = 1 # LM Usage Reward
        beta = args.word_score
        beam = args.beam
        self.processor = processor
        self.decoder = ctcdecode.CTCBeamDecoder(
            self.tgt_dict,
            model_path=args.kenlm_model,
            alpha=alpha,
            beta=beta,
            cutoff_top_n=10,
            cutoff_prob=0.5,
            #beam_width=50,
            beam_width=beam,
            num_processes=4,
            blank_id=self.blank,
            log_probs_input=True
        )

    def _to_text(self, ids: List[int]) -> str:
        """
        Convert a sequence of tokens ids to str.
        Note: that there is not the CTC blank token and duplicated ids stand for duplicated tokens

        Args:
            ids: list of token ids
        Returns:
            Decoded string
        """
        return "".join([self.tgt_dict[i] for i in ids]).upper().replace("|", " ").strip()

    def batch_decode(self, emissions: torch.Tensor) -> List[str]:
        """
        Decoded the output of the model

        Args:
            emissions: output logits
        Returns:
            List of decoded strings
        """
        # TODO fix torch.tensor(emissions) -> not efficient
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(torch.tensor(emissions))
        results = [beam_results[i][0][:out_lens[i][0]].numpy() for i in range(len(beam_results))]
        #results = self.processor.batch_decode(results)
        return [self._to_text(i) for i in results]

class W2V2NeuSpellDecoder(W2V2Decoder):
    """
    Correct the decoded string with NeuSpell. A BERT model is used.

    Args:
        args: arguments for the decoder, passed also to the base_decoder
        processor: processor used by the model
        base_decoder: W2V2Decor used to decod the logits (default: W2V2GreedyDecoder).
            If the type is passed, the object is created by using args and processor
    """

    def __init__(
        self,
        args,
        processor: W2V2DecoderArgs,
        base_decoder: Union[Type[W2V2Decoder], W2V2Decoder] = W2V2GreedyDecoder,
    ) -> None:
        super().__init__(args, processor)
        if isinstance(base_decoder, type) and isinstance(base_decoder, W2V2Decoder):
            base_decoder = base_decoder(args, processor)
        assert isinstance(
            base_decoder, W2V2Decoder
        ), "Invalid decoder, not W2V2Decoder instance"

        self.base_decoder = base_decoder
        self.checker = BertChecker()
        self.checker.from_pretrained()
        #print(self.checker.correct("Testing Neuspell checker"))

    def batch_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decode the output of the model (batch)

        Args:
            logits: logits generated by the model
        Returns:
            Decoded str
        """
        text = self.base_decoder.batch_decode(logits)
        # W2V2 doesn't predict spaces aroud ', but BERT does
        return [self.checker.correct(t).replace(" ' ", "'") if len(t) != 0 else "" for t in text]

    def decoder(self, logtis: torch.Tensor) -> torch.Tensor:
        """
        Decode the output of the model

        Args:
            logits: logits generated by the model
        Returns:
            Decoded str
        """
        text = self.base_decoder.decode(logits)
        if len(text) == 0:
            return ""
        else:
            return self.checker.correct(text)
