import sys

sys.path.append("../")

import pickle
import text_tools
import tools
import torch
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
from newsroom.analyze import Fragments
import time
import datetime

# Similar to Wav2Vec2
diff_pipeline = tools.get_pipeline(
    text_tools.preprocess_numbers,
    text_tools.preprocess_spaces,
    text_tools.preprocess_clean,
    text_tools.preprocess_no_punctuation,
    text_tools.preprocess_case_normalization,
    text_tools.preprocess_spaces,
)

# Simply clean the text
easy_pipeline = tools.get_pipeline(
    text_tools.preprocess_numbers,
    text_tools.preprocess_spaces,
    text_tools.preprocess_clean,
)

# For the reference summary
target_pipeline = tools.get_pipeline(
    text_tools.preprocess_numbers,
    text_tools.preprocess_spaces,
    text_tools.preprocess_clean,
)


class SummarizationSet(Dataset):
    def __init__(self, docs, targets, pipeline, tokenizer, coverage=0.5, max_len=8000):
        assert len(docs) == len(
            targets
        ), "Documents and targets must have the same size."
        # tar_pipeline = tools.get_pipeline(
        #    text_tools.preprocess_numbers,
        #    text_tools.preprocess_spaces,
        #    text_tools.preprocess_clean
        # )
        self.tok = tokenizer

        self.x = pipeline(docs)
        # self.y = tar_pipeline(targets)
        self.y = target_pipeline(targets)
        self.x, self.y = zip(
            *[i for i in zip(self.x, self.y) if len(i[0].strip().split(" ")) <= max_len and len(i[1].strip()) > 0]
        )

        # FIX pegasus overfit on this
        #self.x, self.y = zip(
        #    *[i for i in zip(self.x, self.y) if "This animation is part of the TED-Ed series".lower() not in i[1]]
        #)

        print("Docs:", len(self.x))
        print("Targets:", len(self.y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print("ID:", idx)
        """item = {k:v for k, v in self.tok.prepare_seq2seq_batch(self.x[idx], self.y[idx], padding='longest', return_tensors='pt', pad_to_multiple_of=32, truncation=True).items()}
        #item["decoder_input_ids"] = self.tok.prepare_seq2seq_batch(self.y[idx], padding='longest', return_tensors='pt', pad_to_multiple_of=32)["input_ids"]
        item["decoder_input_ids"] = item["labels"].detach().clone()
        #item["labels"] = item["decoder_input_ids"].detach().clone()
        item["labels"][item["labels"] == self.tok.pad_token_id] = -100
        if "attention_mask" not in item:
            print("Missing mask on", idx)
            item["attention_mask"] = torch.ones(item["input_ids"].shape)"""
        item = {
            "doc": self.x[idx],
            "tar": self.y[idx],
        }
        return item


def main(
    train_doc_file,
    train_targets_file,
    test_docs_file,
    test_targets_file,
    model,
    epochs,
    batch_size,
    test_batch_size,
    warmup_steps,
    weight_decay,
    log,
    output_dir,
    max_len,
    len_penalty,
    device,
    freeze,
    resume,
    coverage,
    dropout,
    do_sample,
    temperature,
    top_k,
    top_p,
    max_pos_emb,
    easy,
    evaluate=False,
    debug=False,
):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("-Device:", device)
    print("-Easy pipeline:", easy)
    print("-Loading model...")
    # load model
    model = text_tools.SimpleSummarizationModel(
        device,
        text_tools.MODELS.Pegasus,
        model=model,
        max_len=max_len,
        len_penalty=len_penalty,
        dropout=dropout,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_pos_emb=max_pos_emb,
    )
    freeze_map = {
        "e": model.freeze_encoder,
        "d": model.freeze_decoder,
        "ie": model.freeze_input_embeddings,
        "oe": model.freeze_output_embeddings,
        "ne": model.freeze_not_embeddings,
    }
    if freeze is not None:
        for i in freeze:
            freeze_map[i]()

    print("--done")

    print("-Loading data...")
    # load data
    with open(test_docs_file, "rb") as f:
        test_docs = pickle.load(f)
    with open(test_targets_file, "rb") as f:
        test_targets = pickle.load(f)

    with open(train_doc_file, "rb") as f:
        train_docs = pickle.load(f)
    with open(train_targets_file, "rb") as f:
        train_targets = pickle.load(f)

    pipeline = None

    if easy:
        pipeline = easy_pipeline
    else:
        pipeline = diff_pipeline

    if debug:
        train = SummarizationSet(
            train_docs[:500], train_targets[:500], pipeline, model.tokenizer, coverage
        )
        test = SummarizationSet(
            test_docs[:100], test_targets[:100], pipeline, model.tokenizer, coverage
        )
    else:
        train = SummarizationSet(
            train_docs, train_targets, pipeline, model.tokenizer, coverage
        )
        test = SummarizationSet(
            test_docs, test_targets, pipeline, model.tokenizer, coverage
        )
    print("--done")

    print("Train set size:", len(train))
    print("Test set size:", len(test))

    # train
    model.setup_trainer(
        train,
        test,
        epochs,
        output_dir,
        batch_size,
        test_batch_size,
        warmup_steps,
        weight_decay,
        log,
        resume,
    )

    if evaluate:
        model.evaluate()
    else:
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pegasus")

    # .add_argument_group(title=None, description=None)¶
    group = parser.add_argument_group("Training")
    group.add_argument(
        "-e", "--epochs", default=2000, type=int, help="Number of training epochs"
    )
    group.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="Trained model to fine-tune. If None, base Pegasus is used",
    )
    group.add_argument(
        "-b", "--batch", default=1, type=int, help="Batch size during the training"
    )
    group.add_argument(
        "-tb", "--test-batch", default=1, type=int, help="Batch size during the testing"
    )
    group.add_argument("-w", "--warmup", default=100, type=int, help="")
    group.add_argument(
        "-wd",
        "--weight",
        default=0.01,
        type=float,
        help="Weight decay for Adafactor optimizer",
    )
    group.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout of the fully connected layers",
    )

    group = parser.add_argument_group("Datasets")
    group.add_argument(
        "--train-x",
        type=str,
        required=True,
        help="Pickle file containing the training documents",
    )
    group.add_argument(
        "--train-y",
        type=str,
        required=True,
        help="Pickle file containing the training summaries",
    )
    group.add_argument(
        "--test-x",
        type=str,
        required=True,
        help="Pickle file containing the testing documents",
    )
    group.add_argument(
        "--test-y",
        type=str,
        required=True,
        help="Pickle file containing the testing summaries",
    )
    group.add_argument(
        "--easy",
        action="store_true",
        help="Clean the text instead of mimicking the Wav2Vec2 output",
    )

    group = parser.add_argument_group("Outputs")
    group.add_argument(
        "-l", "--log", default="./logs", type=str, help="Folder for logs"
    )
    group.add_argument(
        "-o",
        "--output",
        default="./results",
        type=str,
        help="Folder to store the trained models",
    )

    group = parser.add_argument_group("Generation settings")
    group.add_argument(
        "-ml",
        "--max-len",
        default=10000,
        type=int,
        help="Maximum length of the summary",
    )
    group.add_argument(
        "-lp",
        "--length-penalty",
        default=1.1,
        type=float,
        help="Exponential penaly for long summaries (>1: prefer long, 1: no penalty, <1: prefer short)",
    )
    group.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy/beam search",
    )
    group.add_argument("--top-k", default=0, type=int, help="Use top-k filtering")
    group.add_argument(
        "--top-p", default=1, type=float, help="Use top-p/nucleus filtering"
    )
    group.add_argument(
        "--temperature", default=1.0, type=float, help="Temperature for sampling"
    )

    group = parser.add_argument_group("Script commands")
    group.add_argument("--debug", action="store_true", help="Test run of the script")
    group.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    # Remove coverage
    group.add_argument(
        "-c",
        "--coverage",
        default=0.5,
        type=float,
        help="Select documents whose refernce target has at least this extrative fragment coverage",
    )

    group = parser.add_argument_group("Model settings")
    group.add_argument(
        "-d",
        "--device",
        default="auto",
        type=str,
        help="Device to useg (values: cuda, cpu, auto (default))",
    )
    # parser.add_argument("-f", "--freeze-encoder", action="store_true", help="Freeze the encoder")
    group.add_argument(
        "-f",
        "--freeze",
        action="append",
        choices=[
            "encoder",
            "decoder",
            "e",
            "d",
            "ie",
            "oe",
            "input_embeddings",
            "output_embedding",
            "ne",
            "not_embeddings",
        ],
        help="Freeze part of the model during the training",
    )
    group.add_argument(
        "-p",
        "--max-pos-emb",
        type=int,
        default=512,
        help="Maximum input length for the model",
    )
    group.add_argument(
        "--eval",
        action="store_true",
        help="Evalaute the model instead of training it"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" or args.device == "cpu":
        device = args.device
    else:
        raise ValueError(f"Invalid device:{args.device}")

    freeze_map = {
        "encoder": "e",
        "decoder": "d",
        "e": "e",
        "d": "d",
        "ie": "ie",
        "oe": "oe",
        "input_embeddings": "ie",
        "output_embedding": "oe",
        "ne": "ne",
        "not_embeddings": "ne",
    }
    if args.freeze is None:
        freeze = []
    else:
        freeze = [freeze_map[i] for i in args.freeze]

    print("Starting at:", datetime.datetime.now())
    start = time.time_ns()

    print("Training...")
    if args.debug:
        print("Debug...")
        main(
            train_doc_file=args.train_x,
            train_targets_file=args.train_y,
            test_docs_file=args.test_x,
            test_targets_file=args.test_y,
            model=args.model,
            epochs=5,
            batch_size=args.batch,
            test_batch_size=args.test_batch,
            warmup_steps=args.warmup,
            weight_decay=args.weight,
            log=args.log,
            output_dir=args.output,
            max_len=args.max_len,
            len_penalty=args.length_penalty,
            device=device,
            freeze=args.freeze,
            resume=args.resume,
            coverage=args.coverage,
            dropout=args.dropout,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_pos_emb=args.max_pos_emb,
            easy=args.easy,
            evaluate=args.eval,
            debug=True,
        )
    else:
        main(
            train_doc_file=args.train_x,
            train_targets_file=args.train_y,
            test_docs_file=args.test_x,
            test_targets_file=args.test_y,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            test_batch_size=args.test_batch,
            warmup_steps=args.warmup,
            weight_decay=args.weight,
            log=args.log,
            output_dir=args.output,
            max_len=args.max_len,
            device=device,
            len_penalty=args.length_penalty,
            freeze=freeze,
            resume=args.resume,
            coverage=args.coverage,
            dropout=args.dropout,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_pos_emb=args.max_pos_emb,
            evaluate=args.eval,
            easy=args.easy,
            #evaluate=args.eval,
        )
    print("--done")

    end = time.time_ns()
    print("Ending at:", datetime.datetime.now())
    delta = datetime.timedelta(seconds=(end - start) / 1e9)
    print("Time:", delta)
    if args.debug:
        print("Average per epoch:", delta / 5)
    else:
        print("Average per epoch:", delta / args.epochs)
