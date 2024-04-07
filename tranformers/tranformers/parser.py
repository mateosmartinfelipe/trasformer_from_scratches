import argparse


class DefaultParser:
    """_summary_"""

    def __init__(self) -> None:
        """tokenizer_path_root: str
        hf_dataset_name: str = "opus_books"
        opus_books_subset: str = ""
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--root_path", help="Tokenizer", type=str, required=True
        )
        self.parser.add_argument(
            "--tokenizer-folder", help="Tokenizer", type=str, default="tokenizer"
        )
        self.parser.add_argument(
            "--hf_dataset_name", help="hf dataset", type=str, default="opus_books"
        )
        self.parser.add_argument(
            "--opus_books_subset", help="subset of languages", type=str, required=True
        )
        self.parser.add_argument(
            "--src-lang", help="Source_language", type=str, required=True
        )
        self.parser.add_argument(
            "--trg-lang", help="target_language", type=str, required=True
        )
        self.parser.add_argument(
            "--seq-max-len-src", help="length of sequence for tr", type=int, default=512
        )
        self.parser.add_argument(
            "--seq-max-len-trg",
            help="length of sequence for trg",
            type=int,
            default=512,
        )
        self.parser.add_argument(
            "--batch-size-train",
            help="Train batch size",
            type=int,
            default=10,
        )
        self.parser.add_argument(
            "--batch-size-val",
            help="validation batch size",
            type=int,
            default=10,
        )
