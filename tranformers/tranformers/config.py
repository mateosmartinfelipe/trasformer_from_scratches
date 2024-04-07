from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    src_lang: str
    trg_lang: str
    root_path: str
    tokenizer_folder: str
    hf_dataset_name: str
    opus_books_subset: str
    seq_max_len_trg: int
    seq_max_len_src: int
    batch_size_train: int
    batch_size_val: int

    def __post_init__(self):
        self.tokenizer_path = (
            Path(self.root_path)
            / self.tokenizer_folder
            / f"tokenizer_{self.src_lang}.json"
        )
        self.translation_task = f"{self.src_lang}-{self.trg_lang}"
