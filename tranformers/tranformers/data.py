from typing import Any, Dict, Iterable

import datasets
import torch
import torch.utils
import torch.utils.data
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset

from .config import TokenizerConfig


def get_or_build_tokenizer(
    config: TokenizerConfig, ds: datasets.Dataset, lang: str
) -> Tokenizer:
    """_summary_

    Args:
        config (TokenizerConfig): _description_
        ds (Dataset): _description_
        lang (str): _description_

    Returns:
        Tokenizer: _description_
    """

    if not config.tokenizer_path.exists():
        tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[EOS]", "[SOS]", "[PAD]"], min_frequency=2
        )
        tokenizer.train_from_iterator(
            iterator=get_sentence_iter(ds, lang), trainer=trainer
        )
        tokenizer.save(config.tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
    return tokenizer


def get_sentence_iter(ds: Dataset, lang: str) -> Iterable:
    """_summary_

    Args:
        ds (Dataset): _description_
        lang (str): _description_

    Returns:
        Iterable: _description_

    Yields:
        Iterator[Iterable]: _description_
    """
    for item in ds:
        yield item["translation"][lang]


def get_raw_data(config: TokenizerConfig) -> Dataset:
    """_summary_

    Args:
        config (TokenizerConfig): _description_

    Returns:
        Dataset: _description_
    """
    ds_raw = load_dataset(
        config.hf_dataset_name, config.translation_task, split="train"
    )
    return ds_raw


class BilingualDataset(Dataset):

    def __init__(
        self,
        config: TokenizerConfig,
        ds: datasets.Dataset,
        tokenizer_src: Tokenizer,
        tokenizer_trg: Tokenizer,
    ) -> DataLoader:
        """_summary_

        Args:
            config (TokenizerConfig): _description_
            ds (Dataset): _description_
            tokenizer_src (Tokenizer): _description_
            tokenizer_trg (Tokenizer): _description_

        Returns:
            DataLoader: _description_
        """
        self.config = config
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.sos_token = tokenizer_src.token_to_id("[SOS]")
        self.eos_token = tokenizer_src.token_to_id("[EOS]")
        self.pad_token = tokenizer_src.token_to_id("[PAD]")
        self.causal_mask = self.causal_mask_fn(self.config.seq_max_len_trg)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: Any) -> Dict[str, torch.Tensor]:
        src_and_target = self.ds[index]
        src_text = src_and_target["translation"][self.config.src_lang]
        trg_text = src_and_target["translation"][self.config.trg_lang]

        enc_src = self.tokenizer_src.encode(src_text).ids
        enc_trg = self.tokenizer_src.encode(trg_text).ids

        enc_src_num_pads = (
            self.config.seq_max_len - len(enc_src) - 2
        )  # minus 2 because we need to add sos and oes
        assert (
            enc_src_num_pads > 0
        ), f"Source sequence too long {len(enc_src)} and max {self.config.seq_max_len_src}"
        enc_trg_num_pads = (
            self.config.seq_max_len - len(enc_trg) - 1
        )  # minus 1 because we need to add sos
        assert (
            enc_trg_num_pads > 0
        ), f"Target sequence too long {len(enc_trg)} and max {self.config.seq_max_len_trg}"

        encoder_input = (
            [self.sos_token]
            + enc_src
            + [self.pad_token for _ in range(enc_src_num_pads)]
            + [self.eos_token]
        )
        decoder_input = (
            [self.sos_token]
            + enc_trg
            + [self.pad_token for _ in range(enc_trg_num_pads)]
        )
        label = (
            enc_trg
            + [self.eos_token]
            + [self.pad_token for _ in range(enc_trg_num_pads)]
        )
        assert (
            encoder_input.size[0]
            == decoder_input.size[0]
            == label.size[0]
            == self.config.seq_max_len
        ), f"Encoder , decoder or target not equal to max_seq_size {encoder_input.size[0]},{decoder_input.size[0]} or {label.size[0]}not equal to {self.config.seq_max_len}"
        # input to encoder , input to decoder and label ( or target)
        return {
            "encoder_input": torch.Tensor(encoder_input, dtype=torch.int64),
            # ( 1,1,seq_length )
            "encoder_mask": torch.Tensor(
                [0 if code == self.pad_token else 1 for code in encoder_input],
                dtype=torch.int64,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            "decoder_input": torch.Tensor(decoder_input, dtype=torch.int64),
            # ( 1,1, seq_length ) & ( 1,seq_length,seq_length) --> ( 1,seq_length,seq_length)
            "decoder_mask": torch.Tensor(
                [0 if code == self.pad_token else 1 for code in decoder_input],
                dtype=torch.int64,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            & self.causal_mask,
            "label": torch.Tensor(label, dtype=torch.int64),
        }

    def causal_mask_fn(self, size: int) -> torch.Tensor:
        """_summary_

        Args:
            size (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        # ( 1, seq_length,seq_length)
        mask = torch.triu(torch.ones_like((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0
