import argparse

from torch.utils.data import DataLoader, random_split

from .config import Config
from .data import BilingualDataset, get_or_build_tokenizer, get_raw_data
from .model import build_model
from .parser import DefaultParser


def start():
    """_summary_"""
    parser = argparse.ArgumentParser = DefaultParser()
    args = parser.parser()
    config = Config(
        src_lang=args.src_lang,
        trg_lang=args.src_lang,
        root_path=args.root_path,
        tokenizer_folder=args.tokenizer_folder,
        hf_dataset_name=args.hf_dataset_name,
        opus_books_subset=args.opus_books_subset,
        seq_max_len_src=args.seq_max_len_src,
        seq_max_len_trg=args.seq_max_len_trg,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
    )
    ds = get_raw_data(config=config)
    tokenizer_src = get_or_build_tokenizer(config=config, ds=ds, lang=config.src_lang)
    tokenizer_trg = get_or_build_tokenizer(config=config, ds=ds, lang=config.src_lang)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])

    train_dataset = BilingualDataset(
        config, train_dataset, tokenizer_src, tokenizer_trg
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config.batch_size_train
    )
    val_dataset = BilingualDataset(config, val_dataset, tokenizer_src, tokenizer_trg)
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=config.batch_size_val
    )
    model = build_model(
        src_seq_length=config.seq_max_len_src,
        trg_seq_length=config.seq_max_len_trg,
        src_voc_size=tokenizer_src.get_vocab_size(),
        trg_voc_size=tokenizer_trg.get_vocab_size(),
    )

    # make the training loop


if __name__ == "__main__":
    start()
