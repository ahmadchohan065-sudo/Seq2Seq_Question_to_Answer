import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD, UNK, BOS, EOS = 0, 1, 2, 3

class QGDataset(Dataset):
    def __init__(self, tsv_path, sp):
        self.pairs = []
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                src, tgt = line.split("\t")
                self.pairs.append((src, tgt))
        self.sp = sp

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.sp.encode(src)
        tgt_ids = [BOS] + self.sp.encode(tgt) + [EOS]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_seqs])
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs])

    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=PAD)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD)

    return src_padded, src_lens, tgt_padded, tgt_lens


if __name__ == "__main__":
    # quick sanity check
    import sentencepiece as spm
    from torch.utils.data import DataLoader

    sp = spm.SentencePieceProcessor(model_file="ur_sp.model")
    ds = QGDataset("data/train.tsv", sp)
    print("Dataset size:", len(ds))

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    src, src_lens, tgt, tgt_lens = next(iter(loader))
    print("src shape:", src.shape, "tgt shape:", tgt.shape)