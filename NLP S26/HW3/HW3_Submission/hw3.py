from __future__ import annotations

import argparse
import gzip
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Special tokens / constants
# -----------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
IGNORE_INDEX = -100
CASE_PAD_IDX = 5
PAD_CHAR_IDX = 0
UNK_CHAR_IDX = 1


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_data(path: str):
    """Return list of (words, tags) sentence pairs."""
    sentences = []
    words, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
                continue

            parts = line.split()
            if len(parts) == 3:
                words.append(parts[1])
                tags.append(parts[2])
            elif len(parts) == 2:
                words.append(parts[1])
                tags.append("O")

    if words:
        sentences.append((words, tags))
    return sentences


# -----------------------------------------------------------------------------
# Vocab building
# -----------------------------------------------------------------------------
def build_word_vocab(train_sents, min_freq: int = 1):
    counter = Counter()
    for words, _ in train_sents:
        counter.update(words)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def build_char_vocab(train_sents, min_freq: int = 1):
    counter = Counter()
    for words, _ in train_sents:
        for w in words:
            counter.update(list(w))
    vocab = {PAD_TOKEN: PAD_CHAR_IDX, UNK_TOKEN: UNK_CHAR_IDX}
    for ch, c in counter.items():
        if c >= min_freq:
            vocab[ch] = len(vocab)
    return vocab


def build_tag_vocab(train_sents):
    tags = sorted({t for _, sent_tags in train_sents for t in sent_tags})
    return {t: i for i, t in enumerate(tags)}


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
def load_glove(path: str, dim: int = 100) -> Dict[str, np.ndarray]:
    vectors = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] == dim:
                vectors[word] = vec
    print(f"Loaded {len(vectors)} GloVe vectors")
    return vectors


def build_glove_matrix(word2idx, glove_vectors, embed_dim: int = 100):
    rng = np.random.default_rng(SEED)
    matrix = np.zeros((len(word2idx), embed_dim), dtype=np.float32)
    found = 0
    for word, idx in word2idx.items():
        if word == PAD_TOKEN:
            continue
        if word == UNK_TOKEN:
            matrix[idx] = rng.normal(0, 0.1, embed_dim).astype(np.float32)
            continue
        vec = glove_vectors.get(word)
        if vec is None:
            vec = glove_vectors.get(word.lower())
        if vec is not None:
            matrix[idx] = vec
            found += 1
        else:
            matrix[idx] = rng.normal(0, 0.1, embed_dim).astype(np.float32)
    print(f"GloVe coverage: {found}/{len(word2idx)} ({100.0 * found / len(word2idx):.1f}%)")
    return matrix


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
def case_category(word: str) -> int:
    if not word:
        return 3
    if any(c.isdigit() for c in word):
        return 4
    if word.islower():
        return 0
    if word.isupper():
        return 1
    if len(word) > 1 and word[0].isupper() and word[1:].islower():
        return 2
    return 3


def repair_bio(tags: List[str]) -> List[str]:
    repaired = []
    prev_type = None
    for tag in tags:
        if tag == "O":
            repaired.append(tag)
            prev_type = None
            continue
        if tag.startswith("B-"):
            repaired.append(tag)
            prev_type = tag[2:]
            continue
        if tag.startswith("I-"):
            cur_type = tag[2:]
            if prev_type != cur_type:
                repaired.append("B-" + cur_type)
                prev_type = cur_type
            else:
                repaired.append(tag)
        else:
            repaired.append(tag)
            prev_type = None
    return repaired


def extract_spans(tags: Sequence[str]):
    spans = set()
    cur_type = None
    start = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.add((cur_type, start, i - 1))
            cur_type = tag[2:]
            start = i
        elif tag.startswith("I-"):
            t = tag[2:]
            if cur_type != t:
                if cur_type is not None:
                    spans.add((cur_type, start, i - 1))
                cur_type = t
                start = i
        else:
            if cur_type is not None:
                spans.add((cur_type, start, i - 1))
                cur_type = None
                start = None
    if cur_type is not None:
        spans.add((cur_type, start, len(tags) - 1))
    return spans


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class NERDataset(Dataset):
    def __init__(self, sentences, word2idx, tag2idx, char2idx=None):
        self.samples = []
        for words, tags in sentences:
            word_ids = torch.tensor([word2idx.get(w, word2idx[UNK_TOKEN]) for w in words], dtype=torch.long)
            case_ids = torch.tensor([case_category(w) for w in words], dtype=torch.long)
            if char2idx is not None:
                char_ids = [
                    [char2idx.get(ch, char2idx[UNK_TOKEN]) for ch in w] if len(w) > 0 else [char2idx[UNK_TOKEN]]
                    for w in words
                ]
            else:
                char_ids = []
            tag_ids = torch.tensor([tag2idx[t] for t in tags], dtype=torch.long)
            self.samples.append((word_ids, case_ids, char_ids, tag_ids, words))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    word_ids, case_ids, char_ids, tag_ids, raw_words = zip(*batch)
    lengths = torch.tensor([len(x) for x in word_ids], dtype=torch.long)
    word_pad = pad_sequence(word_ids, batch_first=True, padding_value=0)
    case_pad = pad_sequence(case_ids, batch_first=True, padding_value=CASE_PAD_IDX)
    tag_pad = pad_sequence(tag_ids, batch_first=True, padding_value=IGNORE_INDEX)

    has_chars = len(char_ids[0]) > 0
    if has_chars:
        max_seq_len = word_pad.size(1)
        max_word_len = max(max(max(len(chars) for chars in sent) for sent in char_ids), 1)
        char_pad = torch.zeros((len(batch), max_seq_len, max_word_len), dtype=torch.long)
        for i, sent in enumerate(char_ids):
            for j, chars in enumerate(sent):
                if len(chars) > 0:
                    char_pad[i, j, : len(chars)] = torch.tensor(chars, dtype=torch.long)
    else:
        char_pad = torch.zeros((len(batch), word_pad.size(1), 1), dtype=torch.long)

    return word_pad, case_pad, char_pad, tag_pad, lengths, raw_words


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class SimpleBLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, linear_out, num_tags,
                 num_layers=1, dropout=0.33, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, linear_out)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_out, num_tags)

    def forward(self, word_ids, lengths):
        emb = self.input_dropout(self.embedding(word_ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=word_ids.size(1))
        out = self.output_dropout(out)
        out = self.elu(self.linear(out))
        return self.classifier(out)


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size: int, char_emb_dim: int, out_channels: int, kernels=(3, 4, 5), dropout=0.15):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=PAD_CHAR_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(char_emb_dim, out_channels, kernel_size=k, padding=0) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_channels * len(kernels)
        self.max_kernel = max(kernels)

    def forward(self, char_ids):
        B, T, W = char_ids.shape
        flat = char_ids.view(B * T, W)
        x = self.char_embed(flat)  # [B*T, W, C]
        x = x.transpose(1, 2)      # [B*T, C, W]
        if W < self.max_kernel:
            x = F.pad(x, (0, self.max_kernel - W))
        feats = []
        for conv in self.convs:
            y = torch.relu(conv(x))
            y = torch.max(y, dim=-1).values
            feats.append(y)
        out = torch.cat(feats, dim=-1)
        out = self.dropout(out)
        return out.view(B, T, -1)


class AdvancedBLSTM(nn.Module):
    def __init__(self,
                 glove_matrix,
                 num_tags,
                 num_case_cats=5,
                 case_embed_dim=32,
                 char_vocab_size: Optional[int] = None,
                 char_emb_dim: int = 30,
                 char_out_channels: int = 32,
                 char_kernels=(3, 4, 5),
                 hidden_dim=256,
                 linear_out=128,
                 num_layers=1,
                 dropout=0.33,
                 use_char=False):
        super().__init__()
        vocab_size, embed_dim = glove_matrix.shape
        self.word_embed = nn.Embedding.from_pretrained(
            torch.tensor(glove_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0,
        )
        self.case_embed = nn.Embedding(num_case_cats + 1, case_embed_dim, padding_idx=CASE_PAD_IDX)
        self.word_dropout = nn.Dropout(0.20)
        self.use_char = use_char and (char_vocab_size is not None)
        if self.use_char:
            self.char_cnn = CharCNN(
                char_vocab_size=char_vocab_size,
                char_emb_dim=char_emb_dim,
                out_channels=char_out_channels,
                kernels=char_kernels,
                dropout=0.15,
            )
            char_dim = self.char_cnn.out_dim
        else:
            self.char_cnn = None
            char_dim = 0
        lstm_in = embed_dim + case_embed_dim + char_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, linear_out)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_out, num_tags)

    def forward(self, word_ids, case_ids, char_ids, lengths):
        pieces = [self.word_embed(word_ids), self.case_embed(case_ids)]
        if self.use_char:
            pieces.append(self.char_cnn(char_ids))
        x = torch.cat(pieces, dim=-1)
        x = self.word_dropout(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=word_ids.size(1))
        out = self.post_lstm_dropout(out)
        out = self.elu(self.linear(out))
        out = self.post_lstm_dropout(out)
        return self.classifier(out)


# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------
def train_epoch_simple(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for word_ids, _, _, tag_ids, lengths, _ in loader:
        word_ids = word_ids.to(DEVICE)
        tag_ids = tag_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(word_ids, lengths)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), tag_ids.reshape(B * T))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def train_epoch_advanced(model, loader, optimizer, criterion, grad_clip=5.0):
    model.train()
    total_loss = 0.0
    for word_ids, case_ids, char_ids, tag_ids, lengths, _ in loader:
        word_ids = word_ids.to(DEVICE)
        case_ids = case_ids.to(DEVICE)
        char_ids = char_ids.to(DEVICE)
        tag_ids = tag_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(word_ids, case_ids, char_ids, lengths)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), tag_ids.reshape(B * T))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate_simple(model, loader, idx2tag):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for word_ids, _, _, tag_ids, lengths, _ in loader:
            word_ids = word_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            logits = model(word_ids, lengths)
            preds = logits.argmax(-1).cpu()
            tag_ids = tag_ids.cpu()
            lengths = lengths.cpu()
            for i, l in enumerate(lengths.tolist()):
                gold = [idx2tag[int(x)] for x in tag_ids[i, :l]]
                pred = [idx2tag[int(x)] for x in preds[i, :l]]
                gold_spans = extract_spans(gold)
                pred_spans = extract_spans(pred)
                tp += len(gold_spans & pred_spans)
                fp += len(pred_spans - gold_spans)
                fn += len(gold_spans - pred_spans)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def evaluate_advanced(model, loader, idx2tag, repair_output: bool = True):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for word_ids, case_ids, char_ids, tag_ids, lengths, _ in loader:
            word_ids = word_ids.to(DEVICE)
            case_ids = case_ids.to(DEVICE)
            char_ids = char_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            logits = model(word_ids, case_ids, char_ids, lengths)
            preds = logits.argmax(-1).cpu()
            tag_ids = tag_ids.cpu()
            lengths = lengths.cpu()
            for i, l in enumerate(lengths.tolist()):
                gold = [idx2tag[int(x)] for x in tag_ids[i, :l]]
                pred = [idx2tag[int(x)] for x in preds[i, :l]]
                if repair_output:
                    pred = repair_bio(pred)
                gold_spans = extract_spans(gold)
                pred_spans = extract_spans(pred)
                tp += len(gold_spans & pred_spans)
                fp += len(pred_spans - gold_spans)
                fn += len(gold_spans - pred_spans)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def predict_simple(model, sentences, word2idx, idx2tag, out_path):
    model.eval()
    with open(out_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for words, _ in sentences:
                word_ids = torch.tensor([word2idx.get(w, word2idx[UNK_TOKEN]) for w in words], dtype=torch.long).unsqueeze(0).to(DEVICE)
                lengths = torch.tensor([len(words)], dtype=torch.long).to(DEVICE)
                logits = model(word_ids, lengths)
                pred_ids = logits.argmax(-1).squeeze(0).cpu().tolist()
                for i, (w, pred_id) in enumerate(zip(words, pred_ids), start=1):
                    f.write(f"{i} {w} {idx2tag[pred_id]}\n")
                f.write("\n")
    print(f"Saved -> {out_path}")


def predict_advanced(model, sentences, word2idx, char2idx, idx2tag, out_path, repair_output: bool = True):
    model.eval()
    with open(out_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for words, _ in sentences:
                word_ids = torch.tensor([word2idx.get(w, word2idx[UNK_TOKEN]) for w in words], dtype=torch.long).unsqueeze(0).to(DEVICE)
                case_ids = torch.tensor([case_category(w) for w in words], dtype=torch.long).unsqueeze(0).to(DEVICE)
                max_word_len = max(1, max((len(w) for w in words), default=1))
                char_tensor = torch.zeros((1, len(words), max_word_len), dtype=torch.long)
                for j, w in enumerate(words):
                    chars = [char2idx.get(ch, char2idx[UNK_TOKEN]) for ch in w]
                    if chars:
                        char_tensor[0, j, : len(chars)] = torch.tensor(chars, dtype=torch.long)
                char_tensor = char_tensor.to(DEVICE)
                lengths = torch.tensor([len(words)], dtype=torch.long).to(DEVICE)
                logits = model(word_ids, case_ids, char_tensor, lengths)
                pred_ids = logits.argmax(-1).squeeze(0).cpu().tolist()
                pred_tags = [idx2tag[i] for i in pred_ids]
                if repair_output:
                    pred_tags = repair_bio(pred_tags)
                for i, (w, tag) in enumerate(zip(words, pred_tags), start=1):
                    f.write(f"{i} {w} {tag}\n")
                f.write("\n")
    print(f"Saved -> {out_path}")


# -----------------------------------------------------------------------------
# Task runners
# -----------------------------------------------------------------------------
def run_task1(train_sents, dev_sents, test_sents, data_dir: str, output_dir: str):
    print("\n=== Task 1: Simple BiLSTM ===")
    EMBED_DIM = 100
    NUM_LSTM_LAYERS = 1
    LSTM_HIDDEN = 256
    LSTM_DROPOUT = 0.33
    LINEAR_OUT_DIM = 128

    EPOCHS = 40
    BATCH_SIZE = 16
    LEARNING_RATE = 0.1
    LR_DECAY = 0.5
    LR_PATIENCE = 2
    GRAD_CLIP = 1.0

    word2idx = build_word_vocab(train_sents)
    tag2idx = build_tag_vocab(train_sents)
    idx2tag = {v: k for k, v in tag2idx.items()}

    train_ds = NERDataset(train_sents, word2idx, tag2idx, char2idx=None)
    dev_ds = NERDataset(dev_sents, word2idx, tag2idx, char2idx=None)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, generator=g)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SimpleBLSTM(
        vocab_size=len(word2idx),
        embed_dim=EMBED_DIM,
        hidden_dim=LSTM_HIDDEN,
        linear_out=LINEAR_OUT_DIM,
        num_tags=len(tag2idx),
        num_layers=NUM_LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
        pad_idx=word2idx[PAD_TOKEN],
    ).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=LR_DECAY, patience=LR_PATIENCE)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_f1 = -1.0
    best_path = os.path.join(output_dir, "blstm1.pt")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch_simple(model, train_loader, optimizer, criterion)
        p, r, f1 = evaluate_simple(model, dev_loader, idx2tag)
        scheduler.step(f1)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | P: {p:.4f} R: {r:.4f} F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "word2idx": word2idx,
                    "tag2idx": tag2idx,
                    "config": {
                        "embed_dim": EMBED_DIM,
                        "hidden_dim": LSTM_HIDDEN,
                        "linear_out": LINEAR_OUT_DIM,
                        "num_layers": NUM_LSTM_LAYERS,
                        "dropout": LSTM_DROPOUT,
                    },
                },
                best_path,
            )
            print(f"  saved best model (F1={best_f1:.4f})")

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    p, r, f1 = evaluate_simple(model, dev_loader, idx2tag)
    print(f"Best Dev | P: {p:.4f} R: {r:.4f} F1: {f1:.4f}")

    predict_simple(model, dev_sents, word2idx, idx2tag, os.path.join(output_dir, "dev1.out"))
    predict_simple(model, test_sents, word2idx, idx2tag, os.path.join(output_dir, "test1.out"))


def run_task2(train_sents, dev_sents, test_sents, glove_path: str, output_dir: str):
    print("\n=== Task 2: GloVe + Case (+ optional char CNN) ===")
    EMBED_DIM = 100
    CASE_EMBED_DIM = 32
    CHAR_EMBED_DIM = 30
    CHAR_CNN_CHANNELS = 32
    CHAR_KERNELS = (3, 4, 5)

    NUM_LSTM_LAYERS = 1
    LSTM_HIDDEN = 256
    LSTM_DROPOUT = 0.33
    LINEAR_OUT_DIM = 128

    BATCH_SIZE = 32
    EPOCHS = 28
    PATIENCE = 5
    GRAD_CLIP = 5.0
    LR_WORD = 3e-4
    LR_OTHER = 1e-3
    WEIGHT_DECAY = 1e-2

    glove = load_glove(glove_path, dim=EMBED_DIM)
    word2idx = build_word_vocab(train_sents)
    char2idx = build_char_vocab(train_sents)
    tag2idx = build_tag_vocab(train_sents)
    idx2tag = {v: k for k, v in tag2idx.items()}
    glove_matrix = build_glove_matrix(word2idx, glove, embed_dim=EMBED_DIM)

    train_ds = NERDataset(train_sents, word2idx, tag2idx, char2idx)
    dev_ds = NERDataset(dev_sents, word2idx, tag2idx, char2idx)
    test_ds = NERDataset(test_sents, word2idx, tag2idx, char2idx)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, generator=g)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    _ = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = AdvancedBLSTM(
        glove_matrix=glove_matrix,
        num_tags=len(tag2idx),
        num_case_cats=5,
        case_embed_dim=CASE_EMBED_DIM,
        char_vocab_size=len(char2idx),
        char_emb_dim=CHAR_EMBED_DIM,
        char_out_channels=CHAR_CNN_CHANNELS,
        char_kernels=CHAR_KERNELS,
        hidden_dim=LSTM_HIDDEN,
        linear_out=LINEAR_OUT_DIM,
        num_layers=NUM_LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
        use_char=True,
    ).to(DEVICE)

    other_params = [p for n, p in model.named_parameters() if not n.startswith("word_embed")]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.word_embed.parameters(), "lr": LR_WORD},
            {"params": other_params, "lr": LR_OTHER},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_f1 = -1.0
    bad_epochs = 0
    best_path = os.path.join(output_dir, "blstm2.pt")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch_advanced(model, train_loader, optimizer, criterion, grad_clip=GRAD_CLIP)
        p, r, f1 = evaluate_advanced(model, dev_loader, idx2tag, repair_output=True)
        scheduler.step(f1)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | P: {p:.4f} R: {r:.4f} F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "word2idx": word2idx,
                    "char2idx": char2idx,
                    "tag2idx": tag2idx,
                    "config": {
                        "embed_dim": EMBED_DIM,
                        "case_embed_dim": CASE_EMBED_DIM,
                        "char_embed_dim": CHAR_EMBED_DIM,
                        "char_cnn_channels": CHAR_CNN_CHANNELS,
                        "char_kernels": CHAR_KERNELS,
                        "hidden_dim": LSTM_HIDDEN,
                        "linear_out_dim": LINEAR_OUT_DIM,
                        "num_lstm_layers": NUM_LSTM_LAYERS,
                        "dropout": LSTM_DROPOUT,
                        "use_char_cnn": True,
                    },
                },
                best_path,
            )
            print(f"  saved best model (F1={best_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping triggered")
                break

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    p, r, f1 = evaluate_advanced(model, dev_loader, idx2tag, repair_output=True)
    print(f"Best Dev | P: {p:.4f} R: {r:.4f} F1: {f1:.4f}")

    predict_advanced(model, dev_sents, word2idx, char2idx, idx2tag, os.path.join(output_dir, "dev2.out"), repair_output=True)
    predict_advanced(model, test_sents, word2idx, char2idx, idx2tag, os.path.join(output_dir, "test2.out"), repair_output=True)


def run_bonus(train_sents, dev_sents, test_sents, glove_path: str, output_dir: str):
    print("\n=== Bonus: BLSTM + CNN (Character-level) with GloVe ===")
    EMBED_DIM = 100
    CASE_EMBED_DIM = 25
    CHAR_EMBED_DIM = 30
    CHAR_CNN_CHANNELS = 50
    CHAR_KERNELS = (2, 3, 4)

    NUM_LSTM_LAYERS = 1
    LSTM_HIDDEN = 256
    LSTM_DROPOUT = 0.33
    LINEAR_OUT_DIM = 128

    EPOCHS = 40
    BATCH_SIZE = 32
    LR_WORD = 0.005
    LR_OTHER = 0.01
    LR_DECAY = 0.5
    LR_PATIENCE = 3
    GRAD_CLIP = 5.0

    glove = load_glove(glove_path, dim=EMBED_DIM)
    word2idx = build_word_vocab(train_sents)
    char2idx = build_char_vocab(train_sents)
    tag2idx = build_tag_vocab(train_sents)
    idx2tag = {v: k for k, v in tag2idx.items()}
    glove_matrix = build_glove_matrix(word2idx, glove, embed_dim=EMBED_DIM)

    train_ds = NERDataset(train_sents, word2idx, tag2idx, char2idx)
    dev_ds = NERDataset(dev_sents, word2idx, tag2idx, char2idx)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, generator=g)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = AdvancedBLSTM(
        glove_matrix=glove_matrix,
        num_tags=len(tag2idx),
        num_case_cats=5,
        case_embed_dim=CASE_EMBED_DIM,
        char_vocab_size=len(char2idx),
        char_emb_dim=CHAR_EMBED_DIM,
        char_out_channels=CHAR_CNN_CHANNELS,
        char_kernels=CHAR_KERNELS,
        hidden_dim=LSTM_HIDDEN,
        linear_out=LINEAR_OUT_DIM,
        num_layers=NUM_LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
        use_char=True,
    ).to(DEVICE)

    optimizer = torch.optim.SGD(
        [
            {"params": model.word_embed.parameters(), "lr": LR_WORD},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("word_embed")], "lr": LR_OTHER},
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=LR_DECAY, patience=LR_PATIENCE)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_f1 = 0.0
    best_path = os.path.join(output_dir, "blstm_cnn.pt")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch_advanced(model, train_loader, optimizer, criterion, grad_clip=GRAD_CLIP)
        p, r, f1 = evaluate_advanced(model, dev_loader, idx2tag, repair_output=True)
        scheduler.step(f1)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | P: {p:.4f} R: {r:.4f} F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "word2idx": word2idx,
                    "char2idx": char2idx,
                    "tag2idx": tag2idx,
                    "config": {
                        "embed_dim": EMBED_DIM,
                        "case_embed_dim": CASE_EMBED_DIM,
                        "char_embed_dim": CHAR_EMBED_DIM,
                        "char_cnn_channels": CHAR_CNN_CHANNELS,
                        "char_kernels": CHAR_KERNELS,
                        "hidden_dim": LSTM_HIDDEN,
                        "linear_out_dim": LINEAR_OUT_DIM,
                        "num_lstm_layers": NUM_LSTM_LAYERS,
                        "dropout": LSTM_DROPOUT,
                        "use_char_cnn": True,
                    },
                },
                best_path,
            )
            print(f"  saved best model (F1={best_f1:.4f})")

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    p, r, f1 = evaluate_advanced(model, dev_loader, idx2tag, repair_output=True)
    print(f"Best Dev | P: {p:.4f} R: {r:.4f} F1: {f1:.4f}")

    predict_advanced(model, dev_sents, word2idx, char2idx, idx2tag, os.path.join(output_dir, "dev_bonus.out"), repair_output=True)
    predict_advanced(model, test_sents, word2idx, char2idx, idx2tag, os.path.join(output_dir, "pred"), repair_output=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CSCI544 HW3 combined NER training script")
    p.add_argument("--mode", choices=["task1", "task2", "bonus", "all"], default="all",
                   help="Which model(s) to train")
    p.add_argument("--data-dir", type=str, default="data", help="Directory containing train/dev/test")
    p.add_argument("--glove-path", type=str, default="glove.6B.100d.gz", help="Path to GloVe file")
    p.add_argument("--output-dir", type=str, default=".", help="Where to save models and predictions")
    return p.parse_args()


def main():
    set_seed(SEED)
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train")
    dev_path = os.path.join(args.data_dir, "dev")
    test_path = os.path.join(args.data_dir, "test")

    train_sents = load_data(train_path)
    dev_sents = load_data(dev_path)
    test_sents = load_data(test_path)

    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_sents)} | Dev: {len(dev_sents)} | Test: {len(test_sents)}")

    if args.mode in ("task1", "all"):
        run_task1(train_sents, dev_sents, test_sents, args.data_dir, args.output_dir)
    if args.mode in ("task2", "all"):
        run_task2(train_sents, dev_sents, test_sents, args.glove_path, args.output_dir)
    if args.mode in ("bonus", "all"):
        run_bonus(train_sents, dev_sents, test_sents, args.glove_path, args.output_dir)


if __name__ == "__main__":
    main()
