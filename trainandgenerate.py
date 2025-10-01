import os
import re
import math
import datetime
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import MHA

DATA_PATH = "tiny_shakespeare.txt"

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
INPUT_MAX_SEQ_LEN = 128
OUTPUT_MAX_SEQ_LEN = 128
MAX_GEN_LEN = 100
MIN_FREQ = 1
MASK_VAL = float("-inf")
MAX_SEQ_LEN = max(INPUT_MAX_SEQ_LEN, OUTPUT_MAX_SEQ_LEN) + 2
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<bos>", "<eos>"

DEVICE = torch.device("cuda")
log_out = "log_Juliet.txt"
def tokenize(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text)
    return [word.lower() for word in words]


def build_vocab(texts: list[str], min_freq: int = 1) -> tuple[dict, dict]:
    frequency: dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            frequency[tok] = frequency.get(tok, 0) + 1
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    for tok in sorted(frequency):
        if frequency[tok] >= min_freq:
            vocab.setdefault(tok, len(vocab))
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab


def encode(tokens: list[str], vocab: dict) -> list[int]:
    unk = vocab[UNK_TOKEN]
    return [vocab.get(t, unk) for t in tokens]


class TextDataset(Dataset):
    def __init__(self, text: str, vocab: dict, seq_len: int):
        self.seq_len = seq_len
        tokens = tokenize(text)
        self.token_ids = encode(tokens, vocab)

        self.sequences = []
        for i in range(0, len(self.token_ids) - seq_len, seq_len // 2):
            if i + seq_len < len(self.token_ids):
                self.sequences.append(
                    self.token_ids[i : i + seq_len + 1]
                )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]


def collate(batch, pad_id):
    inputs, targets = zip(*batch)
    seq_len = max(len(seq) for seq in inputs)

    input_batch = torch.full((seq_len, len(batch)), pad_id, dtype=torch.long)
    target_batch = torch.full((seq_len, len(batch)), pad_id, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        input_batch[: len(inp), i] = torch.tensor(inp)
        target_batch[: len(tgt), i] = torch.tensor(tgt)

    attn_mask = torch.triu(torch.full((seq_len, seq_len), MASK_VAL), diagonal=1)
    attn_mask.fill_diagonal_(0.0)

    key_pad_mask = (input_batch == pad_id).T  # (B, T)

    return input_batch, target_batch, attn_mask.float(), key_pad_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pe[: x.size(0)])


class CustomDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)

        norm_x_ff = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(norm_x_ff))))
        x = x + self.dropout2(ff_output)
        return x


class TransformerModel(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.d_model = D_MODEL
        self.embed = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=D_MODEL, padding_idx=0
        )
        self.position_encode = PositionalEncoding(
            d_model=D_MODEL, dropout=DROPOUT, max_len=MAX_GEN_LEN + 50
        )

        self.decoder_layers = nn.ModuleList(
            [
                CustomDecoderLayer(
                    d_model=D_MODEL,
                    nhead=NHEAD,
                    dim_feedforward=DIM_FEEDFORWARD,
                    dropout=DROPOUT,
                )
                for _ in range(NUM_LAYERS)
            ]
        )
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.projection = nn.Linear(in_features=D_MODEL, out_features=len(vocab))

    def forward(self, x, attn_mask, key_pad_mask):
        x = self.position_encode(self.embed(x) * math.sqrt(self.d_model))

        for layer in self.decoder_layers:
            x = layer(
                src=x,
                src_mask=attn_mask.to(x.device),
                src_key_padding_mask=key_pad_mask.to(x.device),
            )

        out = self.final_norm(x)
        return self.projection(out)


def train_epoch(model, loader, optimizer_, loss_criterion_, pad_id):
    model.train()
    tot_loss = tot_batches = 0
    tot_tok = tot_correct = 0
    for inputs, targets, attn_mask, key_pad_mask in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        attn_mask, key_pad_mask = attn_mask.to(DEVICE), key_pad_mask.to(DEVICE)

        optimizer_.zero_grad()
        logits = model(inputs, attn_mask, key_pad_mask)
        loss = loss_criterion_(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()

        with torch.no_grad():
            pred = logits.argmax(-1)
            mask = targets.ne(pad_id)
            tot_correct += (pred.eq(targets) & mask).sum().item()
            tot_tok += mask.sum().item()

        tot_loss += loss.item()
        tot_batches += 1
    return tot_loss / tot_batches, tot_correct / tot_tok


@torch.no_grad()
def eval_epoch(model, loader, loss_criterion_, pad_id):
    model.eval()
    tot_loss = tot_batches = 0
    tot_tok = tot_correct = 0
    for inputs, targets, attn_mask, key_pad_mask in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        attn_mask, key_pad_mask = attn_mask.to(DEVICE), key_pad_mask.to(DEVICE)

        logits = model(inputs, attn_mask, key_pad_mask)
        loss = loss_criterion_(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        pred = logits.argmax(-1)
        mask = targets.ne(pad_id)
        tot_correct += (pred.eq(targets) & mask).sum().item()
        tot_tok += mask.sum().item()

        tot_loss += loss.item()
        tot_batches += 1
    return tot_loss / tot_batches, tot_correct / tot_tok


def infer(model, prompt, vocab, inv_vocab, max_len=100, temperature=0.8, top_k=20):
    model.eval()
    tokens = tokenize(prompt)
    token_ids = encode(tokens, vocab)

    if not token_ids or token_ids[0] != vocab[BOS_TOKEN]:
        token_ids = [vocab[BOS_TOKEN]] + token_ids

    for _ in range(max_len):
        x = torch.tensor(token_ids, device=DEVICE).unsqueeze(1)  # (T, 1)
        seq_len = x.size(0)

        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), MASK_VAL, device=DEVICE), diagonal=1
        )
        attn_mask.fill_diagonal_(0.0)
        key_pad_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=DEVICE)

        with torch.no_grad():
            logits = model(x, attn_mask, key_pad_mask)
            logits = logits[-1, 0] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("Inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        if next_token_id == vocab[EOS_TOKEN]:
            break

        token_ids.append(next_token_id)

    words = [inv_vocab.get(i, UNK_TOKEN) for i in token_ids[1:]]  # Skip BOS
    return " ".join(words)


if __name__ == "__main__":
    start_time = time.time()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text_data = f.read()
    print(f"Loaded text with {len(text_data):,} characters")

    vocab, inv_vocab = build_vocab([text_data], MIN_FREQ)
    print(f"Vocab size: {len(vocab):,}")
    PAD_ID = vocab[PAD_TOKEN]

    full_ds = TextDataset(text_data, vocab, INPUT_MAX_SEQ_LEN)
    TRAIN_SZ = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(
        full_ds,
        [TRAIN_SZ, len(full_ds) - TRAIN_SZ],
        generator=torch.Generator().manual_seed(42),
    )
    collate_func = lambda b: collate(b, pad_id=vocab[PAD_TOKEN])
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_func)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_func)

    MODEL = TransformerModel(vocab).to(DEVICE)
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOKEN])

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_perplexities, val_perplexities = [], []
    epochs_range = range(1, EPOCHS + 1)
    for ep in epochs_range:
        epoch_start_time = time.time()
        tr_loss, tr_acc = train_epoch(
            MODEL, train_dl, optimizer, loss_criterion, PAD_ID
        )
        vl_loss, vl_acc = eval_epoch(MODEL, val_dl, loss_criterion, PAD_ID)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        tr_ppl = math.exp(tr_loss)
        vl_ppl = math.exp(vl_loss)
        train_perplexities.append(tr_ppl)
        val_perplexities.append(vl_ppl)

        epoch_end_time = time.time()
        epoch_minutes, epoch_seconds = divmod(
            int(epoch_end_time - epoch_start_time), 60
        )
        print(
            f"Epoch {ep:02d}/{EPOCHS} │ "
            f"train_loss={tr_loss:.3f} acc={tr_acc:.2%} ppl={tr_ppl:.1f} │ "
            f"val_loss={vl_loss:.3f} acc={vl_acc:.2%} ppl={vl_ppl:.1f} │ "
            f"Time: {epoch_minutes}m {epoch_seconds}s"
        )
        with open(log_out, "w", encoding="utf-8") as f:
            f.write("Epochs Range: " + str(epochs_range) + '\n' + f"Training Accuracy:{tr_acc:.2%}" + '\n' + f"Validataion Accuracy: {vl_acc:.2%}\n\n")

    print("\n")
    MODELS_DIR = "models"
    LOGGING_DIR = "logging"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    script_name = os.path.basename(__file__)
    filename_base = os.path.splitext(script_name)[0]

    model_save_path = os.path.join(MODELS_DIR, f"{filename_base}_{ts}.pth")
    latest_model_path = os.path.join(MODELS_DIR, f"{filename_base}_latest.pth")
    save_dict = {
        "model_state": MODEL.state_dict(),
        "vocab": vocab,
    }
    torch.save(save_dict, model_save_path)
    torch.save(save_dict, latest_model_path)
    print(f"Model saved to {model_save_path} and {latest_model_path}")

    total_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(total_seconds, 60)
    print(f"\nTotal runtime: {minutes}m {seconds}s")

    DEMO = "JULIET:"
    print("\nPROMPT:", DEMO)
    print(
        "GENERATED:",
        infer(MODEL, DEMO, vocab, inv_vocab, max_len=50, top_k=10, temperature=0.8),
    )
    generated_out = "generated_Juliet.txt"
    with open(generated_out, "w", encoding="utf-8") as f:
        f.write("Generated Output:" + '\n' + DEMO + infer(MODEL, DEMO, vocab, inv_vocab, max_len=50, top_k=10, temperature=0.8))