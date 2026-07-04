"""
Leave-One-Program-Out (LOPO) Node Classification
=================================================
Pipeline:
  1. Pretrain InductiveTransfer روی InductiveDataset
  2. برای هر fold (11 fold):
       - train_programs  = همه برنامه‌ها به جز یکی
       - test_program    = برنامه کنار گذاشته شده
       - NodeClassifier جدید با وزن‌های pretrain شده MGATConv
       - train روی train_programs (همه node ها)
       - evaluate روی test_program (همه node های valid)
  3. جمع‌بندی نتایج F1/P/R به تفکیک کلاس برای همه fold ها
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch_geometric.utils import to_undirected
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.dense.linear import Linear

from sklearn.metrics import f1_score, precision_score, recall_score

from MGAT2 import MGATConv          # همان import کد اصلی
from modify_data import TrainDataset, custom_collate

# ─────────────────────────── تنظیمات ───────────────────────────
SEED            = 100
NUM_GAT_LAYERS  = 1
NUM_LABELS      = 3
PRETRAIN_EPOCHS = 3000
LOPO_EPOCHS     = 3001
LR              = 1e-3
THRESHOLD       = 0.5
RESULTS_CSV     = "results_lopo.csv"

torch.manual_seed(SEED)

# ─────────────────────────── مدل‌ها ─────────────────────────────
FEAT_DIM = 39 + 16 + 24   # x=39, lpe=16, xx=24


class InductiveTransfer(nn.Module):
    """همان معماری کد اصلی — فقط برای pretrain روی InductiveDataset"""
    def __init__(self):
        super().__init__()
        self.conv11 = GCNConv(39, 39, add_self_loops=False)
        self.conv12 = GCNConv(39, 39, add_self_loops=False)
        self.conv13 = GCNConv(39, 39, add_self_loops=False)
        self.conv14 = GCNConv(39, 39, add_self_loops=False)
        self.conv15 = GCNConv(39, 39, add_self_loops=True)
        self.conv21 = MGATConv(FEAT_DIM, FEAT_DIM, heads=NUM_GAT_LAYERS)
        self.linear31 = Linear(FEAT_DIM * NUM_GAT_LAYERS, NUM_LABELS)

    def forward(self, x, xx, lpe, edge_index):
        res = x
        x = self.conv11(x, edge_index).relu()
        x = self.conv12(x, edge_index).relu()
        x = self.conv13(x, edge_index).relu()
        x = self.conv14(x, edge_index).relu()
        x = F.dropout(x, p=0.05, training=self.training)
        x = res + x
        x = self.conv15(x, edge_index).relu()
        x = torch.cat([x, xx, lpe], dim=1)
        x = self.conv21(x=x, edge_index=edge_index)
        return self.linear31(x)


class NodeClassifier(nn.Module):
    """همان معماری کد اصلی — برای LOPO fine-tune"""
    def __init__(self):
        super().__init__()
        self.conv11 = GCNConv(39, 39, add_self_loops=False)
        self.conv12 = GCNConv(39, 39, add_self_loops=False)
        self.conv13 = GCNConv(39, 39, add_self_loops=False)
        self.conv14 = GCNConv(39, 39, add_self_loops=False)
        self.conv15 = GCNConv(39, 39, add_self_loops=True)
        self.conv21 = MGATConv(FEAT_DIM, FEAT_DIM, heads=NUM_GAT_LAYERS)
        self.linear31 = Linear(FEAT_DIM * NUM_GAT_LAYERS, NUM_LABELS)

    def forward(self, x, xx, lpe, edge_index, edge_weight=None):
        res = x
        x = self.conv11(x, edge_index).relu()
        x = self.conv12(x, edge_index).relu()
        x = self.conv13(x, edge_index).relu()
        x = self.conv14(x, edge_index).relu()
        x = F.dropout(x, p=0.05, training=self.training)
        x = res + x
        x = self.conv15(x, edge_index).relu()
        x = torch.cat([x, xx, lpe], dim=1)
        x = self.conv21(x=x, edge_index=edge_index)
        return self.linear31(x)


# ─────────────────────────── توابع کمکی ─────────────────────────

def prepare_data(data):
    """اضافه کردن y، edge_index undirected و valid_mask به هر data object"""
    data.y = torch.stack([
        (data.nodesmask == 1),
        (data.nodessdc > 0),
        (data.nodesint > 0),
    ], dim=1).float()
    data.edge_index = to_undirected(data.edge_index)
    return data


def get_class_weights(data, mask=None):
    """
    وزن‌های dynamic برای BCEWithLogitsLoss
    mask: اگر None باشد از valid_mask استفاده می‌شود
    """
    m = mask if mask is not None else data.valid_mask
    y = data.y[m]
    all_n = y.shape[0] + 1
    w = []
    for i in range(NUM_LABELS):
        cnt = y[:, i].sum().item() + 1
        w.append(all_n / cnt)
    return torch.tensor(w, dtype=torch.float)


def eval_metrics(out, y, mask):
    """F1/Precision/Recall به تفکیک کلاس روی mask داده شده"""
    with torch.no_grad():
        probs = torch.sigmoid(out[mask])
        preds = (probs > THRESHOLD).int().cpu().numpy()
        labels = y[mask].int().cpu().numpy()

    f1  = f1_score(labels,  preds, zero_division=0, average=None)
    p   = precision_score(labels, preds, zero_division=0, average=None)
    r   = recall_score(labels,   preds, zero_division=0, average=None)

    # اطمینان از اینکه همیشه ۳ مقدار داریم
    def pad(arr):
        arr = list(arr)
        while len(arr) < NUM_LABELS:
            arr.append(0.0)
        return arr[:NUM_LABELS]

    return pad(f1), pad(p), pad(r)


def save_lopo_result(fold_name, f1, p, r, epoch, filename=RESULTS_CSV):
    class_names = ["mask", "sdc", "int"]
    rows = []
    for i, cls in enumerate(class_names):
        rows.append({
            "fold":      fold_name,
            "class":     cls,
            "f1":        round(f1[i], 4),
            "precision": round(p[i],  4),
            "recall":    round(r[i],  4),
            "best_epoch": epoch,
        })
    df_new = pd.DataFrame(rows)
    if not os.path.exists(filename):
        df_new.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(filename, index=False)


def print_fold_summary(fold_name, f1, p, r, epoch):
    class_names = ["mask", "sdc", "int"]
    print(f"\n{'─'*50}")
    print(f"  Test Fold: {fold_name}  |  Best Epoch: {epoch}")
    print(f"  {'Class':<8} {'F1':>8} {'Precision':>12} {'Recall':>10}")
    for i, cls in enumerate(class_names):
        print(f"  {cls:<8} {f1[i]:>8.4f} {p[i]:>12.4f} {r[i]:>10.4f}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════════════
#  مرحله ۱: بارگذاری داده‌ها
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  بارگذاری داده‌ها")
print("═"*60)

TransductiveDataset = TrainDataset(
    directory="data_i/train_data_onehot3/",
    transformer=None, save_location="NO", PCA="NO"
)
InductiveDataset = TrainDataset(
    directory="data_i/inductive_unbiased/",
    transformer=None, save_location="NO", PCA="NO"
)

# آماده‌سازی همه داده‌ها
transductive_list = []
for data in TransductiveDataset:
    data = prepare_data(data)
    transductive_list.append(data)

inductive_list = []
for data in InductiveDataset:
    data = prepare_data(data)
    inductive_list.append(data)

print(f"  برنامه‌های Transductive : {len(transductive_list)}")
print(f"  برنامه‌های Inductive    : {len(inductive_list)}")


# ══════════════════════════════════════════════════════════════════
#  مرحله ۲: Pretrain روی InductiveDataset
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  Pretrain روی InductiveDataset")
print("═"*60)

IDL = DataLoader(InductiveDataset, batch_size=30, collate_fn=custom_collate)

pretrain_loss_fn = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0, 3.0, 43.0])
)
inductive_model = InductiveTransfer()
pretrain_optimizer = torch.optim.Adam(inductive_model.parameters(), lr=LR)

inductive_model.train()
for epoch in range(PRETRAIN_EPOCHS):
    for batch in IDL:
        pretrain_optimizer.zero_grad()
        out = inductive_model(batch.x, batch.xx, batch.lpe, batch.edge_index)
        loss = pretrain_loss_fn(out, batch.y)
        loss.backward()
        pretrain_optimizer.step()

    if epoch % 500 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(out)
            preds = (probs > THRESHOLD).int()
            f1_pre = f1_score(batch.y.numpy(), preds.numpy(),
                              zero_division=0, average='macro')
        print(f"  Epoch {epoch:4d} | loss: {loss.item():.4f} | F1 macro: {f1_pre:.4f}")

# وزن‌های pretrain شده MGATConv ذخیره می‌شوند
pretrained_mgat_state = copy.deepcopy(inductive_model.conv21.state_dict())
print("\n  Pretrain تمام شد — وزن‌های MGATConv ذخیره شد.")


# ══════════════════════════════════════════════════════════════════
#  مرحله ۳: LOPO روی TransductiveDataset
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  LOPO Training — 11 Fold")
print("═"*60)

n_programs = len(transductive_list)
all_results = []   # برای خلاصه نهایی

for fold_idx in range(n_programs):

    test_data   = transductive_list[fold_idx]
    train_data_list = [transductive_list[i]
                       for i in range(n_programs) if i != fold_idx]

    print(f"\n  ┌─ Fold {fold_idx+1}/{n_programs}"
          f"  |  Test: {test_data.name}")
    print(f"  │  Train programs: "
          f"{[d.name for d in train_data_list]}")

    # ── مدل جدید با وزن pretrain شده ──
    model = NodeClassifier()
    model.conv21.load_state_dict(pretrained_mgat_state)

    # ── وزن loss روی مجموع برنامه‌های train ──
    # وزن کلی از union همه برنامه‌های train
    total_pos = torch.zeros(NUM_LABELS)
    total_all = 0
    for d in train_data_list:
        v = d.valid_mask
        total_pos += d.y[v].sum(dim=0)
        total_all += v.sum().item()
    alpha_w = (total_all + 1) / (total_pos + 1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=alpha_w)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── متغیرهای ردیابی بهترین نتیجه ──
    best_f1_sum  = -1.0   # مجموع F1 سه کلاس برای انتخاب بهترین epoch
    best_f1      = [0.0] * NUM_LABELS
    best_p       = [0.0] * NUM_LABELS
    best_r       = [0.0] * NUM_LABELS
    best_epoch   = 0

    start = time.time()
    model.train()

    for epoch in range(LOPO_EPOCHS):
        optimizer.zero_grad()

        # ── train روی همه برنامه‌های train (graph-by-graph) ──
        total_loss = torch.tensor(0.0)
        for d in train_data_list:
            out = model(d.x, d.xx, d.lpe, d.edge_index, d.edge_weight)
            loss = criterion(out[d.valid_mask], d.y[d.valid_mask])
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()

        # ── ارزیابی روی test_program ──
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                out_test = model(
                    test_data.x, test_data.xx, test_data.lpe,
                    test_data.edge_index, test_data.edge_weight
                )
            f1, p, r = eval_metrics(out_test, test_data.y, test_data.valid_mask)

            f1_sum = sum(f1)
            if f1_sum > best_f1_sum:
                best_f1_sum = f1_sum
                best_f1     = f1
                best_p      = p
                best_r      = r
                best_epoch  = epoch
                # ذخیره بهترین مدل این fold
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(),
                           f"models/lopo_fold{fold_idx+1}_{test_data.name}.pt")

            if epoch % 500 == 0:
                print(f"  │  epoch {epoch:4d} | loss {total_loss.item():.4f}"
                      f" | F1[mask={f1[0]:.3f} sdc={f1[1]:.3f} int={f1[2]:.3f}]"
                      f" | best@{best_epoch}")
            model.train()

    elapsed = time.time() - start
    print(f"  └─ زمان: {elapsed:.1f}s")

    print_fold_summary(test_data.name, best_f1, best_p, best_r, best_epoch)
    save_lopo_result(test_data.name, best_f1, best_p, best_r, best_epoch)

    all_results.append({
        "fold":   test_data.name,
        "f1":     best_f1,
        "p":      best_p,
        "r":      best_r,
        "epoch":  best_epoch,
    })


# ══════════════════════════════════════════════════════════════════
#  مرحله ۴: خلاصه نهایی
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  خلاصه نتایج LOPO (میانگین روی ۱۱ fold)")
print("═"*60)

class_names = ["mask", "sdc", "int"]
agg_f1 = np.zeros(NUM_LABELS)
agg_p  = np.zeros(NUM_LABELS)
agg_r  = np.zeros(NUM_LABELS)

for res in all_results:
    agg_f1 += np.array(res["f1"])
    agg_p  += np.array(res["p"])
    agg_r  += np.array(res["r"])

n = len(all_results)
print(f"\n  {'Class':<8} {'F1 mean':>10} {'P mean':>10} {'R mean':>10}")
for i, cls in enumerate(class_names):
    print(f"  {cls:<8} {agg_f1[i]/n:>10.4f} {agg_p[i]/n:>10.4f} {agg_r[i]/n:>10.4f}")

print(f"\n  نتایج در {RESULTS_CSV} ذخیره شدند.")
print("═"*60)