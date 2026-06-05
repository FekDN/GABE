#!/usr/bin/env python3
# GABEtest_new_class_v2.py — GABE New-Class Learning Test
#
# SETUP:  pip install torch torchvision pillow
# Файлы:  tree1.jpg, tree2.jpg, tree3.jpg, tree4.jpg  — рядом со скриптом
#
# ИСПРАВЛЕНИЕ v1 → v2
# ──────────────────────────────────────────────────────────────────────────────
# v1 ОШИБКА: _inject_weights() использовал param.data.copy_()
#   → обрывал граф автодифференцирования
#   → RuntimeError: element 0 of tensors does not require grad
#
# v2 РЕШЕНИЕ: явный forward через F.conv2d с реконструированными тензорами
#   W_i = W̄ + Σⱼ αᵢⱼ Bⱼ  вычисляется как обычный torch-тензор
#   F.conv2d(x, W_i, ...)  — градиенты текут напрямую через W̄ и α
#
# ИЗМЕНЕНИЯ:
#   - N_EPOCHS: 30 → 8  (ResNet18 сходится быстро с предобученными весами)
#   - GABEResNet заменён на GABEResNetV2 с явным послойным forward
#   - Убран _inject_weights()
# ──────────────────────────────────────────────────────────────────────────────

import copy, os, sys, time, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_EPOCHS   = 8
LR         = 1e-3
LR_FULL    = 1e-4
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Группы слоёв ResNet18 одинаковой размерности ──────────────────────────────
# Ключи совпадают с именами в state_dict ResNet18.
# Каждая группа = список весовых матриц одинакового shape → валидный стек для GABE.
#
# Структура свёрточных слоёв ResNet18:
#   layer1: 4 conv [64,64,3,3]    → K = L-1 = 3, точное разложение
#   layer2: 3 conv [128,128,3,3]  → K = 2
#   layer3: 3 conv [256,256,3,3]  → K = 2
#   layer4: 3 conv [512,512,3,3]  → K = 2
#
# Первые свёртки каждого layer (stride=2, разные C_in/C_out) исключены —
# у них другой shape, не подходят под группу.

GROUPS = {
    "l1": ["layer1.0.conv1.weight", "layer1.0.conv2.weight",
            "layer1.1.conv1.weight", "layer1.1.conv2.weight"],
    "l2": ["layer2.0.conv2.weight", "layer2.1.conv1.weight", "layer2.1.conv2.weight"],
    "l3": ["layer3.0.conv2.weight", "layer3.1.conv1.weight", "layer3.1.conv2.weight"],
    "l4": ["layer4.0.conv2.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight"],
}
GROUP_ORDER = ["l1", "l2", "l3", "l4"]

# ── Stride/padding для каждого conv (нужен для F.conv2d) ─────────────────────
# ResNet18: все 3×3 conv используют stride=1, padding=1
CONV_KWARGS = {name: dict(stride=1, padding=1)
               for group in GROUPS.values() for name in group}


# ══════════════════════════════════════════════════════════════════════════════
# GABE — точное разложение при k = L-1
# ══════════════════════════════════════════════════════════════════════════════

def collect_stack(model, names):
    """(L, D) float32: строки = flattened веса из names."""
    sd = {n: p.detach().float() for n, p in model.named_parameters()}
    return torch.stack([sd[n].flatten() for n in names])

def extract_gabe(W):
    """W: (L,D) → W_bar(D,), B(K,D), alpha(L,K), recon_err, K, L, D, orig_shape."""
    L = W.shape[0]; K = L - 1
    W_bar = W.mean(0)
    delta = W - W_bar
    _, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    B     = Vh[:K].clone()
    alpha = delta @ B.T
    err   = (W - (W_bar + alpha @ B)).norm() / (W.norm() + 1e-10)
    return dict(W_bar=W_bar, B=B, alpha=alpha, S=S,
                recon_err=err.item(), K=K, L=L, D=W.shape[1])

def extract_all(model):
    return {g: extract_gabe(collect_stack(model, names))
            for g, names in GROUPS.items()}


# ══════════════════════════════════════════════════════════════════════════════
# GABE_FT — явный forward без инъекции через .data
#
# Ключевая идея: вместо того чтобы записывать реконструированные веса
# в параметры backbone и вызывать стандартный forward,
# мы вызываем F.conv2d напрямую с тензором W_i = W̄ + Σⱼ αᵢⱼ Bⱼ.
# Этот тензор является листом графа вычислений → градиенты текут корректно.
# ══════════════════════════════════════════════════════════════════════════════

class GABEGroup(nn.Module):
    """W̄ и α обучаемы; B — замороженный буфер."""
    def __init__(self, gd, names, shapes):
        super().__init__()
        self.names  = names
        self.shapes = shapes          # оригинальные формы весовых тензоров
        self.L = gd["L"]; self.K = gd["K"]; self.D = gd["D"]
        self.W_bar = nn.Parameter(gd["W_bar"].clone())    # (D,)
        self.alpha = nn.Parameter(gd["alpha"].clone())    # (L, K)
        self.register_buffer("B", gd["B"].clone())        # (K, D) — frozen

    def weights(self):
        """Возвращает список из L тензоров правильных форм."""
        stack = self.W_bar + self.alpha @ self.B           # (L, D)
        return [stack[i].view(self.shapes[i]) for i in range(self.L)]


class GABEResNet(nn.Module):
    """
    ResNet18 с явным послойным forward.
    GABE-группы: W̄ + α обучаемы, B заморожен.
    Остальные слои (BN, первые conv с другим shape, fc) — обычные nn.Module.
    """
    def __init__(self, base, gabe_pre, n_classes=2):
        super().__init__()
        net = copy.deepcopy(base)

        # ── Первый блок (conv1 + bn1 + relu + maxpool) ────────────────────────
        self.conv1    = net.conv1      # [64, 3, 7, 7] — не входит в GABE-группы
        self.bn1      = net.bn1
        self.relu     = net.relu
        self.maxpool  = net.maxpool

        # ── Хранилище BN для каждого блока ────────────────────────────────────
        # BN-слои оставляем как nn.Module (они обучаются отдельно)
        self.bn_l1_0_1 = net.layer1[0].bn1
        self.bn_l1_0_2 = net.layer1[0].bn2
        self.bn_l1_1_1 = net.layer1[1].bn1
        self.bn_l1_1_2 = net.layer1[1].bn2

        self.bn_l2_0_2 = net.layer2[0].bn2
        self.bn_l2_1_1 = net.layer2[1].bn1
        self.bn_l2_1_2 = net.layer2[1].bn2

        self.bn_l3_0_2 = net.layer3[0].bn2
        self.bn_l3_1_1 = net.layer3[1].bn1
        self.bn_l3_1_2 = net.layer3[1].bn2

        self.bn_l4_0_2 = net.layer4[0].bn2
        self.bn_l4_1_1 = net.layer4[1].bn1
        self.bn_l4_1_2 = net.layer4[1].bn2

        # ── Transition conv (stride=2, меняют spatial size) ───────────────────
        # Эти conv имеют другой shape → не входят в GABE-группы, хранятся отдельно
        self.conv_l1_0_1 = net.layer1[0].conv1   # [64,64,3,3]  — входит в l1 группу
        # layer2.0.conv1: [128,64,3,3]  stride=2 — НЕ в группе
        self.conv_l2_0_1  = net.layer2[0].conv1
        self.bn_l2_0_1    = net.layer2[0].bn1
        self.ds2_conv      = net.layer2[0].downsample[0]
        self.ds2_bn        = net.layer2[0].downsample[1]
        # layer3.0.conv1: [256,128,3,3] stride=2
        self.conv_l3_0_1  = net.layer3[0].conv1
        self.bn_l3_0_1    = net.layer3[0].bn1
        self.ds3_conv      = net.layer3[0].downsample[0]
        self.ds3_bn        = net.layer3[0].downsample[1]
        # layer4.0.conv1: [512,256,3,3] stride=2
        self.conv_l4_0_1  = net.layer4[0].conv1
        self.bn_l4_0_1    = net.layer4[0].bn1
        self.ds4_conv      = net.layer4[0].downsample[0]
        self.ds4_bn        = net.layer4[0].downsample[1]

        # ── Pooling + classifier ───────────────────────────────────────────────
        self.avgpool = net.avgpool
        self.fc      = nn.Linear(net.fc.in_features, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # ── GABE groups ────────────────────────────────────────────────────────
        sd = dict(net.named_parameters())
        groups_dict = {}
        for g, names in GROUPS.items():
            shapes = [sd[n].shape for n in names]
            groups_dict[g] = GABEGroup(gabe_pre[g], names, shapes)
        self.gabe = nn.ModuleDict(groups_dict)

        # Замораживаем все не-GABE параметры backbone кроме BN и fc
        for name, p in self.named_parameters():
            is_gabe = name.startswith("gabe.")
            is_bn   = any(bn in name for bn in ["bn", "downsample.1"])
            is_fc   = name.startswith("fc.")
            is_tconv = any(n in name for n in [
                "conv_l2_0_1", "conv_l3_0_1", "conv_l4_0_1",
                "ds2_conv", "ds3_conv", "ds4_conv"])
            p.requires_grad_(is_gabe or is_bn or is_fc)

    # ── Вспомогательные блоки ─────────────────────────────────────────────────

    def _conv(self, x, w, stride=1, padding=1):
        return F.conv2d(x, w, bias=None, stride=stride, padding=padding)

    def _residual_block_gabe(self, x, w1, bn1, w2, bn2, downsample=None):
        """Стандартный residual block с реконструированными весами."""
        identity = x
        out = self.relu(bn1(self._conv(x,   w1)))
        out =           bn2(self._conv(out, w2))
        if downsample is not None:
            identity = downsample(x)
        return self.relu(out + identity)

    def forward(self, x):
        # ── Stem ──────────────────────────────────────────────────────────────
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # ── Layer1 — 4 conv все в группе l1 ──────────────────────────────────
        w = self.gabe["l1"].weights()   # [w0, w1, w2, w3]
        x = self._residual_block_gabe(x, w[0], self.bn_l1_0_1,
                                         w[1], self.bn_l1_0_2)
        x = self._residual_block_gabe(x, w[2], self.bn_l1_1_1,
                                         w[3], self.bn_l1_1_2)

        # ── Layer2 — conv1 (stride=2) обычный; conv2,conv3,conv4 из l2 ───────
        w = self.gabe["l2"].weights()   # [w0, w1, w2]
        # block 0: conv1 (обычный, stride=2) → BN → conv2 (GABE) → BN + shortcut
        out = self.relu(self.bn_l2_0_1(
                F.conv2d(x, self.conv_l2_0_1.weight, bias=None, stride=2, padding=1)))
        out = self.bn_l2_0_2(self._conv(out, w[0]))
        shortcut = self.ds2_bn(F.conv2d(x, self.ds2_conv.weight, bias=None, stride=2))
        x = self.relu(out + shortcut)
        # block 1: оба conv из GABE
        x = self._residual_block_gabe(x, w[1], self.bn_l2_1_1,
                                         w[2], self.bn_l2_1_2)

        # ── Layer3 ────────────────────────────────────────────────────────────
        w = self.gabe["l3"].weights()
        out = self.relu(self.bn_l3_0_1(
                F.conv2d(x, self.conv_l3_0_1.weight, bias=None, stride=2, padding=1)))
        out = self.bn_l3_0_2(self._conv(out, w[0]))
        shortcut = self.ds3_bn(F.conv2d(x, self.ds3_conv.weight, bias=None, stride=2))
        x = self.relu(out + shortcut)
        x = self._residual_block_gabe(x, w[1], self.bn_l3_1_1,
                                         w[2], self.bn_l3_1_2)

        # ── Layer4 ────────────────────────────────────────────────────────────
        w = self.gabe["l4"].weights()
        out = self.relu(self.bn_l4_0_1(
                F.conv2d(x, self.conv_l4_0_1.weight, bias=None, stride=2, padding=1)))
        out = self.bn_l4_0_2(self._conv(out, w[0]))
        shortcut = self.ds4_bn(F.conv2d(x, self.ds4_conv.weight, bias=None, stride=2))
        x = self.relu(out + shortcut)
        x = self._residual_block_gabe(x, w[1], self.bn_l4_1_1,
                                         w[2], self.bn_l4_1_2)

        # ── Head ──────────────────────────────────────────────────────────────
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

AUG = T.Compose([
    T.RandomResizedCrop(224, scale=(0.3, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.3, 0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomRotation(30),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TFM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class TreeDataset(Dataset):
    def __init__(self, pos_paths, neg_paths, n=400, augment=True):
        self.pos   = [Image.open(p).convert("RGB") for p in pos_paths]
        self.neg   = [Image.open(p).convert("RGB") for p in neg_paths]
        self.n     = n
        self.tfm   = AUG if augment else EVAL_TFM
        # цветоинверсия для негативов когда нет отдельных нон-три файлов
        self.neg_tfm = T.Compose([self.tfm, T.Lambda(lambda x: -x)])

    def __len__(self): return self.n * 2

    def __getitem__(self, idx):
        if idx < self.n:
            return self.tfm(self.pos[idx % len(self.pos)]), 1
        else:
            return self.neg_tfm(self.neg[(idx - self.n) % len(self.neg)]), 0

def build_loaders(tree_paths, nontree_paths=None):
    neg = nontree_paths if nontree_paths else tree_paths
    train_ds = TreeDataset(tree_paths, neg, n=200, augment=True)
    val_ds   = TreeDataset(tree_paths, neg, n=40,  augment=False)
    return (DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0))


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader):
    model.eval()
    ok = total = 0
    with torch.no_grad():
        for x, y in loader:
            ok    += (model(x.to(DEVICE)).argmax(-1) == y.to(DEVICE)).sum().item()
            total += len(y)
    return ok / total

def train(model, train_loader, val_loader, params, lr, label):
    loss_fn = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    model.to(DEVICE)
    hist  = []
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        t0 = time.time(); loss_sum = 0.
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss_sum += loss.item()
        sched.step()
        acc = evaluate(model, val_loader)
        hist.append(acc)
        print(f"    [{label} ep {ep:2d}/{N_EPOCHS}]  "
              f"loss={loss_sum/len(train_loader):.4f}  "
              f"val_acc={acc:.4f}  ({time.time()-t0:.1f}s)")
    return hist

@torch.no_grad()
def tree_prob(model, paths):
    model.eval().to(DEVICE)
    return [F.softmax(model(EVAL_TFM(Image.open(p).convert("RGB"))
                            .unsqueeze(0).to(DEVICE)), -1)[0, 1].item()
            for p in paths]

@torch.no_grad()
def kl_agreement(m1, m2, loader, n=10):
    m1.eval(); m2.eval()
    kls = []
    for i, (x, _) in enumerate(loader):
        if i >= n: break
        x = x.to(DEVICE)
        p = F.softmax(m1(x), -1)
        kls.append(F.kl_div(F.log_softmax(m2(x), -1), p, reduction="batchmean").item())
    return float(np.mean(kls))


# ══════════════════════════════════════════════════════════════════════════════
# Анализ компонент после обучения
# ══════════════════════════════════════════════════════════════════════════════

def component_drift(gabe_pre, model_gabe):
    """ΔW̄ и Δα для каждой группы после GABE_FT."""
    rows = []
    for g in GROUP_ORDER:
        gp    = gabe_pre[g]
        grp   = model_gabe.gabe[g]
        dW    = (grp.W_bar.detach().cpu() - gp["W_bar"])
        dA    = (grp.alpha.detach().cpu() - gp["alpha"])
        rows.append(dict(
            group      = g,
            dWbar_rel  = dW.norm().item() / (gp["W_bar"].norm().item() + 1e-10),
            dAlpha_rel = (dA.norm(dim=1) / (gp["alpha"].norm(dim=1) + 1e-10)).tolist(),
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def hr(c="─"): print(c * 80)

def main():
    # ── Найти изображения ─────────────────────────────────────────────────────
    tree_paths    = sorted(glob.glob(os.path.join(SCRIPT_DIR, "tree*.jpg")))
    nontree_paths = sorted(glob.glob(os.path.join(SCRIPT_DIR, "nontree*.jpg"))) or None
    if not tree_paths:
        sys.exit("ERROR: tree*.jpg не найдены рядом со скриптом.")

    print("=" * 80)
    print("GABE New-Class Learning Test  — v2  (ResNet18 + 'tree')")
    print("=" * 80)
    print(f"  device     = {DEVICE}")
    print(f"  tree imgs  = {len(tree_paths)}: {[os.path.basename(p) for p in tree_paths]}")
    print(f"  nontree    = {len(nontree_paths) if nontree_paths else 0} (цветоинверсия если 0)")
    print(f"  epochs     = {N_EPOCHS}  |  LR head/GABE = {LR}  |  LR full = {LR_FULL}")
    print()

    # ── Загрузить ResNet18 ────────────────────────────────────────────────────
    print("  Загружаем ResNet18 (ImageNet-1k pretrained)...")
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    print(f"  Параметров: {sum(p.numel() for p in base.parameters())/1e6:.1f}M")

    train_loader, val_loader = build_loaders(tree_paths, nontree_paths)
    print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    # ── PHASE 1: GABE-разложение ──────────────────────────────────────────────
    hr(); print("PHASE 1 — GABE Exact Decomposition  (k = L-1)"); hr()
    gabe_pre = extract_all(base)
    param_std = param_gabe = 0
    print(f"\n  {'Group':5}  {'L':>3}  {'K':>3}  {'D':>10}  {'recon_err':>10}  "
          f"{'Std L×D':>12}  {'GABE D+LK':>12}  {'Ratio':>7}")
    print("  " + "─" * 72)
    for g in GROUP_ORDER:
        gd = gabe_pre[g]
        L, K, D = gd["L"], gd["K"], gd["D"]
        s, c = L*D, D+L*K
        param_std += s; param_gabe += c
        print(f"  {g:5}  {L:>3}  {K:>3}  {D:>10,}  {gd['recon_err']:>10.2e}  "
              f"{s:>12,}  {c:>12,}  {s/c:>6.1f}×")
    print("  " + "─" * 72)
    print(f"  {'total':5}  {'':>3}  {'':>3}  {'':>10}  {'':>10}  "
          f"{param_std:>12,}  {param_gabe:>12,}  {param_std/param_gabe:>6.1f}×")

    # ── PHASE 2: HEAD_FT ──────────────────────────────────────────────────────
    hr(); print(f"PHASE 2 — HEAD_FT  ({N_EPOCHS} эпох, backbone заморожен)"); hr()
    head_m = copy.deepcopy(base)
    for p in head_m.parameters(): p.requires_grad_(False)
    head_m.fc = nn.Linear(head_m.fc.in_features, 2)
    nn.init.xavier_uniform_(head_m.fc.weight); nn.init.zeros_(head_m.fc.bias)
    hparams = list(head_m.fc.parameters())
    print(f"    Обучаемых параметров: {sum(p.numel() for p in hparams):,}")
    train(head_m, train_loader, val_loader, hparams, LR, "HEAD_FT")
    acc_head = evaluate(head_m.to(DEVICE), val_loader)
    print(f"  HEAD_FT итог: {acc_head:.4f}")
    head_m.cpu().eval()

    # ── PHASE 3: FULL_FT ──────────────────────────────────────────────────────
    hr(); print(f"PHASE 3 — FULL_FT  ({N_EPOCHS} эпох, все веса)"); hr()
    full_m = copy.deepcopy(base)
    full_m.fc = nn.Linear(full_m.fc.in_features, 2)
    nn.init.xavier_uniform_(full_m.fc.weight); nn.init.zeros_(full_m.fc.bias)
    for p in full_m.parameters(): p.requires_grad_(True)
    fparams = list(full_m.parameters())
    print(f"    Обучаемых параметров: {sum(p.numel() for p in fparams):,}")
    train(full_m, train_loader, val_loader, fparams, LR_FULL, "FULL_FT")
    acc_full = evaluate(full_m.to(DEVICE), val_loader)
    print(f"  FULL_FT итог: {acc_full:.4f}")
    full_m.cpu().eval()

    # ── PHASE 4: GABE_FT ──────────────────────────────────────────────────────
    hr(); print(f"PHASE 4 — GABE_FT  ({N_EPOCHS} эпох, W̄ + α + BN, B заморожен)"); hr()
    gabe_m  = GABEResNet(base, gabe_pre, n_classes=2)
    gparams = [p for p in gabe_m.parameters() if p.requires_grad]
    print(f"    Обучаемых параметров: {sum(p.numel() for p in gparams):,}")
    print(f"    Backbone экономия:    {param_std/param_gabe:.1f}× меньше параметров в conv-группах")
    train(gabe_m, train_loader, val_loader, gparams, LR, "GABE_FT")
    acc_gabe = evaluate(gabe_m.to(DEVICE), val_loader)
    print(f"  GABE_FT итог: {acc_gabe:.4f}")
    gabe_m.cpu().eval()

    # ── PHASE 5: Оценка на tree-фото ─────────────────────────────────────────
    hr(); print("PHASE 5 — P(tree) для каждого входного изображения"); hr()
    s_head = tree_prob(head_m, tree_paths)
    s_full = tree_prob(full_m, tree_paths)
    s_gabe = tree_prob(gabe_m, tree_paths)
    print(f"\n  {'Файл':20}  {'HEAD_FT':>9}  {'FULL_FT':>9}  {'GABE_FT':>9}")
    print("  " + "─" * 52)
    for i, p in enumerate(tree_paths):
        print(f"  {os.path.basename(p):20}  {s_head[i]:>9.4f}  "
              f"{s_full[i]:>9.4f}  {s_gabe[i]:>9.4f}")
    print("  " + "─" * 52)
    print(f"  {'среднее':20}  {np.mean(s_head):>9.4f}  "
          f"{np.mean(s_full):>9.4f}  {np.mean(s_gabe):>9.4f}")

    # ── PHASE 6: KL между моделями ────────────────────────────────────────────
    hr(); print("PHASE 6 — Logit Agreement  (KL divergence)"); hr()
    full_m.to(DEVICE); gabe_m.to(DEVICE)
    kl_fg = kl_agreement(full_m, gabe_m, val_loader)
    kl_fh = kl_agreement(full_m, head_m.to(DEVICE), val_loader)
    full_m.cpu(); gabe_m.cpu(); head_m.cpu()
    print(f"\n  KL(FULL_FT || GABE_FT) = {kl_fg:.6f}")
    print(f"  KL(FULL_FT || HEAD_FT) = {kl_fh:.6f}")
    print(f"  {'→ GABE ближе к FULL чем HEAD' if kl_fg < kl_fh else '→ HEAD ближе к FULL чем GABE'}")

    # ── PHASE 7: Дрейф компонент ──────────────────────────────────────────────
    hr(); print("PHASE 7 — Дрейф компонент GABE после обучения"); hr()
    drift = component_drift(gabe_pre, gabe_m)
    print(f"\n  {'Group':5}  {'ΔW̄/W̄₀':>10}  {'Δα по слоям (rel)':>40}")
    print("  " + "─" * 60)
    for d in drift:
        al = "  ".join(f"{v:.4f}" for v in d["dAlpha_rel"])
        print(f"  {d['group']:5}  {d['dWbar_rel']:>10.5f}  {al}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    hr("═"); print("VERDICT"); hr("═")
    gap = abs(acc_full - acc_gabe)
    gabe_better_than_head = acc_gabe >= acc_head

    print(f"""
  Accuracy (val, бинарная: tree / non-tree):
    HEAD_FT  (только голова)   : {acc_head:.4f}
    FULL_FT  (все веса)        : {acc_full:.4f}
    GABE_FT  (W̄ + α, B frozen): {acc_gabe:.4f}   gap vs FULL = {gap:.4f}

  P(tree) на входных фото:
    HEAD_FT  среднее : {np.mean(s_head):.4f}
    FULL_FT  среднее : {np.mean(s_full):.4f}
    GABE_FT  среднее : {np.mean(s_gabe):.4f}

  Logit agreement:
    KL(FULL||GABE) = {kl_fg:.6f}
    KL(FULL||HEAD) = {kl_fh:.6f}

  Параметры conv-групп:
    Стандарт : {param_std:,}
    GABE_FT  : {param_gabe:,}   ({param_std/param_gabe:.1f}× меньше)
""")

    if gabe_better_than_head and gap < 0.05:
        verdict = "✓ GABE_FT ≥ HEAD_FT и ≈ FULL_FT — гипотеза подтверждена"
        detail  = ("  W̄ смещает общее распределение признаков к 'дереву'.\n"
                   "  α перераспределяет веса слоёв — каждый слой адаптируется "
                   "по-своему\n  используя уже существующие базисные направления B.")
    elif gabe_better_than_head:
        verdict = f"◑ GABE_FT > HEAD_FT, отстаёт от FULL_FT на {gap:.4f}"
        detail  = ("  Замороженный B частично ограничивает адаптацию.\n"
                   "  Направления дрейфа в FULL_FT выходят за span(B_pre) — "
                   "см. v2 результат Δ⊥/Δ = 1.0.")
    else:
        verdict = f"✗ GABE_FT < HEAD_FT — требуется доработка"
        detail  = ("  Возможные причины: слишком мало эпох, LR не подходит,\n"
                   "  или предобученный B не охватывает направления дрейфа.")

    print(f"  {verdict}")
    print(f"\n{detail}\n")


if __name__ == "__main__":
    main()