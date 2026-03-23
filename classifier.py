"""
Legal QA Complexity Classifier
================================
Model   : nlpaueb/legal-bert-base-uncased  (fine-tuned)
Input   : question text
Output  : confidence score in [0, 1]  where  1.0 = definitely complex
          score > 0.5  →  complex
          score < 0.5  →  simple
          score ~ 0.5  →  borderline / uncertain

Evaluation:
  • Accuracy, Precision, Recall, F1
  • ROC-AUC curve
  • Confusion Matrix
  • Calibration Curve  (reliability of confidence scores)

Install:
    pip install transformers torch scikit-learn matplotlib seaborn

Training time on CPU: ~15-25 minutes for 3 epochs
"""

import json, os, random, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_JSON      = r"C:\Users\sarat\OneDrive\Documents\Legal_QA_Retrival\RL-GUIDED-MULTI-HOP-LEGAL-QUESTION-ANSWERING-SYSTEM\qa_dataset_chroma.json"   # your generated dataset
MODEL_NAME      = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR      = "complexity_classifier"    # saved model goes here
PLOTS_DIR       = "evaluation_plots"

# Training hyperparameters (tuned for CPU)
MAX_LENGTH      = 128    # legal questions rarely exceed 128 tokens
BATCH_SIZE      = 16
EPOCHS          = 4
LEARNING_RATE   = 2e-5
WARMUP_RATIO    = 0.1
DROPOUT         = 0.3
TEST_SIZE       = 0.15   # 15% test set
VAL_SIZE        = 0.15   # 15% validation set
SEED            = 42

# Score convention:  1.0 = complex,  0.0 = simple
# complex subtypes (comparative, analytical, conditional) all get label 1
COMPLEX_TYPES   = {"comparative", "analytical", "conditional"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions, labels = [], []
    type_counts = {}
    for entry in data:
        q    = entry.get("question", "").strip()
        qtype = entry.get("type", "").strip().lower()
        if not q:
            continue
        label = 1 if qtype in COMPLEX_TYPES else 0
        questions.append(q)
        labels.append(label)
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    print("📊 Dataset composition:")
    for t, c in sorted(type_counts.items()):
        tag = "COMPLEX" if t in COMPLEX_TYPES else "simple "
        print(f"   [{tag}] {t:15s} : {c}")
    print(f"   Total simple  : {labels.count(0)}")
    print(f"   Total complex : {labels.count(1)}")
    print(f"   Total         : {len(labels)}\n")
    return questions, labels


# ─────────────────────────────────────────────
# 2. DATASET CLASS
# ─────────────────────────────────────────────
class QuestionDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_length):
        self.questions  = questions
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.questions[idx],
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.float)
        }


# ─────────────────────────────────────────────
# 3. MODEL
# Legal-BERT + 2-layer classification head
# Output: single sigmoid score in [0,1]
# ─────────────────────────────────────────────
class LegalComplexityClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden    = self.bert.config.hidden_size   # 768 for BERT-base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
            nn.Sigmoid()          # output in [0, 1]
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding — sentence-level representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        score = self.classifier(cls_embedding)
        return score.squeeze(-1)   # shape: (batch_size,)


# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        scores = model(input_ids, attention_mask)
        loss   = criterion(scores, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients on CPU
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 5. EVALUATE (returns scores + labels for plots)
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            scores   = model(input_ids, attention_mask)
            loss     = criterion(scores, labels)
            total_loss += loss.item()

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_scores), np.array(all_labels)


# ─────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────
def compute_metrics(scores, labels, threshold=0.5, split_name="Test"):
    preds = (scores >= threshold).astype(int)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, scores)

    print(f"\n{'='*50}")
    print(f"  {split_name} Metrics  (threshold = {threshold})")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (of predicted complex, how many are truly complex)")
    print(f"  Recall    : {rec:.4f}  (of all complex questions, how many did we catch)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}  (1.0 = perfect, 0.5 = random)")
    print(f"{'='*50}")

    print(f"\n  Classification Report:")
    print(classification_report(labels, preds, target_names=["Simple", "Complex"]))

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc}


# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def plot_all(test_scores, test_labels, train_losses, val_losses, plots_dir):
    # ── color palette ──
    C1, C2 = "#2E4057", "#E84855"

    # ── Figure layout: 2 rows x 3 cols ──
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Legal QA Complexity Classifier — Evaluation Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    preds = (test_scores >= 0.5).astype(int)

    # ── Plot 1: Training curve ──
    ax1 = fig.add_subplot(gs[0, 0])
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, "o-", color=C1, label="Train Loss")
    ax1.plot(epochs_range, val_losses,   "s--", color=C2,  label="Val Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Plot 2: ROC Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    auc_val      = roc_auc_score(test_labels, test_scores)
    ax2.plot(fpr, tpr, color=C1, lw=2, label=f"ROC (AUC = {auc_val:.3f})")
    ax2.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax2.set_title("ROC Curve", fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ── Plot 3: Confusion Matrix ──
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(test_labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Simple", "Complex"],
                yticklabels=["Simple", "Complex"],
                ax=ax3, cbar=False)
    ax3.set_title("Confusion Matrix", fontweight="bold")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    # ── Plot 4: Calibration Curve ──
    ax4 = fig.add_subplot(gs[1, 0])
    fraction_of_pos, mean_pred_val = calibration_curve(
        test_labels, test_scores, n_bins=10, strategy="uniform"
    )
    ax4.plot(mean_pred_val, fraction_of_pos, "o-", color=C1, label="Model")
    ax4.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
    ax4.set_title("Calibration Curve\n(Reliability of Confidence Scores)",
                  fontweight="bold")
    ax4.set_xlabel("Mean Predicted Score")
    ax4.set_ylabel("Fraction of Complex Questions")
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

    # ── Plot 5: Score Distribution ──
    ax5 = fig.add_subplot(gs[1, 1])
    simple_scores  = test_scores[test_labels == 0]
    complex_scores = test_scores[test_labels == 1]
    ax5.hist(simple_scores,  bins=25, alpha=0.7, color="#4CAF50", label="Simple",  density=True)
    ax5.hist(complex_scores, bins=25, alpha=0.7, color=C2,        label="Complex", density=True)
    ax5.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="Threshold=0.5")
    ax5.set_title("Score Distribution by Class", fontweight="bold")
    ax5.set_xlabel("Confidence Score  P(complex)")
    ax5.set_ylabel("Density")
    ax5.legend()
    ax5.grid(alpha=0.3)

    # ── Plot 6: Per-class Metrics Bar Chart ──
    ax6 = fig.add_subplot(gs[1, 2])
    report   = classification_report(
        test_labels, preds,
        target_names=["Simple", "Complex"],
        output_dict=True
    )
    metrics  = ["precision", "recall", "f1-score"]
    simple_v = [report["Simple"][m]  for m in metrics]
    complex_v= [report["Complex"][m] for m in metrics]
    x        = np.arange(len(metrics))
    w        = 0.35
    ax6.bar(x - w/2, simple_v,  w, label="Simple",  color="#4CAF50", alpha=0.85)
    ax6.bar(x + w/2, complex_v, w, label="Complex", color=C2,        alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(["Precision", "Recall", "F1"])
    ax6.set_ylim(0, 1.05)
    ax6.set_title("Per-Class Metrics", fontweight="bold")
    ax6.legend()
    ax6.grid(alpha=0.3, axis="y")
    for bar in ax6.patches:
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}",
                 ha="center", va="bottom", fontsize=8)

    path = os.path.join(plots_dir, "evaluation_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📊 Evaluation dashboard saved to: {path}")


# ─────────────────────────────────────────────
# 8. MAIN  — with full resume logic
#
# Three states the script handles:
#   A) Training fully done (best_model.pt + training_complete.flag exist)
#      → skip straight to evaluation + plots
#   B) Training partially done (resume.pt exists, epochs < EPOCHS)
#      → resume from last completed epoch, continue training
#   C) Fresh start (nothing saved)
#      → train from epoch 1
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device : {device}\n")

    best_model_path    = os.path.join(OUTPUT_DIR, "best_model.pt")
    resume_path        = os.path.join(OUTPUT_DIR, "resume.pt")
    losses_path        = os.path.join(OUTPUT_DIR, "losses.json")
    complete_flag_path = os.path.join(OUTPUT_DIR, "training_complete.flag")

    # ── Load data (always needed — same SEED guarantees same split) ──
    questions, labels = load_data(INPUT_JSON)

    q_train, q_temp, l_train, l_temp = train_test_split(
        questions, labels, test_size=(TEST_SIZE + VAL_SIZE),
        stratify=labels, random_state=SEED
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    q_val, q_test, l_val, l_test = train_test_split(
        q_temp, l_temp, test_size=(1 - val_ratio),
        stratify=l_temp, random_state=SEED
    )

    print(f"📂 Split sizes:")
    print(f"   Train : {len(q_train)}  (simple={l_train.count(0)}, complex={l_train.count(1)})")
    print(f"   Val   : {len(q_val)}  (simple={l_val.count(0)}, complex={l_val.count(1)})")
    print(f"   Test  : {len(q_test)}  (simple={l_test.count(0)}, complex={l_test.count(1)})\n")

    # ── Tokenizer ──
    # Load from saved dir if available (faster), else download
    tok_source = OUTPUT_DIR if os.path.exists(os.path.join(OUTPUT_DIR, "tokenizer_config.json")) \
                 else MODEL_NAME
    print(f"⏳ Loading tokenizer from: {tok_source}")
    tokenizer = AutoTokenizer.from_pretrained(tok_source)

    train_ds = QuestionDataset(q_train, l_train, tokenizer, MAX_LENGTH)
    val_ds   = QuestionDataset(q_val,   l_val,   tokenizer, MAX_LENGTH)
    test_ds  = QuestionDataset(q_test,  l_test,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    criterion = nn.BCELoss()

    # ════════════════════════════════════════════
    # STATE A — Training already fully complete
    # → skip straight to evaluation
    # ════════════════════════════════════════════
    if os.path.exists(complete_flag_path) and os.path.exists(best_model_path):
        print("✅ Training already complete (training_complete.flag found).")
        print("   Skipping training — going straight to evaluation & plots.\n")

        # Load model
        model = LegalComplexityClassifier(MODEL_NAME, dropout=DROPOUT).to(device)
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"📥 Loaded best model from epoch {checkpoint['epoch']} "
              f"(val_loss={checkpoint['val_loss']:.4f})\n")

        # Load saved losses for the training curve plot
        train_losses, val_losses = [], []
        if os.path.exists(losses_path):
            with open(losses_path) as f:
                loss_data    = json.load(f)
                train_losses = loss_data.get("train", [])
                val_losses   = loss_data.get("val",   [])

        # Run test evaluation + plots
        _, test_scores, test_labels_arr = evaluate(model, test_loader, criterion, device)
        metrics = compute_metrics(test_scores, np.array(test_labels_arr), threshold=0.5)

        metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        print(f"📄 Metrics saved to: {metrics_path}")

        print("\n📊 Generating evaluation plots...")
        plot_all(test_scores, np.array(test_labels_arr),
                 train_losses, val_losses, PLOTS_DIR)

        _print_score_summary(test_scores, np.array(test_labels_arr))
        _print_inference_snippet()
        return

    # ════════════════════════════════════════════
    # STATE B or C — Need to train (full or partial)
    # ════════════════════════════════════════════
    print(f"⏳ Loading model: {MODEL_NAME} ...")
    model = LegalComplexityClassifier(MODEL_NAME, dropout=DROPOUT).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    train_losses  = []
    val_losses    = []
    best_val_loss = float("inf")
    start_epoch   = 1

    # ── STATE B: Resume from checkpoint ──
    if os.path.exists(resume_path):
        print(f"\n↩️  Resume checkpoint found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        start_epoch   = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_val_loss"]

        if os.path.exists(losses_path):
            with open(losses_path) as f:
                loss_data    = json.load(f)
                train_losses = loss_data.get("train", [])
                val_losses   = loss_data.get("val",   [])

        print(f"   Resuming from epoch {start_epoch} / {EPOCHS}")
        print(f"   Best val loss so far: {best_val_loss:.4f}\n")
    else:
        print(f"✅ Model loaded — starting fresh from epoch 1\n")

    # ── Training loop ──
    print(f"🚀 Training epochs {start_epoch}→{EPOCHS} on {device}...")
    print(f"   Batch size : {BATCH_SIZE}  |  LR : {LEARNING_RATE}  "
          f"|  Max length : {MAX_LENGTH}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"── Epoch {epoch}/{EPOCHS} ──")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 criterion, device)
        val_loss, val_scores, val_labels_ep = evaluate(model, val_loader,
                                                        criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save losses to disk after every epoch
        with open(losses_path, "w") as f:
            json.dump({"train": train_losses, "val": val_losses}, f)

        val_preds = (val_scores >= 0.5).astype(int)
        val_f1    = f1_score(val_labels_ep, val_preds, zero_division=0)
        val_auc   = roc_auc_score(val_labels_ep, val_scores)

        print(f"   Train Loss : {train_loss:.4f}")
        print(f"   Val Loss   : {val_loss:.4f}  |  Val F1: {val_f1:.4f}  |  Val AUC: {val_auc:.4f}")

        # Save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : val_loss,
                "val_f1"      : val_f1,
                "val_auc"     : val_auc,
                "config"      : {"model_name": MODEL_NAME,
                                 "max_length": MAX_LENGTH,
                                 "dropout"   : DROPOUT}
            }, best_model_path)
            print(f"   💾 Best model saved (val_loss={val_loss:.4f})")

        # Save resume checkpoint (overwrites each epoch — always latest)
        torch.save({
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss"  : best_val_loss,
        }, resume_path)
        print(f"   💾 Resume checkpoint saved (epoch {epoch})")
        print()

    # ── All epochs done — write completion flag ──
    with open(complete_flag_path, "w") as f:
        f.write(f"Training completed. Best val_loss={best_val_loss:.4f}\n")
    print("🏁 Training complete — flag written.\n")

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Load best model for final evaluation ──
    print("📥 Loading best model for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"   Best was epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})\n")

    _, test_scores, test_labels_arr = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(test_scores, np.array(test_labels_arr), threshold=0.5)

    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"📄 Metrics saved to: {metrics_path}")

    print("\n📊 Generating evaluation plots...")
    plot_all(test_scores, np.array(test_labels_arr),
             train_losses, val_losses, PLOTS_DIR)

    _print_score_summary(test_scores, np.array(test_labels_arr))
    _print_inference_snippet()


# ─────────────────────────────────────────────
# HELPERS (called from both State A and training path)
# ─────────────────────────────────────────────
def _print_score_summary(test_scores, test_labels):
    print("\n── Confidence Score Summary ──")
    print(f"   Simple  questions → mean score : "
          f"{test_scores[test_labels == 0].mean():.3f}")
    print(f"   Complex questions → mean score : "
          f"{test_scores[test_labels == 1].mean():.3f}")
    print(f"   (ideal gap ≥ 0.4 between the two means)")
    print(f"\n✅ Done! Model saved in: {OUTPUT_DIR}/")
    print(f"   Use best_model.pt for inference in your RL pipeline.\n")

def _print_inference_snippet():
    print("── How to use in your RL pipeline ──")
    print("""
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("complexity_classifier/")
    model     = LegalComplexityClassifier("nlpaueb/legal-bert-base-uncased")
    ckpt      = torch.load("complexity_classifier/best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def get_complexity_score(question: str) -> float:
        enc = tokenizer(question, return_tensors="pt",
                        max_length=128, truncation=True, padding="max_length")
        with torch.no_grad():
            score = model(enc["input_ids"], enc["attention_mask"])
        return score.item()   # float in [0,1] — pass directly to RL agent
    """)


if __name__ == "__main__":
    main()