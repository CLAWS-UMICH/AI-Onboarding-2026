"""
DIRECTIONS -------------------------------------------------------

Step 1: fine-tune. Write your own version in your folder.

Outline (do it however you like):
- Load data/train.jsonl and hold out the same ~15% split as your baseline.
- Map labels to ids; train the encoder + head with cross-entropy.
- Report accuracy and macro-F1 on the held-out set; print every miss.
- Save to runs/: model weights (model.pt, not committed), label2id.json,
  and metrics.json with at least model, accuracy, macro_f1.

STEPS -----------------------------------------------------------

1. Reuse baseline.py's load_rows, VAL_SIZE and SEED so we train on the same 85% as the baseline
2. Sort the intent names and number them 0..35 (label2id), predict.py needs this map back
3. Train IntentModel (encoder + head, all weights update) with cross-entropy, AdamW,
   and a learning rate that warms up for the first WARMUP fraction of steps then decays to 0
4. After each epoch, score the validation split (15% of train.jsonl): accuracy + macro-F15. Print every validation sentence the final model gets wrong (that's where the data needs work)
6. Score data/test.jsonl ONCE at the end: that's the number in metrics.json
   - only the count of test misses is printed, so test.jsonl doesn't end up steering the data fixes
7. Save runs/model.pt, runs/label2id.json, runs/metrics.json
"""

# LIBRARIES ------------------------------------------------------

import json
import random

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

from baseline import ROOT, DATA, VAL_SIZE, SEED, load_rows, load_test
from model import MIN_CONF, MODEL_NAME, IntentModel, load_tokenizer, pick_device, tokenize

# VARIABLES ------------------------------------------------------

RUNS = ROOT / "runs"

EPOCHS = 4
BATCH = 8
LR = 1e-4 # small, so the pretrained encoder is nudged rather than overwritten, changed from 5e-5 to 1e-4 (worked pretty good acc)
WARMUP = 0.4 # fraction of steps spent ramping the learning rate up from 0

# HELPER FUNCTIONS ------------------------------------------------

# Run the model over a list of sentences in batches, return predicted label ids and average loss
# (loss is only computed when ids + loss_fn are given, otherwise it's 0.0)
# Like predict.py, a top probability under MIN_CONF becomes unhandled
def predict_ids(model, tokenizer, texts, device, unhandled_id, ids=None, loss_fn=None):
    # switches model to evaluation mode, disabling dropout
    model.eval()

    # will store predictions + summed loss later
    preds = []
    total_loss = 0.0

    # disables gradient computation (backprop) - saves memory + speeds up inference
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):

            # tokenize batch of texts and pass through mdoel to get logits
            logits = model(tokenize(tokenizer, texts[i:i + BATCH], device))

            # top probability + its intent id per sentence, low-confidence guesses fall back to unhandled
            conf, idx = logits.softmax(dim=1).max(dim=1)
            idx[conf < MIN_CONF] = unhandled_id
            preds.extend(idx.tolist())

            # held-out loss, weighted by batch size so the average is per sentence
            if loss_fn is not None:
                target = torch.tensor(ids[i:i + BATCH], device=device)
                total_loss += loss_fn(logits, target).item() * len(target)
    return preds, total_loss / len(texts)

# MAIN LOOP ---------------------------------------------------------

def main():
    random.seed(SEED) # sets seed to same as baseline
    torch.manual_seed(SEED) # sets seed for Pytorch's random number generator (for weight initialization)
    device = pick_device()

    texts, labels = load_rows(DATA) # loads raw data
    train_x, val_x, train_y, val_y = train_test_split(
        texts, labels, test_size=VAL_SIZE, stratify=labels, random_state=SEED
    ) # same call as baseline.py, so same split
    test_x, test_y = load_test(texts) # separate handwritten file

    # sorts labels + removes duplicates
    intents = sorted(set(labels))

    # creates mapping dict where each intent maps to # (ID)
    # this lets model work with integer instead of strings
    label2id = {name: i for i, name in enumerate(intents)}

    # converts all training, validation + testing labels from intent names to their numeric IDs from mapping
    train_ids = [label2id[y] for y in train_y]
    val_ids = [label2id[y] for y in val_y]
    test_ids = [label2id[y] for y in test_y]

    print(f"{MODEL_NAME} on {device}: {len(train_x)} train / {len(val_x)} val / {len(test_x)} test, and {len(intents)} intents")

    # loads pre-trained tokenizer
    tokenizer = load_tokenizer()

    # new intent classification model, output layer matches number of intents
    model = IntentModel(len(intents)).to(device)

    # initializes AdamW optimizer with model's parameters and learning rate LR (used to update weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # calculates total training weights
    steps = EPOCHS * ((len(train_x) + BATCH - 1) // BATCH)

    # creates learning rate scheduler that warms up for WARMUP * steps steps, then linearly decays to 0 over rest of training
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * steps), steps)

    # Defines loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # One entry per epoch, saved to runs/history.json for plot_curves.py
    history = []

    # Loops through each training epoch
    for epoch in range(1, EPOCHS + 1):

        # Turns dropout on
        model.train()

        # Creates a list of indices [0, 1, ..., len(train_x) - 1]
        order = list(range(len(train_x)))

        # Randomly shuffles indices so each epoch sees training data in different order (helps with overfitting)
        random.shuffle(order)

        total_loss = 0.0

        # goes through each line in batch
        for i in range(0, len(order), BATCH):

            # gets shuffled indices for this batch
            idx = order[i:i + BATCH]

            # tokenizes: converts to tensors
            batch = tokenize(tokenizer, [train_x[j] for j in idx], device)

            # creates a tensor of correct labels (IDs) for this batch
            target = torch.tensor([train_ids[j] for j in idx], device=device)

            # Runs batch through model to get logits, then computes cross entropy loss between predictions and true labels
            loss = loss_fn(model(batch), target)

            # Clears gradidents from previous step - PyTorch accumulates gradients by default
            optimizer.zero_grad()

            # Backprop
            loss.backward()

            # Updates weights
            optimizer.step()

            # Updates learning rate according to schedule (warmup then decay)
            scheduler.step()

            # Accumulates loss
            total_loss += loss.item() * len(idx)

        # Runs model on the validation split to get predictions + validation loss
        val_pred, val_loss = predict_ids(model, tokenizer, val_x, device, label2id["unhandled"], val_ids, loss_fn)

        # Accuracy + F1 on validation (prints out)
        train_loss = total_loss / len(train_x)
        val_acc = accuracy_score(val_ids, val_pred)
        val_f1 = f1_score(val_ids, val_pred, average="macro")
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  val-loss {val_loss:.4f}  val-acc {val_acc:.3f}  val-macro-F1 {val_f1:.3f}")

        # Records this epoch for ze learning curves
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

    # Creates list of tuples containing only validation samples where model's prediction didn't match true label
    # converts nummeric IDs back to intent names
    misses = [(t, intents[g], intents[p]) for t, g, p in zip(val_x, val_ids, val_pred) if g != p]
    print(f"\n{len(misses)} validation misses out of {len(val_x)} :/")

    # Loops through each mistake and prints text, expected intent, predicted intent
    for text, gold, guess in misses:
        print(f"  MISS '{text}'  expected {gold}  got {guess}")

    # Final score on test.jsonl, computed once with the finished model
    # Misses aren't printed cause then I look at them and wanna fix stuff
    pred, _ = predict_ids(model, tokenizer, test_x, device, label2id["unhandled"])
    acc = accuracy_score(test_ids, pred)
    macro_f1 = f1_score(test_ids, pred, average="macro")
    weighted_f1 = f1_score(test_ids, pred, average="weighted")
    test_misses = sum(g != p for g, p in zip(test_ids, pred))
    print(f"\nTEST ({len(test_x)} sentences)  acc {acc:.3f}  macro-F1 {macro_f1:.3f}  weighted-F1 {weighted_f1:.3f}  ({test_misses} misses)")
    # creates output directory
    RUNS.mkdir(exist_ok=True)

    # saves trained model weights to a file
    torch.save(model.state_dict(), RUNS / "model.pt")

    # Saves intent name, ID mapping as JSON - needed later to convert predictions back to intent naems
    (RUNS / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")

    # creates dict with all training hyperparameters and metrics (results)
    metrics = {
        "model": MODEL_NAME,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "train_rows": len(train_x),
        "val_rows": len(val_x),
        "test_rows": len(test_x),
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "warmup": WARMUP,
        "seed": SEED,
    }

    # saves metrics dict as JSON file for later comparison
    (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # saves per-epoch history so plot_curves.py can draw learning curves without ze retraining
    (RUNS / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
