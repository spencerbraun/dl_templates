import argparse
import json
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

import wandb

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.utils.dummy_pt_objects import AutoModel


def json_loader(path):
    with open(path) as f:
        return json.load(f)


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
    )

    model = AutoModel.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
    )

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))

    return tokenizer, model


def load_data(args, Xname="X_train.json", yname="y_train.json", shuffle=True):

    X = json_loader(args.data_dir + "/" + Xname)
    y = json_loader(args.data_dir + "/" + yname)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    return dl


def train(args, model, train_loader, val_loader):

    epochs = args.epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_steps = int(len(train_loader) * args.num_train_epochs)

    scaler = GradScaler()
    warmup_steps = min(int(total_steps * args.warmup_ratio), 200)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))

    output = []
    nsteps = 0
    for epoch in tqdm(range(epochs)):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            output.append(out)

            loss = model.loss(y)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()

            if not args.debug:
                wandb.log({"loss": loss.item()}, step=nsteps)
            nsteps += 1

            if nsteps % args.eval_freq == 0:
                val_loss = evaluate(model, val_loader)
                wandb.log({"val_loss": val_loss})

        if args.save_loc and (total_steps > 500):
            torch.save(model.state_dict(), args.save_loc + f"_epoch{epoch}")

        print("Epoch: {}/{}".format(epoch, epochs))
        print("Loss: {:.4f}".format(loss.item()))
        print("Accuracy: {:.4f}".format(evaluate(model, val_loader)))

    if args.save_loc:
        torch.save(model.state_dict(), args.save_loc + f"_final")


def evaluate(model, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.max(dim=-1).values
            correct += pred.eq(y).sum().item()
    return correct / len(val_loader.dataset)


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not args.debug:
        run_name = f"{args.model}_{args.dataset}_{args.input_format}_format_{timestamp}"
        wandb.init(project=args.project, config=args.__dict__, name=run_name)
        transformers.logging.set_verbosity_info()

    tokenizer, model = load_model(args)

    train_loader = load_data(args, Xname="X_train.json", yname="y_train.json")
    val_loader = load_data(args, Xname="X_val.json", yname="y_val.json")

    train(args, model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--project_name", type=str, default="pytorch_base")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--save_loc", type=str, default="../models/")
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    main(args)
