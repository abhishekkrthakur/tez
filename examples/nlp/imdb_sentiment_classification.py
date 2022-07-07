import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics, model_selection
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping


class args:
    model = "bert-base-uncased"
    epochs = 20
    batch_size = 32
    learning_rate = 5e-5
    train_batch_size = 32
    valid_batch_size = 32
    max_len = 128
    accumulation_steps = 1


class IMDBDataset:
    def __init__(self, review, target, tokenizer, max_len):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }


class IMDBModel(nn.Module):
    def __init__(self, model_name, num_train_steps, learning_rate):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": 1,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 1)

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = torch.optim.AdamW(optimizer_parameters, lr=self.learning_rate)
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )

        return opt, sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        device = targets.device.type
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": torch.tensor(accuracy, device=device)}

    def forward(self, ids, mask, token_type_ids, targets=None):
        transformer_out = self.transformer(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
        )
        out = transformer_out.pooler_output
        out = self.dropout(out)
        output = self.output(out)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc


if __name__ == "__main__":
    dfx = pd.read_csv("~/data/imdb.csv").fillna("none")
    dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = IMDBDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    valid_dataset = IMDBDataset(
        review=df_valid.review.values,
        target=df_valid.sentiment.values,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    n_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    model = IMDBModel(
        model_name=args.model,
        num_train_steps=n_train_steps,
        learning_rate=args.learning_rate,
    )
    model = Tez(model)
    es = EarlyStopping(monitor="valid_loss", model_path="model.bin")
    config = TezConfig(
        training_batch_size=args.train_batch_size,
        validation_batch_size=args.valid_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        step_scheduler_after="batch",
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        callbacks=[es],
        config=config,
    )
