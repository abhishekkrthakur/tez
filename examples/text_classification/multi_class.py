import pandas as pd
import tez
import torch
import torch.nn as nn
import transformers
from sklearn import metrics, model_selection, preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup


class BERTDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = 64

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
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
            "targets": torch.tensor(self.target[item], dtype=torch.long),
        }


class BERTBaseUncased(tez.Model):
    def __init__(self, num_train_steps, num_classes):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)

        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.CrossEntropyLoss()(outputs, targets)

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc


if __name__ == "__main__":
    dfx = pd.read_csv("/home/abhishek/datasets/bbc-text.csv", nrows=2000)
    dfx = dfx.dropna().reset_index(drop=True)
    lbl_enc = preprocessing.LabelEncoder()
    dfx.category = lbl_enc.fit_transform(dfx.category.values)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.category.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = BERTDataset(
        text=df_train.text.values, target=df_train.category.values
    )

    valid_dataset = BERTDataset(
        text=df_valid.text.values, target=df_valid.category.values
    )

    n_train_steps = int(len(df_train) / 32 * 10)
    model = BERTBaseUncased(
        num_train_steps=n_train_steps, num_classes=dfx.category.nunique()
    )

    # model.load("model.bin")
    tb_logger = tez.callbacks.TensorBoardLogger(log_dir=".logs/")
    es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path="model.bin")
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=32,
        device="cuda",
        epochs=3,
        callbacks=[tb_logger, es],
        fp16=True,
    )
    model.save("model.bin")

    preds = model.predict(valid_dataset, batch_size=16, n_jobs=-1, device="cuda")
    for p in preds:
        print(p)
