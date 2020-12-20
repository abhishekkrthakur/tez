# tez: a simple pytorch trainer

**NOTE: Currently, we are not accepting any pull requests! All PRs will be closed. If you want a feature or something doesn't work, please create an issue.**

tez (तेज़ / تیز) means sharp, fast & active. This is a simple, to-the-point, library to make your pytorch training easy.

_This library is in very early-stage currently! So, there might be breaking changes._

### Idea around tez is simple:

- keep things as simple as possible
- make it as customizable as possible
- clean code
- faster prototyping
- production ready

Currently, tez supports cpu and gpu training. More coming soon!

Using tez is super-easy. We don't want you to be far away from pytorch. So, you do everything on your own and just use tez to make a few things simpler.

### Training using Tez:

- To train a model, define a dataset and model. The dataset class is the same old class you would write when writing pytorch models.

- Create your model class. Instead of inheriting from `nn.Module`, import tez and inherit from `tez.Model` as shown in the following example.


```python
class MyModel(tez.Model):
    def __init__(self):
        super().__init__()
        .
        .
        # tell when to step the scheduler
        self.step_scheduler_after="batch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_scheduler(self):
        # create your own scheduler

    def fetch_optimizer(self):
        # create your own optimizer

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)

        # calculate loss here
        loss = nn.BCEWithLogitsLoss()(output, targets)

        # calculate the metric dictionary here
        metric_dict = self.monitor_metrics(output, targets)
        return output, loss, metric_dict
```

Everything is super-intuitive!

- Now you can train your model!

```python
# init datasets
train_dataset = SomeTrainDataset()
valid_dataset = SomeValidDataset()

# init model
model = MyModel()


# init callbacks, you can also write your own callback
tb_logger = tez.callbacks.TensorBoardLogger(log_dir=".logs/")
es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path="model.bin")

# train model. a familiar api!
model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_bs=32,
    device="cuda",
    epochs=50,
    callbacks=[tb_logger, es],
    fp16=True,
)

# save model (with optimizer and scheduler for future!)
model.save("model.bin")
```

You can checkout examples in `examples/`
