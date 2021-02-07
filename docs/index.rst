.. Tez documentation master file, created by
   sphinx-quickstart on Sun Feb  7 21:17:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tez's documentation!
===============================

Tez is a simple pytorch trainer to make your life easy.
It comes with some useful dataset classes and callbacks.


Instead of inheriting from `nn.Module`, we inherit from `tez.Model`. 

.. code-block:: python

   import tez

   class MyModel(tez.Model):
      def __init__(self):
         super().__init__()
         # do something here
      
      def forward(self, arg1, arg2):
         # do something here
         return outputs, loss, metrics


In Tez, the dataset class and model's forward function are closely related.
The output names from dataset class must be same as the input arguments in forward function of the model.

See an example below:


.. code-block:: python

   import tez

   class MyModel(tez.Model):
      def __init__(self):
         super().__init__()
         .
         .
         # tell when to step the scheduler
         self.step_scheduler_after="batch"

      def monitor_metrics(self, outputs, targets):
         outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
         targets = targets.cpu().detach().numpy()
         accuracy = metrics.accuracy_score(targets, outputs)
         return {"accuracy": accuracy}

      def fetch_scheduler(self):
         # create your own scheduler

      def fetch_optimizer(self):
         # create your own optimizer

      def forward(self, ids, mask, token_type_ids, targets):
         _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
         b_o = self.bert_drop(o_2)
         output = self.out(b_o)

         # calculate loss here
         loss = nn.BCEWithLogitsLoss()(output, targets)

         # calculate the metric dictionary here
         metric_dict = self.monitor_metrics(output, targets)
         return output, loss, metric_dict


.. toctree::
   :maxdepth: 2
   
   tez.model
   tez.datasets
   tez.callbacks
