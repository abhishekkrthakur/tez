.. Tez documentation master file, created by
   sphinx-quickstart on Sun Feb  7 21:17:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tez's documentation!
===============================

Tez is a simple pytorch trainer to make your life easy.
It comes with some useful dataset classes and callbacks.

.. code-block:: python

   from tez import Tez, TezConfig

   class MyModel(nn.Module):
      def __init__(self):
         super().__init__()
         # do something here

      def optimizer_scheduler(self):
         # return optimizer and scheduler here
         return opt, sch
      
      def forward(self, arg1, arg2):
         # do something here
         return outputs, loss, metrics


In Tez, the dataset class and model's forward function are closely related.
The output names from dataset class must be same as the input arguments in forward function of the model.

.. toctree::
   :maxdepth: 2
   
   tez.datasets
   tez.callbacks
