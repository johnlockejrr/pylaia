# Training

The `pylaia-htr-train-ctc` command can be used to train a PyLaia model. To know more about the options of this command, use `pylaia-htr-train-ctc --help`.

## Purpose

This command trains a PyLaia architecture on a dataset.

It requires:

- a [formatted dataset](../datasets/index.md),
- the pickled `model` file created during [model initialization](../initialization/index.md).

!!! note

    The [`pylaia-htr-dataset-validate`](../datasets/index.md) command can help you analyze your dataset and point out issues.

## Parameters

The full list of parameters is detailed in this section.

### General parameters

| Parameter      | Description                                                                                                         | Type   | Default |
| -------------- | ------------------------------------------------------------------------------------------------------------------- | ------ | ------- |
| `syms`         | Positional argument. Path to a file mapping characters to integers. The CTC symbol **must** be mapped to integer 0. | `str`  |         |
| `img_dirs`     | Positional argument. Directories containing line images.                                                            | `str`  |         |
| `tr_txt_table` | Positional argument. Path to a file mapping training image ids and tokenized transcription.                         | `str`  |         |
| `va_txt_table` | Positional argument. Path to a file mapping validation image ids and tokenized transcription.                       | `str`  |         |
| `config`       | Path to a JSON configuration file                                                                                   | `json` |         |

### Common parameters

| Name                        | Description                                                                                                                                                                                                                                         | Type            | Default          |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------- |
| `common.seed`               | Seed for random number generators.                                                                                                                                                                                                                  | `int`           | `74565`          |
| `common.train_path`         | Directory where the model will be saved                                                                                                                                                                                                             | `str`           | `.`              |
| `common.model_filename`     | Filename of the model.                                                                                                                                                                                                                              | `str`           | `model`          |
| `common.experiment_dirname` | Directory name of the experiment.                                                                                                                                                                                                                   | `str`           | `experiment`     |
| `common.monitor`            | Metric to monitor for early stopping and checkpointing.                                                                                                                                                                                             | `Monitor`       | `Monitor.va_cer` |
| `common.checkpoint`         | Checkpoint to load. Must be a filepath, a filename, a glob pattern or `None` (in this case, the best checkpoint will be loaded). Note that the checkpoint will be searched in `common.experiment_dirname`, unless you provide an absolute filepath. | `Optional[str]` | `None`           |

### Data arguments

| Name               | Description                                       | Type        | Default       |
| ------------------ | ------------------------------------------------- | ----------- | ------------- |
| `data.batch_size`  | Batch size.                                       | `int`       | `8`           |
| `data.color_mode`  | Color mode. Must be either `L`, `RGB` or `RGBA`.  | `ColorMode` | `ColorMode.L` |
| `data.num_workers`  | Number of worker processes created in dataloaders | `int`       | `None`        |
| `data.reading_order` | Reading order on the input lines: LFT (Left-to-Right) or RTL (Right-to-Left). | `ReadingOrder`       | `LFT`        |

### Train arguments

| Name                            | Description                                                                                                                   | Type               | Default       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------- |
| `train.delimiters`              | List of symbols representing the word delimiters.                                                                             | `List`             | `["<space>"]` |
| `train.checkpoint_k`            | Model saving mode: `-1` all models will be saved, `0`: no models are saved, `k` the `k` best models are saved.                | `int`              | `3`           |
| `train.resume`                  | Whether to resume training with a checkpoint. This option can be used to continue training on the same dataset. | `bool` | `False`       |
| `train.pretrain`                | Whether to load pretrained weights from a checkpoint. This option can be used to load pretrained weights when fine-tuning a model on a new dataset. | `bool` | `False`       |
| `train.freeze_layers`           | List of layers to freeze during training: `"conv"` to freeze convolutional layers, `"rnn"` to freeze recurrent layers, `"linear"` to freeze the linear layer | `List[str]` | `None`       |
| `train.early_stopping_patience` | Number of validation epochs with no improvement after which training will be stopped.                                         | `int`              | `20`          |
| `train.gpu_stats`               | Whether to include GPU stats in the training progress bar.                                                                    | `bool`             | `False`       |
| `train.augment_training`        | Whether to use data augmentation.                                                                                             | `bool`             | `False`       |


### Logging arguments

| Name                      | Description                                                                                                    | Type            | Default                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------- |
| `logging.fmt`             | Logging format.                                                                                                | `str`           | `%(asctime)s %(levelname)s %(name)s] %(message)s` |
| `logging.level`           | Logging level. Should be in `{NOTSET,DEBUG,INFO,WARNING,ERROR,CRITICAL}`                                       | `Level`         | `INFO`                                            |
| `logging.filepath`        | Filepath for the logs file. Can be a filepath or a filename to be created in `train_path`/`experiment_dirname` | `Optional[str]` |                                                   |
| `logging.overwrite`       | Whether to overwrite the logfile or to append.                                                                 | `bool`          | `False`                                           |
| `logging.to_stderr_level` | If filename is set, use this to log also to stderr at the given level.                                         | `Level`         | `ERROR`                                           |

### Optimizer arguments

| Name                           | Description                                               | Type    | Default   |
| ------------------------------ | --------------------------------------------------------- | ------- | --------- |
| `optimizers.name`              | Optimization algorithm. Must be `SGD`, `RMSProp`, `Adam`. | `List`  | `RMSProp` |
| `optimizers.learning_rate`     | Learning rate.                                            | `float` | `0.0005`  |
| `optimizers.momentum`          | Momentum.                                                 | `float` | `0.0`     |
| `optimizers.weight_l2_penalty` | Apply this L2 weight penalty to the loss function.        | `float` | `0.0`     |
| `optimizers.nesterov`          | Whether to use Nesterov momentum.                         | `bool`  | `False`   |

### Scheduler arguments

| Name                 | Description                                                                     | Type      | Default           |
| -------------------- | ------------------------------------------------------------------------------- | --------- | ----------------- |
| `scheduler.active`   | Whether to use an on-plateau learning rate scheduler.                           | `bool`    | `False`           |
| `scheduler.monitor`  | Metric for the scheduler to monitor.                                            | `Monitor` | `Monitor.va_loss` |
| `scheduler.patience` | Number of epochs with no improvement after which learning rate will be reduced. | `int`     | `5`               |
| `scheduler.factor`   | Factor by which the learning rate will be reduced.                              | `float`   | `0.1`             |

### Trainer arguments

Pytorch Lighning `Trainer` flags can also be set using the `--trainer` argument. See [the documentation](https://github.com/Lightning-AI/lightning/blob/1.7.0/docs/source-pytorch/common/trainer.rst#trainer-flags).


## Examples

The model can be trained using command-line arguments or a YAML configuration file. Note that CLI arguments override the values from the configuration file.


### Train from scratch with Command Line Arguments (CLI)

Run the following command to train a model:
```sh
pylaia-htr-train-ctc /path/to/syms.txt \
   `cat img_dirs_args.txt`\
   /path/to/train.txt \
   /path/to/val.txt \
   --trainer.gpus 1 \
   --data.batch_size 32
```

### Train from scratch with a YAML configuration file

Run the following command to train a model:
```sh
pylaia-htr-train-ctc --config config_train_model.yaml
```

Where `config_train_model.yaml` is:

```yaml
syms: /path/to/syms.txt
img_dirs:
  - /path/to/images/
tr_txt_table: /path/to/train.txt
va_txt_table: /path/to/val.txt
common:
  experiment_dirname: experiment-dataset
logging:
  filepath: pylaia_training.log
scheduler:
  active: true
train:
  augment_training: true
  early_stopping_patience: 80
trainer:
  auto_select_gpus: true
  gpus: 1
  max_epochs: 600
```

### Resume training from a checkpoint

Run the following command to continue training from a checkpoint for 200 epochs.
```sh
pylaia-htr-train-ctc --config config_train_model.yaml --train.resume true --trainer.max_epochs 200
```

!!! note
    If `common.checkpoint` is not set, PyLaia will select the best checkpoint from `common.experiment_dirname`

### Fine-tune from a checkpoint

Run the following command to load pretrained weights and fine-tune on a new dataset for 200 epochs.
```sh
pylaia-htr-train-ctc --config config_train_model.yaml --common.experiment_dirname experiment/ --common.checkpoint initial_checkpoint.ckpt --train.pretrain true --trainer.max_epochs 200
```

!!! warning
    This option requires that your model architecture `model` matches the one used to train `initial_checkpoint.ckpt`.
    The last linear layer will be reinitialized using the Xavier initialization to match the new vocabulary size.

!!! note
    The initial checkpoint is expected to be in the following directory: `{common.experiment_dirname}/pretrained/`.
    If it is located in `common.experiment_dirname`, the subdirectory `pretrained` will be created and the checkpoint will be moved there automatically.

### Train on Right-To-Left reading order

By default, PyLaia expects images with Left-to-Right reading order.
To train a model on Right-To-Left data, use the following command:
```sh
pylaia-htr-train-ctc --config config_train_model_rtl.yaml
```

Where `config_train_model_rtl.yaml` is:

```yaml
syms: /path/to/syms.txt
img_dirs:
  - /path/to/images/
tr_txt_table: /path/to/train.txt
va_txt_table: /path/to/val.txt
common:
  experiment_dirname: experiment-dataset
logging:
  filepath: pylaia_training.log
scheduler:
  active: true
train:
  augment_training: true
  early_stopping_patience: 80
trainer:
  auto_select_gpus: true
  gpus: 1
  max_epochs: 600
data:
  reading_order: RTL
```
