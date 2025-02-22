# Dataset

PyLaia datasets must be formatted following a specific format. Learn how to build a dataset by following this [page](./format.md).

Once the dataset is created, you may use the `pylaia-htr-dataset-validate` command to compute statistics and make sure your dataset is valid. To know more about the options of this command, use `pylaia-htr-dataset-validate --help`.


## Purpose

This command will:

* issue a warning if some images are missing (they will be ignored during training)
* issue a warning if some images have an invalid width (they will be padded during training)
* fail if images have variable height when  `fixed_input_height>0`
* fail if a character is missing in the list of symbols `syms.txt`

If the dataset is valid, the script will:

* display `Dataset is valid` and
* save a summary of the dataset statistics in a Markdown file named after the argument provided in `--statistics_output`.

## Parameters

The full list of parameters is detailed in this section.

### General parameters

| Parameter            | Description                                                                                                                                                                                          | Type   | Default |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------- |
| `syms`               | Positional argument. Path to a file mapping characters to integers. The CTC symbol must be mapped to integer 0.                                                                                      | `str`  |         |
| `img_dirs`           | Positional argument. Directories containing line images.                                                                                                                                             | `str`  |         |
| `tr_txt_table`       | Positional argument. Path to a file mapping training image ids and tokenized transcription.                                                                                                          | `str`  |         |
| `va_txt_table`       | Positional argument. Path to a file mapping validation image ids and tokenized transcription.                                                                                                        | `str`  |         |
| `te_txt_table`       | Positional argument. Path to a file mapping validation image ids and tokenized transcription.                                                                                                        | `str`  |         |
| `fixed_input_height` | Height of the input images. If set to 0, a variable height model will be used (see `adaptive_pooling`). This will be used to compute the model output height at the end of the convolutional layers. | `int`  | 0       |
| `statistics_output`  | Where the Markdown summary will be written.                                                                                                                                                          | `str`  | `"statistics.md"`       |
| `config`             | Path to a JSON configuration file                                                                                                                                                                    | `json` |         |

### Common parameters

| Name                    | Description                             | Type  | Default |
| ----------------------- | --------------------------------------- | ----- | ------- |
| `common.train_path`     | Directory where the model will be saved | `str` | `.`     |
| `common.model_filename` | Filename of the model.                  | `str` | `model` |

### Logging arguments

| Name                      | Description                                                                                                    | Type            | Default                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------- |
| `logging.fmt`             | Logging format.                                                                                                | `str`           | `%(asctime)s %(levelname)s %(name)s] %(message)s` |
| `logging.level`           | Logging level. Should be in `{NOTSET,DEBUG,INFO,WARNING,ERROR,CRITICAL}`                                       | `Level`         | `INFO`                                            |
| `logging.filepath`        | Filepath for the logs file. Can be a filepath or a filename to be created in `train_path`/`experiment_dirname` | `Optional[str]` |                                                   |
| `logging.overwrite`       | Whether to overwrite the logfile or to append.                                                                 | `bool`          | `False`                                           |
| `logging.to_stderr_level` | If filename is set, use this to log also to stderr at the given level.                                         | `Level`         | `ERROR`                                           |

### Train arguments

| Name               | Description                                       | Type   | Default       |
| ------------------ | ------------------------------------------------- | ------ | ------------- |
| `train.delimiters` | List of symbols representing the word delimiters. | `List` | `["<space>"]` |

## Examples

This arguments can be passed using command-line arguments or a YAML configuration file. Note that CLI arguments override the values from the configuration file.

### Example with Command Line Arguments (CLI)

Run the following command to create a model:
```sh
pylaia-htr-dataset-validate /data/Esposalles/dataset/syms.txt \
                            [/data/Esposalles/dataset/images/] \
                            /data/Esposalles/dataset/train.txt \
                            /data/Esposalles/dataset/val.txt \
                            /data/Esposalles/dataset/test.txt \
                            --common.experiment_dirname experiment-esposalles/ \
                            --fixed_input_size 128 \
                            --statistice_ouput statistics.md
```

Will output:
```bash
[2024-04-23 12:58:31,399 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-23 12:58:32,010 INFO laia] Installed:
[2024-04-23 12:58:32,050 INFO laia.common.loader] Loaded model model
[2024-04-23 12:58:32,094 INFO laia] Dataset is valid
[2024-04-23 12:58:32,094 INFO laia] Statistics written to statistics.md
```

### Example with a YAML configuration file

Run the following command to validate a dataset:
```sh
pylaia-htr-dataset-validate --config config_dataset.yaml
```

Where `config_dataset.yaml` is:

```yaml
syms: /data/Esposalles/dataset/syms.txt
img_dirs: [/data/Esposalles/dataset/images/]
tr_txt_table: /data/Esposalles/dataset/train.txt
va_txt_table: /data/Esposalles/dataset/val.txt
te_txt_table: /data/Esposalles/dataset/test.txt
fixed_input_height: 128
statistics_output: statistics.md
common:
  experiment_dirname: experiment-esposalles
```

### Example with perfect dataset

```bash
[2024-04-23 12:58:31,399 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-23 12:58:32,010 INFO laia] Installed:
[2024-04-23 12:58:32,050 INFO laia.common.loader] Loaded model model
[2024-04-23 12:58:32,094 INFO laia] Dataset is valid
[2024-04-23 12:58:32,094 INFO laia] Statistics written to statistics.md
```

### Example with missing images

```bash
[2024-04-23 13:01:34,646 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-23 13:01:35,200 INFO laia] Installed:
[2024-04-23 13:01:35,232 INFO laia.common.loader] Loaded model model
[2024-04-23 13:01:35,782 WARNING laia.data.text_image_from_text_table_dataset] No image file found for image ID '0d7cf548-742b-4067-9084-52478806091d_Line0_30af78fd-e15d-4873-91d1-69ad7c0623c3.jpg', ignoring example...
[2024-04-23 13:01:35,783 WARNING laia.data.text_image_from_text_table_dataset] No image file found for image ID '0d7cf548-742b-4067-9084-52478806091d_Line0_b1fb9275-5d49-4266-9de0-e6a93fc6dfaf.jpg', ignoring example...
[2024-04-23 13:01:35,894 INFO laia] Dataset is valid
[2024-04-23 13:01:35,894 INFO laia] Statistics written to statistics.md
```

### Example with small images

```sh
[2024-04-23 13:01:34,646 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-23 13:01:35,200 INFO laia] Installed:
[2024-04-23 13:01:35,232 INFO laia.common.loader] Loaded model model
[2024-04-23 13:01:36,052 ERROR laia] Issues found in the dataset.
[2024-04-23 13:01:36,052 ERROR laia] train - Found some images too small for convolutions (width<8). They will be padded during training.
```

### Example with variable image height

```sh
[2024-04-23 13:01:34,646 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-23 13:01:35,200 INFO laia] Installed:
[2024-04-23 13:01:35,232 INFO laia.common.loader] Loaded model model
[2024-04-23 13:01:36,052 ERROR laia] Issues found in the dataset.
[2024-04-23 13:01:36,052 ERROR laia] train - Found images with variable heights: ['/data/Esposalles/dataset/images/f6d2b699-e910-4191-bc7d-f56e60fe979a_Line2_91b43b71-ea60-4f42-a896-880676aed723.jpg'].
[2024-04-23 13:01:36,052 ERROR laia] test - Found images with variable heights: ['/data/Esposalles/dataset/images/fd1e6b3b-48cb-41c0-b1e9-2924b9562876_Line3_27e23ff1-f730-44ac-844f-479e5cc9e9aa.jpg'].
```

### Example with missing symbol

```sh
[2024-04-23 13:01:34,646 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-23 13:01:35,200 INFO laia] Installed:
[2024-04-23 13:01:35,232 INFO laia.common.loader] Loaded model model
[2024-04-23 13:01:36,052 ERROR laia] Issues found in the dataset.
[2024-04-23 13:01:36,052 ERROR laia] train - Found some unknown symbols: {'='}
[2024-04-23 13:01:36,052 ERROR laia] val - Found some unknown symbols: {'='}
[2024-04-23 13:01:36,052 ERROR laia] test - Found some unknown symbols: {'='}
```
