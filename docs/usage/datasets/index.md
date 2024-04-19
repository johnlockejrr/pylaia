# Dataset

PyLaia datasets must be formatted following a specific format. Learn how to build a dataset by following this [page](./format.md).

Once the dataset is created, you may use the `pylaia-htr-dataset` command to compute statistics and make sure your dataset is valid. To know more about the options of this command, use `pylaia-htr-dataset --help`.



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
pylaia-htr-dataset  /data/Esposalles/dataset/syms.txt \
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
[2024-04-18 10:30:39,352 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-18 10:30:39,913 INFO laia] Installed:
[2024-04-18 10:30:39,943 INFO laia.common.loader] Loaded model model
[2024-04-18 10:30:40,415 INFO laia] Dataset is valid
[2024-04-18 10:30:40,416 INFO laia] Statistics written to statistics.md
```

### Example with a YAML configuration file

Run the following command to validate a dataset:
```sh
pylaia-htr-dataset --config config_dataset.yaml
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
[2024-04-18 10:30:39,352 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-18 10:30:39,913 INFO laia] Installed:
[2024-04-18 10:30:39,943 INFO laia.common.loader] Loaded model model
[2024-04-18 10:30:40,415 INFO laia] Dataset is valid
[2024-04-18 10:30:40,416 INFO laia] Statistics written to statistics.md
```

### Example with missing images

```bash
[2024-04-18 10:30:39,352 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-18 10:30:39,913 INFO laia] Installed:
[2024-04-18 10:30:39,943 INFO laia.common.loader] Loaded model model
[2024-04-18 10:30:40,316 WARNING laia.data.text_image_from_text_table_dataset] No image file found for image ID '0d7cf548-742b-4067-9084-52478806091d_Line0_30af78fd-e15d-4873-91d1-69ad7c0623c3.jpg', ignoring example...
[2024-04-18 10:30:40,318 WARNING laia.data.text_image_from_text_table_dataset] No image file found for image ID '0d7cf548-742b-4067-9084-52478806091d_Line0_b1fb9275-5d49-4266-9de0-e6a93fc6dfaf.jpg', ignoring example...
[2024-04-18 10:30:40,415 INFO laia] Dataset is valid
[2024-04-18 10:30:40,416 INFO laia] Statistics written to statistics.md
```

### Example with small images

```sh
[2024-04-18 11:24:13,352 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-18 11:24:13,977 INFO laia] Installed:
[2024-04-18 11:24:14,007 INFO laia.common.loader] Loaded model model
[2024-04-18 11:24:14,081 WARNING laia] Found some images too small for convolutions (width<8). They will be padded during training.
[2024-04-18 11:24:14,182 INFO laia] Dataset is correct
[2024-04-18 11:24:14,182 INFO laia] Statistics written to statistics.md
```

### Example with variable image height

```sh
[2024-04-18 11:32:07,852 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-18 11:32:08,185 INFO laia] Installed:
[2024-04-18 11:32:08,216 INFO laia.common.loader] Loaded model model
[2024-04-18 11:32:08,319 CRITICAL laia] Uncaught exception:
Traceback (most recent call last):
  File "/usr/bin/pylaia-htr-dataset", line 8, in <module>
    sys.exit(main())
  File "/usr/lib/python3.10/site-packages/laia/scripts/htr/dataset_check.py", line 149, in main
    run(**args)
  File "/home/solene/miniconda3/envs/python3.10/lib/python3.10/site-packages/laia/scripts/htr/dataset_check.py", line 56, in run
    assert dataset_stats.is_fixed_height, f"Found images with variable heights in {split} set: {dataset_stats.get_invalid_images_height(fixed_input_height)}."
AssertionError: Found images with variable heights in train set: ['../../Esposalles/dataset/images/f6d2b699-e910-4191-bc7d-f56e60fe979a_Line2_91b43b71-ea60-4f42-a896-880676aed723.jpg'].
```

### Example with missing symbol

```sh
[2024-04-18 11:27:46,352 INFO laia] Arguments: {'syms': '/data/Esposalles/dataset/syms.txt', 'img_dirs': ['/data/Esposalles/dataset/images/'], 'tr_txt_table': '/data/Esposalles/dataset/train.txt', 'va_txt_table': '/data/Esposalles/dataset/val.txt', 'te_txt_table': '/data/Esposalles/dataset/test.txt', 'fixed_input_height': 128, 'common': CommonArgs(seed=74565, train_path='', model_filename='model', experiment_dirname='experiment-esposalles', monitor=<Monitor.va_cer: 'va_cer'>, checkpoint=None), 'train': TrainArgs(delimiters=['<space>'], checkpoint_k=3, resume=False, early_stopping_patience=80, gpu_stats=False, augment_training=True)}
[2024-04-18 11:27:46,870 INFO laia] Installed:
[2024-04-18 11:27:46,904 INFO laia.common.loader] Loaded model model
[2024-04-18 11:27:47,060 CRITICAL laia] Uncaught exception:
Traceback (most recent call last):
  File "/home/solene/miniconda3/envs/python3.10/bin/pylaia-htr-dataset", line 8, in <module>
    sys.exit(main())
  File "/home/solene/miniconda3/envs/python3.10/lib/python3.10/site-packages/laia/scripts/htr/dataset_check.py", line 149, in main
    run(**args)
  File "/home/solene/miniconda3/envs/python3.10/lib/python3.10/site-packages/laia/scripts/htr/dataset_check.py", line 61, in run
    char in syms
AssertionError: The character "=" is not in the symbols file but appears in the train set
```
