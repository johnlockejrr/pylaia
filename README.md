<div align="center">

# PyLaia

**PyLaia is a device agnostic, PyTorch based, deep learning toolkit for handwritten document analysis.**

**It is also a successor to [Laia](https://github.com/jpuigcerver/Laia).**

[![pipeline status](https://gitlab.teklia.com/atr/pylaia/badges/master/pipeline.svg)](https://gitlab.teklia.com/atr/pylaia/-/commits/master)
[![Coverage](https://gitlab.teklia.com/atr/pylaia/badges/master/coverage.svg)](https://gitlab.teklia.com/atr/pylaia/-/commits/master)
[![Code quality](https://img.shields.io/codefactor/grade/github/jpuigcerver/PyLaia?&label=CodeFactor&logo=CodeFactor&labelColor=2782f7)](https://www.codefactor.io/repository/github/jpuigcerver/PyLaia)

[![Python: 3.9 | 3.10](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)](https://www.python.org/)
[![PyTorch: 1.13.0 | 1.13.1](https://img.shields.io/badge/PyTorch-1.13.0%20%7C%201.13.1-8628d5.svg?&logo=PyTorch&logoColor=white&labelColor=%23ee4c2c)](https://pytorch.org/)
[![pre-commit: enabled](https://img.shields.io/badge/pre--commit-enabled-76877c?&logo=pre-commit&labelColor=1f2d23)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?)](https://github.com/ambv/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

Get started by having a look at our [Documentation](https://atr.pages.teklia.com/pylaia)!

## Installation

To install PyLaia from PyPi:

```bash
pip install pylaia
```

The following Python scripts will be installed in your system:

- [`pylaia-htr-create-model`](laia/scripts/htr/create_model.py): Create a VGG-like model with BLSTMs on top for handwriting text recognition. The script has different options to customize the model. The architecture is based on the paper ["Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?"](https://ieeexplore.ieee.org/document/8269951) (2017) by J. Puigcerver.
- [`pylaia-htr-train-ctc`](laia/scripts/htr/train_ctc.py): Train a model using the CTC algorithm and a set of text-line images and their transcripts.
- [`pylaia-htr-decode-ctc`](laia/scripts/htr/decode_ctc.py): Decode text line images using a trained model and the CTC algorithm. It can also output the char/word segmentation boundaries of the symbols recognized.
- [`pylaia-htr-netout`](laia/scripts/htr/netout.py): Dump the output of the model for a set of text-line images in order to decode using an external language model.

## Contributing

If you want to contribute new feature or found a text that is incorrectly segmented using pySBD, then please head to [CONTRIBUTING.md](https://gitlab.teklia.com/atr/pylaia/-/blob/master/CONTRIBUTING.md) to know more and follow these steps.

1.  Fork it ( <https://gitlab.teklia.com/atr/pylaia/-/forks/new> )
2.  Create your feature branch (`git checkout -b my-new-feature`)
3.  Commit your changes (`git commit -am 'Add some feature'`)
4.  Push to the branch (`git push origin my-new-feature`)
5.  Create a new Merge Request ( <https://gitlab.teklia.com/atr/pylaia/-/merge_requests/new> )

### Code of conduct

We are committed to providing a friendly, safe and welcoming environment for all. Please read and
respect the [PyLaia Code of Conduct](https://gitlab.teklia.com/atr/pylaia/-/blob/master/CODE_OF_CONDUCT.md).

## Acknowledgments

Work in this toolkit was financially supported by the [Pattern Recognition and Human Language Technology (PRHLT) Research Center](https://www.prhlt.upv.es/).

## Citation

* Article describing the latest contributions to PyLaia

```bib
@inproceedings{pylaia2024,
    author = "Tarride, Solène and Schneider, Yoann and Generali, Marie and Boillet, Melodie and Abadie, Bastien and Kermorvant, Christopher",
    title = "Improving Automatic Text Recognition with Language Models in the PyLaia Open-Source Library",
    booktitle = "Submitted at ICDAR",
    year = "2024"
}
```

* Original article

```bib
@inproceedings{laia2017,
  author={Puigcerver, Joan},
  booktitle={2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR)},
  title={Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?},
  year={2017},
  volume={01},
  number={},
  pages={67-72},
  doi={10.1109/ICDAR.2017.20}}
```

* GitLab repository

```bib
@software{pylaia-teklia,
  author = {Teklia},
  title = {PyLaia},
  year = {2022},
  url = {https://gitlab.teklia.com/atr/pylaia/},
  version = {1.1.0},
  note = {commit SHA}
}
```

* GitHub repository

```bib
@misc{puigcerver2018pylaia,
  author = {Joan Puigcerver and Carlos Mocholí},
  title = {PyLaia},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jpuigcerver/PyLaia/}},
  commit = {commit SHA}
}
```

## Contact

🆘 Have a question about PyLaia? Please contact us on [support.teklia.com](https://support.teklia.com/c/machine-learning/pylaia/13).
