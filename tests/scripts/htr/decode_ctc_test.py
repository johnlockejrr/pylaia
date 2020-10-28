import shutil
from distutils.version import StrictVersion
from os.path import join

import pytest
import torch
from pytorch_lightning import seed_everything

from laia.common.saver import ModelSaver
from laia.dummies import DummyMNISTLines, DummyModel
from laia.scripts.htr import decode_ctc as script
from laia.utils import SymbolsTable
from tests.scripts.htr.conftest import call_script


@pytest.mark.parametrize("nprocs", (1, 2))
def test_decode_on_dummy_mnist_lines_data(tmpdir, nprocs):
    # prepare data
    seed_everything(0x12345)
    data_module = DummyMNISTLines(tr_n=0, va_n=5, batch_size=3, samples_per_space=3)
    data_module.prepare_data()
    # prepare model file
    model_args = [(3, 3), 12]
    ModelSaver(tmpdir).save(DummyModel, *model_args)
    # prepare ckpt file
    ckpt = tmpdir / "model.ckpt"
    torch.save(DummyModel(*model_args).state_dict(), str(ckpt))
    # prepare syms file
    syms = str(tmpdir / "syms")
    syms_table = SymbolsTable()
    for k, v in data_module.syms.items():
        syms_table.add(v, k)
    syms_table.save(syms)
    # prepare img list
    img_list = tmpdir / "img_list"
    img_list.write_text(
        "\n".join(f"va-{i}" for i in range(data_module.n["va"])), "utf-8"
    )

    args = [
        syms,
        str(img_list),
        join(data_module.root, "va"),
        f"--checkpoint={ckpt}",
        f"--train_path={tmpdir}",
        f"--batch_size={data_module.batch_size}",
    ]
    if nprocs > 1:
        args.append("--accelerator=ddp_cpu")
        args.append(f"--num_processes={nprocs}")

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    img_ids = [l.split(" ", maxsplit=1)[0] for l in stdout.strip().split("\n")]
    assert sorted(img_ids) == [f"va-{i}" for i in range(data_module.n["va"])]
    assert "Using checkpoint" in stderr


@pytest.mark.skipif(
    StrictVersion(torch.__version__) < StrictVersion("1.5.0"), reason="torch 1.4.0 bug"
)  # https://github.com/pytorch/vision/issues/1943
@pytest.mark.parametrize(
    "accelerator",
    [None, "ddp_cpu", "ddp"] if torch.cuda.device_count() > 1 else [None, "ddp_cpu"],
)
def test_decode_with_trained_ckpt_fixed_height(tmpdir, downloader, accelerator):
    syms = downloader("print/syms.txt")
    img_list = downloader("print/imgs.lst")
    ckpt = downloader("print/experiment_h128")
    images = downloader("print/imgs_h128", archive=True)
    model = downloader("print/model_h128")
    shutil.copy(model, tmpdir)

    args = [
        syms,
        img_list,
        images,
        f"--train_path={tmpdir}",
        f"--checkpoint={ckpt}",
        "--model_filename=model_h128",
        "--batch_size=3",
        "--join_str=",
        "--convert_spaces",
    ]
    if accelerator:
        args.append(f"--accelerator={accelerator}")
        args.append(f"--{'num_processes' if accelerator == 'ddp_cpu' else 'gpus'}=2")

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    lines = sorted(stdout.strip().split("\n"))
    assert lines == [
        "ONB_aze_18950706_4.r_10_2.tl_125 Deutschland.",
        "ONB_aze_18950706_4.r_10_3.tl_126 — Wir haben gestern von dem merkwürdigen Tadel",
        "ONB_aze_18950706_4.r_10_3.tl_127 erzahlt, den der Colberger Bürgermeister von dem vorgesetzten",
        "ONB_aze_18950706_4.r_10_3.tl_128 Regierungspräsidenten in Köslin erlitten hat, weil er anläß¬",
        "ONB_aze_18950706_4.r_10_3.tl_129 lich der Reichsrathswahl im Kreise Colberg=Köslin den",
    ]


@pytest.mark.skipif(
    StrictVersion(torch.__version__) < StrictVersion("1.5.0"), reason="torch 1.4.0 bug"
)  # https://github.com/pytorch/vision/issues/1943
def test_decode_with_old_trained_ckpt(tmpdir, downloader):
    syms = downloader("print/syms.txt")
    img_list = downloader("print/imgs.lst")
    ckpt = downloader("print/old_experiment")
    images = downloader("print/imgs", archive=True)
    # download and move model
    model = downloader("print/old_model")
    shutil.copy(model, tmpdir)

    args = [
        syms,
        img_list,
        images,
        f"--train_path={tmpdir}",
        f"--checkpoint={ckpt}",
        "--model_filename=old_model",
        f"--batch_size=3",
        "--join_str=",
        "--convert_spaces",
    ]
    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    lines = sorted(stdout.strip().split("\n"))
    assert lines == [
        "ONB_aze_18950706_4.r_10_2.tl_125 Deuklichland.",
        "ONB_aze_18950706_4.r_10_3.tl_126 — Wir haben gestern von dem merkwürdigen Tadel",
        "ONB_aze_18950706_4.r_10_3.tl_127 erzählt, den der Colberger Bürgermeister von dem vorgesetzten",
        "ONB_aze_18950706_4.r_10_3.tl_128 Regierungspräsidenten in Köslin erlitten hat, weil er anläß¬",
        "ONB_aze_18950706_4.r_10_3.tl_129 lich der Reicherathswahl in Kreise Colberg=Köslin den",
    ]


@pytest.mark.skipif(
    StrictVersion(torch.__version__) < StrictVersion("1.5.0"), reason="torch 1.4.0 bug"
)  # https://github.com/pytorch/vision/issues/1943
@pytest.mark.parametrize(
    "accelerator",
    [None, "ddp_cpu", "ddp"] if torch.cuda.device_count() > 1 else [None, "ddp_cpu"],
)
def test_segmentation(tmpdir, downloader, accelerator):
    syms = downloader("print/syms.txt")
    img_list = downloader("print/imgs.lst")
    ckpt = downloader("print/experiment_h128")
    images = downloader("print/imgs_h128", archive=True)
    model = downloader("print/model_h128")
    shutil.copy(model, tmpdir)

    args = [
        syms,
        img_list,
        images,
        f"--train_path={tmpdir}",
        f"--experiment_dirname={tmpdir}",
        f"--checkpoint={ckpt}",
        "--model_filename=model_h128",
        "--batch_size=3",
        "--print_segmentation=word",
    ]
    if accelerator:
        args.append(f"--accelerator={accelerator}")
        args.append(f"--{'num_processes' if accelerator == 'ddp_cpu' else 'gpus'}=2")

    stdout, stderr = call_script(script.__file__, args)
    print(f"Script stdout:\n{stdout}")
    print(f"Script stderr:\n{stderr}")

    lines = sorted(stdout.strip().split("\n"))
    expected = [
        "ONB_aze_18950706_4.r_10_2.tl_125 [('Deutschland.', 1, 1, 735, 128)]",
        "ONB_aze_18950706_4.r_10_3.tl_126 [('—', 1, 1, 23, 128),",
        "ONB_aze_18950706_4.r_10_3.tl_127 [('erzahlt,', 1, 1, 247, 128),",
        "ONB_aze_18950706_4.r_10_3.tl_128 [('Regierungspräsidenten', 1, 1, 568, 128),",
        "ONB_aze_18950706_4.r_10_3.tl_129 [('lich', 1, 1, 71, 128),",
    ]
    assert all(l.startswith(e) for l, e in zip(lines, expected))
