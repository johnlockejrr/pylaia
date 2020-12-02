from logging import DEBUG, INFO

import pytest
import torch

from laia.utils import check_tensor


@pytest.mark.parametrize("raise_exception", [True, False])
def test_check_tensor(caplog, raise_exception):
    tensor = torch.tensor([1, float("inf"), 3])
    caplog.set_level(DEBUG)
    if raise_exception:
        with pytest.raises(ValueError, match=r"1 \(33.33%\) infinite values found"):
            check_tensor(tensor, raise_exception=True)
    else:
        assert check_tensor(
            tensor, raise_exception=False, msg="{foo} message", foo="test"
        )
        assert caplog.messages.count("test message") == 1


@pytest.mark.parametrize(["has_inf", "log_level"], [(True, INFO), (False, DEBUG)])
def test_check_tensor_no_action(caplog, has_inf, log_level):
    tensor = torch.tensor([1, float("inf"), 3] if has_inf else [1, 2, 3])
    with caplog.at_level(log_level):
        assert not check_tensor(tensor, msg="test message")
    assert caplog.messages.count("test message") == 0
