import numpy as np
from typing import Sequence, Union


def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1):
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def gaussian_1d(sigma, truncated=4.0):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    tail = int(sigma * truncated + 0.5)
    sigma2 = sigma * sigma
    x = np.arange(-tail, tail + 1)
    out = np.exp(-0.5 / sigma2 * x ** 2)
    out /= out.sum()
    return out
