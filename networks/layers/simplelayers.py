import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union, cast


from utils.misc import ensure_tuple_rep
from networks.layers.convutils import gaussian_1d, same_padding


class GaussianFilter(nn.Module):
    def __init__(self, spatial_dims: int, sigma: Union[Sequence[float], float], truncated: float = 4.0) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std.
            truncated: spreads how many stds.
        """
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        _sigma = ensure_tuple_rep(sigma, self.spatial_dims)
        self.kernel = [
            torch.nn.Parameter(torch.as_tensor(gaussian_1d(s, truncated), dtype=torch.float), False) for s in _sigma
        ]
        self.padding = [cast(int, (same_padding(k.size()[0]))) for k in self.kernel]
        self.conv_n = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
        for idx, param in enumerate(self.kernel):
            self.register_parameter(f"kernel_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].

        Raises:
            TypeError: When ``x`` is not a ``torch.Tensor``.

        """
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
        chns = x.shape[1]
        sp_dim = self.spatial_dims
        x = x.clone()  # no inplace change of x

        def _conv(input_: torch.Tensor, d: int) -> torch.Tensor:
            if d < 0:
                return input_
            s = [1] * (sp_dim + 2)
            s[d + 2] = -1
            kernel = self.kernel[d].reshape(s)
            kernel = kernel.repeat([chns, 1] + [1] * sp_dim)
            padding = [0] * sp_dim
            padding[d] = self.padding[d]
            return self.conv_n(input=_conv(input_, d - 1), weight=kernel, padding=padding, groups=chns)

        return _conv(x, sp_dim - 1)
