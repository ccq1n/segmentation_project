# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from config.deviceconfig import get_torch_version_tuple
from transforms.compose import Randomizable, Transform
from transforms.croppad.array import CenterSpatialCrop
from transforms.utils import create_grid, create_control_grid, create_rotate, create_scale, create_shear, create_translate

from utils.module import optional_import
from utils.enums import GridSampleMode, GridSamplePadMode, InterpolateMode, NumpyPadMode
from utils.misc import ensure_tuple, ensure_tuple_size, ensure_tuple_rep, fall_back_tuple


nib, _ = optional_import("nibabel")

_torch_interp: Callable[..., torch.Tensor]

if get_torch_version_tuple() >= (1, 5):
    # additional argument since torch 1.5 (to avoid warnings)
    def _torch_interp(**kwargs):
        return torch.nn.functional.interpolate(recompute_scale_factor=True, **kwargs)


else:
    _torch_interp = torch.nn.functional.interpolate


class Rotate90(Transform):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    """

    def __init__(self, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.k = k
        self.spatial_axes = spatial_axes

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        rotated = list()
        for channel in img:
            rotated.append(np.rot90(channel, self.k, self.spatial_axes))
        return np.stack(rotated).astype(img.dtype)


class RandRotate90(Randomizable, Transform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(self, prob: float = 0.1, max_k: int = 3, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate90(self._rand_k, self.spatial_axes)
        return rotator(img)


class Flip(Transform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
    """

    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]]) -> None:
        self.spatial_axis = spatial_axis

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        flipped = list()
        for channel in img:
            flipped.append(np.flip(channel, self.spatial_axis))
        return np.stack(flipped).astype(img.dtype)


class RandFlip(Randomizable, Transform):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, prob: float = 0.1, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        self.prob = prob
        self.flipper = Flip(spatial_axis=spatial_axis)
        self._do_transform = False

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        self.randomize()
        if not self._do_transform:
            return img
        return self.flipper(img)


class RandDeformGrid(Randomizable, Transform):
    """
    Generate random deformation grid.
    """

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        magnitude_range: Tuple[float, float],
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            spacing: spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range: the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range

        self.rand_mag = 1.0
        self.as_tensor_output = as_tensor_output
        self.random_offset = 0.0
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        self.random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size))).astype(np.float32)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing)
        self.randomize(control_grid.shape[1:])
        control_grid[: len(spatial_size)] += self.rand_mag * self.random_offset
        if self.as_tensor_output:
            control_grid = torch.as_tensor(np.ascontiguousarray(control_grid), device=self.device)
        return control_grid


class AffineGrid(Transform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_range: angle range in radians. rotate_range[0] with be used to generate the 1st rotation
            parameter from `uniform[-rotate_range[0], rotate_range[0])`. Similarly, `rotate_range[1]` and
            `rotate_range[2]` are used in 3D affine for the range of 2nd and 3rd axes.
        shear_range: shear_range[0] with be used to generate the 1st shearing parameter from
            `uniform[-shear_range[0], shear_range[0])`. Similarly, `shear_range[1]` to
            `shear_range[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        translate_range : translate_range[0] with be used to generate the 1st shift parameter from
            `uniform[-translate_range[0], translate_range[0])`. Similarly, `translate_range[1]`
            to `translate_range[N]` controls the range of the uniform distribution used to generate
            the 2nd to N-th parameter.
        scale_range: scaling_range[0] with be used to generate the 1st scaling factor from
            `uniform[-scale_range[0], scale_range[0]) + 1.0`. Similarly, `scale_range[1]` to
            `scale_range[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        as_tensor_output: whether to output tensor instead of numpy array.
            defaults to True.
        device: device to store the output grid data.

    """

    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params

        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")

        spatial_dims = len(grid.shape) - 1
        affine = np.eye(spatial_dims + 1)
        if self.rotate_params:
            affine = affine @ create_rotate(spatial_dims, self.rotate_params)
        if self.shear_params:
            affine = affine @ create_shear(spatial_dims, self.shear_params)
        if self.translate_params:
            affine = affine @ create_translate(spatial_dims, self.translate_params)
        if self.scale_params:
            affine = affine @ create_scale(spatial_dims, self.scale_params)
        affine = torch.as_tensor(np.ascontiguousarray(affine), device=self.device)

        grid = torch.tensor(grid) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            grid = grid.to(self.device)
        grid = (affine.float() @ grid.reshape((grid.shape[0], -1)).float()).reshape([-1] + list(grid.shape[1:]))
        if self.as_tensor_output:
            return grid
        return grid.cpu().numpy()


class RandAffineGrid(Randomizable, Transform):
    """
    Generate randomised affine grid.
    """

    def __init__(
        self,
        rotate_range: Optional[Union[Sequence[float], float]] = None,
        shear_range: Optional[Union[Sequence[float], float]] = None,
        translate_range: Optional[Union[Sequence[float], float]] = None,
        scale_range: Optional[Union[Sequence[float], float]] = None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. rotate_range[0] with be used to generate the 1st rotation
                parameter from `uniform[-rotate_range[0], rotate_range[0])`. Similarly, `rotate_range[1]` and
                `rotate_range[2]` are used in 3D affine for the range of 2nd and 3rd axes.
            shear_range: shear_range[0] with be used to generate the 1st shearing parameter from
                `uniform[-shear_range[0], shear_range[0])`. Similarly, `shear_range[1]` to
                `shear_range[N]` controls the range of the uniform distribution used to generate the 2nd to
                N-th parameter.
            translate_range : translate_range[0] with be used to generate the 1st shift parameter from
                `uniform[-translate_range[0], translate_range[0])`. Similarly, `translate_range[1]`
                to `translate_range[N]` controls the range of the uniform distribution used to generate
                the 2nd to N-th parameter.
            scale_range: scaling_range[0] with be used to generate the 1st scaling factor from
                `uniform[-scale_range[0], scale_range[0]) + 1.0`. Similarly, `scale_range[1]` to
                `scale_range[N]` controls the range of the uniform distribution used to generate the 2nd to
                N-th parameter.
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`
        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: Optional[List[float]] = None
        self.shear_params: Optional[List[float]] = None
        self.translate_params: Optional[List[float]] = None
        self.scale_params: Optional[List[float]] = None

        self.as_tensor_output = as_tensor_output
        self.device = device

    def randomize(self, data: Optional[Any] = None) -> None:
        if self.rotate_range:
            self.rotate_params = [self.R.uniform(-f, f) for f in self.rotate_range if f is not None]
        if self.shear_range:
            self.shear_params = [self.R.uniform(-f, f) for f in self.shear_range if f is not None]
        if self.translate_range:
            self.translate_params = [self.R.uniform(-f, f) for f in self.translate_range if f is not None]
        if self.scale_range:
            self.scale_params = [self.R.uniform(-f, f) + 1.0 for f in self.scale_range if f is not None]

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        self.randomize()
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            as_tensor_output=self.as_tensor_output,
            device=self.device,
        )
        return affine_grid(spatial_size, grid)


class Resample(Transform):
    def __init__(
        self,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: whether to return a torch tensor. Defaults to False.
            device: device on which the tensor will be allocated.
        """
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        grid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """

        if not torch.is_tensor(img):
            img = torch.as_tensor(np.ascontiguousarray(img))
        assert grid is not None, "Error, grid argument must be supplied as an ndarray or tensor "
        grid = torch.tensor(grid) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            img = img.to(self.device)
            grid = grid.to(self.device)

        for i, dim in enumerate(img.shape[1:]):
            grid[i] = 2.0 * grid[i] / (dim - 1.0)
        grid = grid[:-1] / grid[-1:]
        index_ordering: List[int] = list(range(img.ndimension() - 2, -1, -1))
        grid = grid[index_ordering]
        grid = grid.permute(list(range(grid.ndimension()))[1:] + [0])
        out = torch.nn.functional.grid_sample(
            img.unsqueeze(0).float(),
            grid.unsqueeze(0).float(),
            mode=self.mode.value if mode is None else GridSampleMode(mode).value,
            padding_mode=self.padding_mode.value if padding_mode is None else GridSamplePadMode(padding_mode).value,
            align_corners=True,
        )[0]
        if self.as_tensor_output:
            return out
        return out.cpu().numpy()


class Rand2DElastic(Randomizable, Transform):
    """
    Random elastic deformation and affine in 2D
    """

    def __init__(
        self,
        spacing: Union[Tuple[float, float], float],
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[float], float]] = None,
        shear_range: Optional[Union[Sequence[float], float]] = None,
        translate_range: Optional[Union[Sequence[float], float]] = None,
        scale_range: Optional[Union[Sequence[float], float]] = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            spacing : distance in between the control points.
            magnitude_range: the random offsets will be generated from ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. rotate_range[0] with be used to generate the 1st rotation
                parameter from `uniform[-rotate_range[0], rotate_range[0])`.
            shear_range: shear_range[0] with be used to generate the 1st shearing parameter from
                `uniform[-shear_range[0], shear_range[0])`. Similarly, `shear_range[1]` controls
                the range of the uniform distribution used to generate the 2nd parameter.
            translate_range : translate_range[0] with be used to generate the 1st shift parameter from
                `uniform[-translate_range[0], translate_range[0])`. Similarly, `translate_range[1]` controls
                the range of the uniform distribution used to generate the 2nd parameter.
            scale_range: scaling_range[0] with be used to generate the 1st scaling factor from
                `uniform[-scale_range[0], scale_range[0]) + 1.0`. Similarly, `scale_range[1]` controls
                the range of the uniform distribution used to generate the 2nd parameter.
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        self.deform_grid = RandDeformGrid(
            spacing=spacing, magnitude_range=magnitude_range, as_tensor_output=True, device=device
        )
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            as_tensor_output=True,
            device=device,
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)

        self.spatial_size = spatial_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.prob = prob
        self.do_transform = False

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Rand2DElastic":
        self.deform_grid.set_random_state(seed, state)
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, spatial_size: Sequence[int]) -> None:
        self.do_transform = self.R.rand() < self.prob
        self.deform_grid.randomize(spatial_size)
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W),
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        self.randomize(spatial_size=sp_size)
        if self.do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = self.rand_affine_grid(grid=grid)
            grid = _torch_interp(
                input=grid.unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = create_grid(spatial_size=sp_size)
        return self.resampler(img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)
