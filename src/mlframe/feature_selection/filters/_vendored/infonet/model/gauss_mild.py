"""Vendored (third-party) InfoNet component: a fixed (non-trainable) Gaussian-blur convolution built from a
discretized Gaussian CDF kernel, replicated per channel and applied as a depthwise ``conv2d``."""
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F

def gauss_kernel(kernlen=5, nsig=3, channels=1):
    """Build a ``(1, 1, kernlen, kernlen)`` 2D Gaussian kernel (outer product of a 1D CDF-difference kernel with
    itself, normalized to sum 1), replicated to ``channels`` along the input-channel axis, as a torch tensor."""
    interval = (2 * nsig + 1.0) / (kernlen)
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return torch.from_numpy(out_filter)

def make_gauss_var(size, nsig, c_i):
    """Wrap ``gauss_kernel(size, nsig, c_i)`` in a non-trainable ``torch.nn.Parameter`` (``requires_grad=False``)."""
    kernel = gauss_kernel(size, nsig, c_i)
    var = torch.nn.Parameter(kernel, requires_grad=False)
    return var

class GaussConv(nn.Module):
    """Fixed depthwise Gaussian-blur convolution module (non-trainable kernel from :func:`gauss_kernel`)."""

    def __init__(self, size, nsig, channels, padding="same"):
        super().__init__()
        self.padding = padding
        self.kernel = make_gauss_var(size, nsig, channels)

    def forward(self, img):
        """Apply the fixed Gaussian kernel as a grouped (depthwise, ``groups=channels``) 2D convolution to ``img``."""
        c_i = img.shape[1]

        if self.padding == "same":
            padding = self.kernel.shape[2] // 2
        else:
            padding = 0
        return F.conv2d(img, self.kernel, padding=padding, stride=1, groups=c_i)

# Sample usage
if __name__ == "__main__":
    img = torch.randn(1, 3, 10, 10)  # Random image with shape (batch_size, channels, height, width)
    gauss_conv = GaussConv(5, 3, img.shape[1])
    output = gauss_conv(img)
    print("Output shape:", output.shape)
