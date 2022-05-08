import math
import torch
import torch.nn.functional as F


def k0(t):
    """
    Computes the zeroth order arc-cosine kernel.

    Args:
        t: cosines
    """

    k0 = (math.pi - torch.arccos(torch.clamp(t, min=-1.0, max=1.0))) / math.pi

    return k0


def k1(t):

    """
    Computes the first order arc-cosine kernel.

    Args:
        t: cosines
    """

    t = torch.clamp(t, min=-1.0, max=1.0)
    k1 = (torch.sqrt(1.0 - t.pow(2)) + (math.pi - torch.arccos(t)) * t) / math.pi

    return k1


def dntk(x, y, depth, normalize=False):
    """
    Computes the NTK of a (deep) FCN without bias.

    Args:
        x: first input
        y: second input
        depth: network depth
        normalize: If set to True normalise the inputs on a sphere. 
            Default = False
    """

    if normalize:

        x /= torch.norm(x, dim=-1, keepdim=True)
        y /= torch.norm(y, dim=-1, keepdim=True)

    cosines = torch.matmul(x, y.t())

    ntk = cosines

    for i in range(depth):

        rfk = k1(cosines)
        ntk = (k0(cosines) * ntk) + rfk
        cosines = rfk

    return ntk


def dcntk(x, y, filtersizes, normalize=False):
    """
    Computes the NTK of a 1d hierarchical (L)CNN with non-overlapping patches.

    Args:
        x: first input
        y: second input
        filtersizes: list with filter sizes starting from first layet
        normalize: If set to True normalise the input patches on a sphere. 
            Default = False
    """

    d = x.size(-1)
    assert y.size(-1) == d, "The inputs must have the same dimension"

    prod = torch.tensor(filtersizes).prod().item()
    assert d % prod == 0, "The filter sizes are incompatible with input dimension"

    temp_x = [x.size(0)]
    temp_y = [y.size(0)]

    xpatch = F.unfold(
        x.reshape(torch.tensor(temp_x).prod().item(), 1, 1, -1),
        kernel_size=(1, prod),
        dilation=1,
        padding=0,
        stride=prod,
    ).transpose(1, 2)

    ypatch = F.unfold(
        y.reshape(torch.tensor(temp_y).prod().item(), 1, 1, -1),
        kernel_size=(1, prod),
        dilation=1,
        padding=0,
        stride=prod,
    ).transpose(1, 2)

    temp_x.append(d // prod)
    temp_y.append(d // prod)

    for filtersize in filtersizes[1:][::-1]:

        prod //= filtersize

        xpatch = F.unfold(
            xpatch.reshape(torch.tensor(temp_x).prod().item(), 1, 1, -1),
            kernel_size=(1, prod),
            dilation=1,
            padding=0,
            stride=prod,
        ).transpose(1, 2)

        ypatch = F.unfold(
            ypatch.reshape(torch.tensor(temp_y).prod().item(), 1, 1, -1),
            kernel_size=(1, prod),
            dilation=1,
            padding=0,
            stride=prod,
        ).transpose(1, 2)

        temp_x.append(filtersize)
        temp_y.append(filtersize)

        xpatch = xpatch.reshape(*temp_x, prod)
        ypatch = ypatch.reshape(*temp_y, prod)

    if normalize:

        xpatch /= torch.norm(xpatch, dim=-1, keepdim=True)
        ypatch /= torch.norm(ypatch, dim=-1, keepdim=True)

    cosines = torch.matmul(
        xpatch.permute(
            *[i + 1 for i in range(len(filtersizes))], 0, len(filtersizes) + 1
        ),
        ypatch.permute(
            *[i + 1 for i in range(len(filtersizes))], len(filtersizes) + 1, 0
        ),
    )

    ntk = cosines

    for _ in range(len(filtersizes)):

        rfk = k1(cosines).mean(dim=-3)
        ntk = (k0(cosines) * ntk).mean(dim=-3) + rfk
        cosines = rfk

    return ntk


def dcrfk(x, y, filtersizes, normalize=False):
    """
    Computes the RFK (of NNGP kernel) of a 1d hierarchical (L)CNN with non-overlapping patches.

    Args:
        x: first input
        y: second input
        filtersizes: list with filter sizes starting from first layet
        normalize: If set to True normalise the input patches on a sphere. 
            Default = False
    """

    d = x.size(-1)
    assert y.size(-1) == d, "The inputs must have the same dimension"

    prod = torch.tensor(filtersizes).prod().item()
    assert d % prod == 0, "The filter sizes are incompatible with input dimension"

    temp_x = [x.size(0)]
    temp_y = [y.size(0)]

    xpatch = F.unfold(
        x.reshape(torch.tensor(temp_x).prod().item(), 1, 1, -1),
        kernel_size=(1, prod),
        dilation=1,
        padding=0,
        stride=prod,
    ).transpose(1, 2)

    ypatch = F.unfold(
        y.reshape(torch.tensor(temp_y).prod().item(), 1, 1, -1),
        kernel_size=(1, prod),
        dilation=1,
        padding=0,
        stride=prod,
    ).transpose(1, 2)

    temp_x.append(d // prod)
    temp_y.append(d // prod)

    for filtersize in filtersizes[1:][::-1]:

        prod //= filtersize

        xpatch = F.unfold(
            xpatch.reshape(torch.tensor(temp_x).prod().item(), 1, 1, -1),
            kernel_size=(1, prod),
            dilation=1,
            padding=0,
            stride=prod,
        ).transpose(1, 2)

        ypatch = F.unfold(
            ypatch.reshape(torch.tensor(temp_y).prod().item(), 1, 1, -1),
            kernel_size=(1, prod),
            dilation=1,
            padding=0,
            stride=prod,
        ).transpose(1, 2)

        temp_x.append(filtersize)
        temp_y.append(filtersize)

        xpatch = xpatch.reshape(*temp_x, prod)
        ypatch = ypatch.reshape(*temp_y, prod)

    if normalize:

        xpatch /= torch.norm(xpatch, dim=-1, keepdim=True)
        ypatch /= torch.norm(ypatch, dim=-1, keepdim=True)

    cosines = torch.matmul(
        xpatch.permute(
            *[i + 1 for i in range(len(filtersizes))], 0, len(filtersizes) + 1
        ),
        ypatch.permute(
            *[i + 1 for i in range(len(filtersizes))], len(filtersizes) + 1, 0
        ),
    )

    for _ in range(len(filtersizes)):

        rfk = k1(cosines).mean(dim=-3)
        cosines = rfk

    return rfk


def power_ntk(ntk, q):
    """
    Computes the power of an NTK matrix elementwise.

    Args:
        ntk: ntk matrix
        q: exponent
    """

    return ntk.pow(q)
