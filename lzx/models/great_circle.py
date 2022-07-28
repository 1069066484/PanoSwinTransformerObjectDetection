import torch
import math
pi = math.pi


"""
Washington to Beijing
uv1 = torch.tensor([-77/ 180 * pi, 39 / 180 * pi])
# uv1 = torch.tensor([121.489 / 180 * pi, 31.225 / 180 * pi])
uv2 = torch.tensor([116.4 / 180 * pi, 39.9 / 180 * pi])
ret = torch.arccos(
    torch.cos(uv1[1]) * torch.cos(uv2[1]) * torch.cos(uv2[0] - uv1[0]) +
    torch.sin(uv1[1]) * torch.sin(uv2[1])
)
print(ret / 2 / pi * 40076)
"""

def great_circle_pairwise(uv1, uv2):
    """
    uv1: B x 2
    uv2: B x 2
    return B

    we assume that the sphere radius is 1
    u: [0, 2pi],
    v: [-0.5pi, 0.5pi]
    AB = arccos( cos(v1)cos(v2)cos(u1 - u2) + sin(v1)sin(v2))
    """
    return torch.arccos(
        torch.cos(uv1[:, 1]) * torch.cos(uv2[:, 1]) * torch.cos(uv2[:, 0] - uv1[:, 0]) +
        torch.sin(uv1[:, 1]) * torch.sin(uv2[:, 1])
    )


def great_circle22(uv1, uv2):
    """
    uv1: B x 2
    uv2: B x 2
    return B x B

    we assume that the sphere radius is 1
    u: [0, 2pi],
    v: [-0.5pi, 0.5pi]
    AB = arccos( cos(v1)cos(v2)cos(u1 - u2) + sin(v1)sin(v2))
    """
    return torch.arccos(
        torch.cos(uv1[:, 1][:, None]) * torch.cos(uv2[:, 1][None, :]) *
        torch.cos(uv1[:, 0][:, None] - uv2[:, 0][None, :]) +
        torch.sin(uv1[:, 1][:, None]) * torch.sin(uv2[:, 1][None, :])
    )


def haversine_pairwise(uv1, uv2):
    """
    uv1: B x 2
    uv2: B x 2
    return B

    we assume that the sphere radius is 1
    u: [0, 2pi],
    v: [-0.5pi, 0.5pi]
    AB = arccos( cos(v1)cos(v2)cos(u1 - u2) + sin(v1)sin(v2))
    """
    return torch.arcsin(
        (torch.sin(0.5 * torch.abs(uv2[:, 1] - uv1[:, 1])) ** 2 + \
           torch.cos(uv2[:, 1]) * torch.cos(uv1[:, 1]) * torch.sin(0.5 * (uv2[:, 0] - uv1[:, 0])) ** 2) ** 0.5
    ) * 2



def haversine22(uv1, uv2):
    """
    uv1: B x 2
    uv2: B x 2
    return B x B

    we assume that the sphere radius is 1
    u: [0, 2pi],
    v: [-0.5pi, 0.5pi]
    AB = arccos( cos(v1)cos(v2)cos(u1 - u2) + sin(v1)sin(v2))
    """
    return torch.arcsin(
        (torch.sin(0.5 * torch.abs(uv2[..., 1][..., None, :] - uv1[..., 1][..., None])) ** 2 + \
        torch.cos(uv2[..., 1][..., None, :]) * torch.cos(uv1[..., 1][..., None]) *
         torch.sin(0.5 * (uv2[..., 0][..., None, :] - uv1[..., 0][..., None])) ** 2) ** 0.5
    ) * 2


def haversine22_approx(uv1, uv2):
    """
    uv1: B x 2
    uv2: B x 2
    return B x B

    we assume that the sphere radius is 1
    u: [0, 2pi],
    v: [-0.5pi, 0.5pi]
    AB = arccos( cos(v1)cos(v2)cos(u1 - u2) + sin(v1)sin(v2))
    """
    return (
        (torch.sin(0.5 * torch.abs(uv2[..., 1][..., None, :] - uv1[..., 1][..., None])) ** 2 + \
        torch.cos(uv2[..., 1][..., None, :]) * torch.cos(uv1[..., 1][..., None]) *
         torch.sin(0.5 * (uv2[..., 0][..., None, :] - uv1[..., 0][..., None])) ** 2) ** 0.5
    ) * 2



def _test():
    earth_radius = 6400
    uv1 = torch.tensor([[-77, 39],         # Washington
                        [121.489, 31.225]  # Shanghai
                        ]) / 180 * pi

    uv2 = torch.tensor([[116.4, 39.9]]*2)  / 180 * pi  # Peking
    print(great_circle_pairwise(uv1, uv2) * earth_radius)
    print(haversine_pairwise(uv1, uv2) * earth_radius)
    print(great_circle22(uv1, uv2) * earth_radius)
    print(haversine22(uv1, uv2) * earth_radius)


if __name__=="__main__":
    _test()


