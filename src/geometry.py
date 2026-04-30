import torch
from jaxtyping import Float
from torch import Tensor


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    return torch.cat([points, torch.zeros_like(points[..., :1])], dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    return (transform @ xyz.unsqueeze(-1)).squeeze(-1)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    return transform_rigid(xyz, torch.linalg.inv(cam2world))


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    return transform_rigid(xyz, cam2world)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    # Perspective divide: normalize by z
    z = xyz[..., 2:3]
    xy_norm = xyz[..., :2] / z
    ones = torch.ones_like(xy_norm[..., :1])
    xy_h = torch.cat([xy_norm, ones], dim=-1)  # (*batch, 3)

    # Apply normalized intrinsics → result is in [0, 1] pixel space
    uv = (intrinsics @ xy_h.unsqueeze(-1)).squeeze(-1)  # (*batch, 3)
    return uv[..., :2]
