import torch
from jaxtyping import Float
from torch import Tensor

from src.geometry import homogenize_points, project, transform_world2cam


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    height, width = resolution
    batch = extrinsics.shape[0]

    # Homogenize and broadcast vertices over the camera batch: (1, V, 4)
    verts_h = homogenize_points(vertices).unsqueeze(0)

    # Transform world → camera for all cameras: (B, V, 4)
    cam_verts = transform_world2cam(verts_h, extrinsics.unsqueeze(1))

    # Project to normalized pixel coords: (B, V, 2)
    pixel_coords = project(cam_verts, intrinsics.unsqueeze(1))

    depths = cam_verts[..., 2]  # (B, V)
    u = (pixel_coords[..., 0] * width).long()   # (B, V)
    v = (pixel_coords[..., 1] * height).long()  # (B, V)

    # White canvas; paint valid projected points black
    canvas = torch.ones(batch, height, width, dtype=vertices.dtype, device=vertices.device)
    valid = (depths > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    for b in range(batch):
        m = valid[b]
        canvas[b, v[b][m], u[b][m]] = 0.0

    return canvas
