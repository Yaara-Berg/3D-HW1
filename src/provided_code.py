import urllib.request
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int, UInt8
from beartype.typing import List
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from trimesh.exchange.obj import load_obj
import cv2

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def download_file(url: str, path: Path) -> None:
    """Download a file from the specified URL."""
    if path.exists():
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    urllib.request.urlretrieve(url, path)


def load_mesh(
    path: Path, device: torch.device = torch.device("cpu")
) -> tuple[Float[Tensor, "vertex 3"], Int[Tensor, "face 3"]]:
    """Load a mesh."""
    with path.open("r") as f:
        mesh = load_obj(f)
        mesh_data = next(iter(mesh["geometry"].values()))
    vertices = torch.tensor(mesh_data["vertices"], dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_data["faces"], dtype=torch.int64, device=device)
    return vertices, faces


def get_bunny(
    bunny_url: str = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
    bunny_path: Path = Path("data/stanford_bunny.obj"),
    device: torch.device = torch.device("cpu"),
) -> tuple[Float[Tensor, "vertex 3"], Int[Tensor, "face 3"]]:
    download_file(bunny_url, bunny_path)
    vertices, faces = load_mesh(bunny_path, device=device)

    # Center and rescale the bunny.
    maxima, _ = vertices.max(dim=0, keepdim=True)
    minima, _ = vertices.min(dim=0, keepdim=True)
    centroid = 0.5 * (maxima + minima)
    vertices -= centroid
    vertices /= (maxima - minima).max()

    return vertices, faces


def generate_spin(
    num_steps: int,
    elevation: float,
    radius: float,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "batch 4 4"]:
    # Translate back along the camera's look vector.
    tf_translation = torch.eye(4, dtype=torch.float32, device=device)
    tf_translation[2, 3] = -radius
    tf_translation[:2, :2] *= -1  # Use +Y as world up instead of -Y.

    # Generate the transformation for the azimuth.
    t = np.linspace(0, 1, num_steps, endpoint=False)
    azimuth = [
        R.from_rotvec(np.array([0, x * 2 * np.pi, 0], dtype=np.float32)).as_matrix()
        for x in t
    ]
    azimuth = torch.tensor(np.array(azimuth), dtype=torch.float32, device=device)
    tf_azimuth = torch.eye(4, dtype=torch.float32, device=device)
    tf_azimuth = repeat(tf_azimuth, "i j -> b i j", b=num_steps).clone()
    tf_azimuth[:, :3, :3] = azimuth

    # Generate the transformation for the elevation.
    deg_elevation = np.deg2rad(elevation)
    elevation = R.from_rotvec(np.array([deg_elevation, 0, 0], dtype=np.float32))
    elevation = torch.tensor(elevation.as_matrix())
    tf_elevation = torch.eye(4, dtype=torch.float32, device=device)
    tf_elevation[:3, :3] = elevation

    return tf_azimuth @ tf_elevation @ tf_translation


def plot_point_cloud(
    vertices: Float[Tensor, "batch dim"],
    alpha: float = 0.5,
    max_points: int = 10_000,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
    zlim: tuple[float, float] = (-1.0, 1.0),
):
    """Plot a point cloud."""
    vertices = vertices.cpu()

    batch, dim = vertices.shape

    if batch > max_points:
        vertices = np.random.default_rng().choice(vertices, max_points, replace=False)
    fig = plt.figure(figsize=(6, 6))
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("z")
        ax.set_zlim(zlim)
        ax.view_init(elev=120.0, azim=270)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(*vertices.T, alpha=alpha, marker=",", lw=0.5, s=1, color="black")
    plt.show()


def save_image_subplots(images: List[np.ndarray], titles: List[str], out_path: Path):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 3))

    if len(images) == 1:
        axes = [axes]

    for idx in range(len(images)):
        axes[idx].imshow(images[idx], cmap="gray", vmin=0, vmax=1)
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def scale_intensity(image: np.ndarray) -> np.ndarray:
    v_min, v_max = image.min(), image.max()
    scaled_image = (image * 1.0 - v_min) / (v_max - v_min)
    scaled_image = scaled_image - np.mean(scaled_image) + 0.5
    scaled_image = np.clip(scaled_image, 0, 1)

    return scaled_image


def load_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    frames = np.array(frames) / 255.0

    return frames


def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (424, 240))
        frame = frame[:, 80 : 80 + 240]
        processed_frames.append(frame)

    return np.array(processed_frames)


def match_brightness_and_contrast(magnified, reference):
    out = magnified / np.max(magnified)

    for ch in range(3):
        ref_first_frame = reference[0, :, :, ch]
        mag_first_frame = out[0, :, :, ch]

        ref_std = np.std(ref_first_frame[:])
        mag_std = np.std(mag_first_frame[:])
        if mag_std == 0:
            continue

        scale = ref_std / mag_std
        mag_mean = np.mean(mag_first_frame[:])
        out[:, :, :, ch] = mag_mean + scale * (out[:, :, :, ch] - mag_mean)

    return np.clip(out, 0, 1)


def save_video(frames: np.ndarray, path: Path, fps: int = 30) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)

    frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    height, width = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter(str(path), fourcc, fps, (height, width))

    for frame in frames:
        out.write(frame)

    out.release()
