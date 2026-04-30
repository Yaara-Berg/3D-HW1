from pathlib import Path
import json
from typing import Literal, TypedDict

from jaxtyping import Float
from PIL import Image
import torch
from torch import Tensor


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""
    with (path / "metadata.json").open("r") as f:
        metadata = json.load(f)

    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    intrinsics = torch.tensor(metadata["intrinsics"], dtype=torch.float32)

    # Load grayscale images if present (some distributed puzzle copies only ship metadata).
    image_paths = sorted(list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg")))
    if image_paths:
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("L")
            images.append(torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image.height, image.width) / 255.0)
        stacked_images = torch.stack(images, dim=0)
    else:
        stacked_images = torch.empty((0, 0, 0), dtype=torch.float32)

    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "images": stacked_images,
    }


def _axis_name(v: Tensor) -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    axis = int(torch.argmax(torch.abs(v)).item())
    sign = "+" if float(v[axis]) > 0 else "-"
    names = ["x", "y", "z"]
    return f"{sign}{names[axis]}"  # type: ignore[return-value]


def _all_signed_permutation_mats(device: torch.device) -> list[Tensor]:
    mats: list[Tensor] = []
    perms = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    signs = (-1.0, 1.0)
    for p in perms:
        for sx in signs:
            for sy in signs:
                for sz in signs:
                    m = torch.zeros((3, 3), dtype=torch.float32, device=device)
                    m[p[0], 0] = sx
                    m[p[1], 1] = sy
                    m[p[2], 2] = sz
                    mats.append(m)
    return mats


def _score_c2w(c2w: Float[Tensor, "batch 4 4"]) -> Tensor:
    origins = c2w[:, :3, 3]
    right = c2w[:, :3, 0]
    up = -c2w[:, :3, 1]  # OpenCV up is -Y in camera coordinates.
    look = c2w[:, :3, 2]

    # Constraints from the README.
    radius_error = (origins.norm(dim=-1) - 2.0).abs().mean()
    positive_y_penalty = torch.relu(-origins[:, 1]).mean()
    tangent_error = (origins * look).sum(dim=-1).abs().mean()
    up_penalty = torch.relu(-up[:, 1]).mean()

    # Mild orthonormality regularizer (should already be close for valid rigid transforms).
    ortho_error = (
        (right * up).sum(dim=-1).abs().mean()
        + (right * look).sum(dim=-1).abs().mean()
        + (up * look).sum(dim=-1).abs().mean()
    )
    return radius_error + positive_y_penalty + tangent_error + up_penalty + 0.1 * ortho_error


def _infer_conversion(extrinsics: Float[Tensor, "batch 4 4"]) -> tuple[Literal["w2c", "c2w"], Tensor]:
    device = extrinsics.device
    best_score = None
    best_format: Literal["w2c", "c2w"] = "c2w"
    best_s = torch.eye(3, dtype=torch.float32, device=device)

    for s in _all_signed_permutation_mats(device):
        s_inv = s.transpose(0, 1)

        # Candidate: dataset is c2w in unknown camera axes.
        c2w_candidate = extrinsics.clone()
        c2w_candidate[:, :3, :3] = extrinsics[:, :3, :3] @ s_inv
        score = _score_c2w(c2w_candidate)
        if best_score is None or float(score) < float(best_score):
            best_score = score
            best_format = "c2w"
            best_s = s

        # Candidate: dataset is w2c in unknown camera axes.
        w2c_candidate = extrinsics.clone()
        w2c_candidate[:, :3, :3] = s @ extrinsics[:, :3, :3]
        w2c_candidate[:, :3, 3] = (s @ extrinsics[:, :3, 3].unsqueeze(-1)).squeeze(-1)
        c2w_from_w2c = torch.linalg.inv(w2c_candidate)
        score = _score_c2w(c2w_from_w2c)
        if float(score) < float(best_score):
            best_score = score
            best_format = "w2c"
            best_s = s

    return best_format, best_s


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    extrinsics = dataset["extrinsics"]
    format_kind, s = _infer_conversion(extrinsics)

    if format_kind == "c2w":
        c2w = extrinsics.clone()
        c2w[:, :3, :3] = extrinsics[:, :3, :3] @ s.transpose(0, 1)
    else:
        w2c = extrinsics.clone()
        w2c[:, :3, :3] = s @ extrinsics[:, :3, :3]
        w2c[:, :3, 3] = (s @ extrinsics[:, :3, 3].unsqueeze(-1)).squeeze(-1)
        c2w = torch.linalg.inv(w2c)

    return {
        "extrinsics": c2w,
        "intrinsics": dataset["intrinsics"],
        "images": dataset["images"],
    }


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "c2w"
    with metadata_path.open("r") as f:
        metadata = json.load(f)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    fmt, _ = _infer_conversion(extrinsics)
    return fmt


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "+z"
    with metadata_path.open("r") as f:
        metadata = json.load(f)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    _, s = _infer_conversion(extrinsics)
    # l_cv = S l_unknown  =>  l_unknown = S^T l_cv
    l_unknown = s.transpose(0, 1) @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return _axis_name(l_unknown)


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "-y"
    with metadata_path.open("r") as f:
        metadata = json.load(f)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    _, s = _infer_conversion(extrinsics)
    u_unknown = s.transpose(0, 1) @ torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
    return _axis_name(u_unknown)


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "+x"
    with metadata_path.open("r") as f:
        metadata = json.load(f)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    _, s = _infer_conversion(extrinsics)
    r_unknown = s.transpose(0, 1) @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    return _axis_name(r_unknown)


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    return (
        "I inferred the puzzle camera convention by searching over all signed axis "
        "permutations and both extrinsic conventions (c2w or w2c). For each candidate, "
        "I converted to OpenCV-style c2w and scored it against the documented dataset "
        "constraints: camera radius of 2, nonnegative world y for camera origins, look "
        "vectors tangent to the viewing sphere, and upward-pointing camera up vectors. "
        "I selected the best-scoring convention and used it to convert all extrinsics."
    )
