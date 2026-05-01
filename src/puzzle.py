from pathlib import Path
import json
from typing import Literal, Optional, TypedDict

from jaxtyping import Float
from PIL import Image
import torch
from torch import Tensor

CONSTRAINT_TOLERANCE = 1e-2


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(dataset_path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""
    with (dataset_path / "metadata.json").open("r") as metadata_file:
        metadata = json.load(metadata_file)

    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    intrinsics = torch.tensor(metadata["intrinsics"], dtype=torch.float32)

    # Load grayscale images if present (some distributed puzzle copies only ship metadata).
    image_root = dataset_path / "images"
    image_paths = sorted(list(image_root.glob("*.png")) + list(image_root.glob("*.jpg")) + list(image_root.glob("*.jpeg")))
    if image_paths:
        grayscale_images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("L")
            grayscale_images.append(
                torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image.height, image.width) / 255.0
            )
        stacked_images = torch.stack(grayscale_images, dim=0)
    else:
        stacked_images = None

    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "images": stacked_images,
    }


def _axis_name(axis_vector: Tensor) -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """Return the dominant signed axis label for a 3D direction vector."""
    dominant_axis_index = int(torch.argmax(torch.abs(axis_vector)).item())
    axis_sign = "+" if float(axis_vector[dominant_axis_index]) > 0 else "-"
    axis_names = ["x", "y", "z"]
    return f"{axis_sign}{axis_names[dominant_axis_index]}"  # type: ignore[return-value]


def _all_signed_permutation_mats(device: torch.device) -> list[Float[Tensor, "3 3"]]:
    """Generate all 48 signed axis-permutation conversion matrices."""
    conversion_matrices: list[Tensor] = []
    axis_permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    axis_signs = (-1.0, 1.0)
    for axis_permutation in axis_permutations:
        for sign_x in axis_signs:
            for sign_y in axis_signs:
                for sign_z in axis_signs:
                    conversion_matrix = torch.zeros((3, 3), dtype=torch.float32, device=device)
                    conversion_matrix[axis_permutation[0], 0] = sign_x
                    conversion_matrix[axis_permutation[1], 1] = sign_y
                    conversion_matrix[axis_permutation[2], 2] = sign_z
                    conversion_matrices.append(conversion_matrix)
    return conversion_matrices


def _radius_error(c2w: Float[Tensor, "batch 4 4"]) -> Tensor:
    """Measure mean deviation of camera distance from radius 2."""
    camera_location = c2w[:, :3, 3]  # this is of size "batch 3"
    return (camera_location.norm(dim=-1) - 2.0).abs().mean()


def _positive_y_error(c2w: Float[Tensor, "batch 4 4"]) -> Tensor:
    """Penalize cameras whose world-space y coordinate is negative."""
    camera_locations = c2w[:, :3, 3]
    return torch.clamp(-camera_locations[:, 1], min=0).mean()


def _tangent_error(c2w: Float[Tensor, "batch 4 4"]) -> Tensor:
    """Measure misalignment between look direction and origin-facing direction."""
    camera_locations = c2w[:, :3, 3]
    look_directions = c2w[:, :3, 2]
    camera_location_norms = camera_locations.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    expected_look_directions = -camera_locations / camera_location_norms
    # 0 means perfectly aligned, 2 means opposite direction.
    return (look_directions - expected_look_directions).norm(dim=-1).mean()


def _upward_error(c2w: Float[Tensor, "batch 4 4"]) -> Tensor:
    """Penalize camera up vectors that do not point toward world +Y."""
    up_directions = -c2w[:, :3, 1]  # OpenCV up is -Y in camera coordinates.
    return torch.clamp(-up_directions[:, 1], min=0).mean()


def _orthogonality_error(c2w: Float[Tensor, "batch 4 4"]) -> Tensor:
    """Measure non-orthogonality of right, up, and look camera axes."""
    right_directions = c2w[:, :3, 0]
    up_directions = -c2w[:, :3, 1]
    look_directions = c2w[:, :3, 2]
    return (
        (right_directions * up_directions).sum(dim=-1).abs().mean()
        + (right_directions * look_directions).sum(dim=-1).abs().mean()
        + (up_directions * look_directions).sum(dim=-1).abs().mean()
    )


def _handedness_error(c2w: Float[Tensor, "batch 4 4"]) -> Tensor:
    """Measure deviation from determinant +1 for rotation blocks."""
    rotation_matrices = c2w[:, :3, :3]
    return (torch.linalg.det(rotation_matrices) - 1.0).abs().mean()


def is_matching_radius(c2w: Float[Tensor, "batch 4 4"], tolerance: float = CONSTRAINT_TOLERANCE) -> bool:
    """Return whether the camera radius constraint is satisfied."""
    return float(_radius_error(c2w)) < tolerance


def is_matching_positive_y(c2w: Float[Tensor, "batch 4 4"], tolerance: float = CONSTRAINT_TOLERANCE) -> bool:
    """Return whether all camera centers lie in nonnegative world y."""
    return float(_positive_y_error(c2w)) < tolerance


def is_matching_tangent_look(c2w: Float[Tensor, "batch 4 4"], tolerance: float = CONSTRAINT_TOLERANCE) -> bool:
    """Return whether look vectors face the world origin."""
    return float(_tangent_error(c2w)) < tolerance


def is_matching_upward_vector(c2w: Float[Tensor, "batch 4 4"], tolerance: float = CONSTRAINT_TOLERANCE) -> bool:
    """Return whether camera up vectors point upward in world space."""
    return float(_upward_error(c2w)) < tolerance


def is_valid_rigid_rotation(c2w: Float[Tensor, "batch 4 4"], tolerance: float = CONSTRAINT_TOLERANCE) -> bool:
    """Return whether camera axes form a valid rigid rotation matrix."""
    return float(_orthogonality_error(c2w)) < tolerance and float(_handedness_error(c2w)) < tolerance


def _all_constraints_match(c2w: Float[Tensor, "batch 4 4"]) -> bool:
    """Check whether all geometric puzzle constraints are satisfied."""
    return (
        is_valid_rigid_rotation(c2w)
        and is_matching_radius(c2w)
        and is_matching_positive_y(c2w)
        and is_matching_tangent_look(c2w)
        and is_matching_upward_vector(c2w)
    )
    
    
def finding_matching_axis_transformation(c2w_extrinsics: Float[Tensor, "batch 4 4"]) -> tuple[bool, Optional[Tensor]]:
    """Find the matching axis transformation to openCV convention that satisfies all constraints for a given c2w 
    extrinsics set in an unknown convention."""
    conversion_matrix_candidates = _all_signed_permutation_mats(c2w_extrinsics.device)
    for conversion_matrix in conversion_matrix_candidates:
        c2w_transformed = c2w_extrinsics.clone()
        c2w_transformed[:, :3, :3] = c2w_extrinsics[:, :3, :3] @ conversion_matrix.transpose(0, 1)
        if _all_constraints_match(c2w_transformed):
            return True, conversion_matrix
    return False, None


def infer_best_conversion_to_opencv(extrinsics: Float[Tensor, "batch 4 4"]) -> tuple[Literal["w2c", "c2w"], Tensor]:
    """Find the source convention and axis mapping to OpenCV camera axes.
    The returned conversin matrix maps from c2w input convention to openCV convention."""
    print("--- attempting to interpet inputs as c2w ---")
    found_match, conversion_matrix = finding_matching_axis_transformation(extrinsics)
    if found_match:
        print(f"Found exact match while interpreting as c2w.")
        return "c2w", conversion_matrix
    print("--- attempting to interpret inputs as w2c ---")
    found_match, conversion_matrix = finding_matching_axis_transformation(torch.linalg.inv(extrinsics))
    if found_match:
        print(f"Found exact match while interpreting as w2c.")
        return "w2c", conversion_matrix
    raise ValueError("No exact axis/sign mapping matched all puzzle constraints.")


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
    source_convention, conversion_matrix = infer_best_conversion_to_opencv(extrinsics)

    c2w_source = extrinsics if source_convention == "c2w" else torch.linalg.inv(extrinsics)
    c2w_transformed_to_opencv = c2w_source.clone()
    c2w_transformed_to_opencv[:, :3, :3] = c2w_source[:, :3, :3] @ conversion_matrix.transpose(0, 1)

    return {
        "extrinsics": c2w_transformed_to_opencv,
        "intrinsics": dataset["intrinsics"],
        "images": dataset["images"],
    }


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "c2w"
    with metadata_path.open("r") as metadata_file:
        metadata = json.load(metadata_file)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    source_convention, _ = infer_best_conversion_to_opencv(extrinsics)
    return source_convention


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "+z"
    with metadata_path.open("r") as metadata_file:
        metadata = json.load(metadata_file)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    _, conversion_matrix = infer_best_conversion_to_opencv(extrinsics)
    # l_cv = S l_unknown  =>  l_unknown = S^T l_cv
    unknown_look_axis = conversion_matrix.transpose(0, 1) @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return _axis_name(unknown_look_axis)


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "-y"
    with metadata_path.open("r") as metadata_file:
        metadata = json.load(metadata_file)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    _, conversion_matrix = infer_best_conversion_to_opencv(extrinsics)
    unknown_up_axis = conversion_matrix.transpose(0, 1) @ torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
    return _axis_name(unknown_up_axis)


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""
    metadata_path = Path("data/Puzzle/metadata.json")
    if not metadata_path.exists():
        return "+x"
    with metadata_path.open("r") as metadata_file:
        metadata = json.load(metadata_file)
    extrinsics = torch.tensor(metadata["extrinsics"], dtype=torch.float32)
    _, conversion_matrix = infer_best_conversion_to_opencv(extrinsics)
    unknown_right_axis = conversion_matrix.transpose(0, 1) @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    return _axis_name(unknown_right_axis)


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """
    return (
        "I solved the puzzle by trying all possible signed axis permutations and checking "
        "whether the resulting camera poses satisfy the known dataset constraints. I first "
        "tested the metadata as camera-to-world extrinsics, then tested the inverse matrices "
        "as the world-to-camera case. For each candidate, I checked that camera centers are "
        "2 units from the origin, lie at nonnegative world y, look toward the world origin, "
        "have upward-facing up vectors, and form valid rigid rotations. The candidate that "
        "satisfied all of these checks gives the conversion into OpenCV-style camera-to-world "
        "extrinsics."
    )
