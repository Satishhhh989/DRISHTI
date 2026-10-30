"""PHOTO → 3D ENGINE - Common Geometric & Spatial Schemas
"""

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Vector3D(BaseModel):
    x: float = Field(..., description="X coordinate in meters/world units")
    y: float = Field(..., description="Y coordinate in meters/world units")
    z: float = Field(..., description="Z coordinate in meters/world units")


class EulerRotation(BaseModel):
    x: float = Field(..., description="X rotation in radians (pitch)")
    y: float = Field(..., description="Y rotation in radians (yaw)")
    z: float = Field(..., description="Z rotation in radians (roll)")
    order: Literal["XYZ", "YXZ", "ZXY", "ZYX", "YZX", "XZY"] = Field(
        default="XYZ", description="Euler rotation order"
    )


class Transform(BaseModel):
    position: Vector3D = Field(
        default_factory=lambda: Vector3D(x=0.0, y=0.0, z=0.0),
        description="Position vector"
    )
    rotation: EulerRotation = Field(
        default_factory=lambda: EulerRotation(x=0.0, y=0.0, z=0.0),
        description="Rotation in radians"
    )
    scale: Vector3D = Field(
        default_factory=lambda: Vector3D(x=1.0, y=1.0, z=1.0),
        description="Scale multiplier along axes"
    )

    @model_validator(mode="after")
    def validate_scale_positive(self) -> "Transform":
        if self.scale.x <= 0 or self.scale.y <= 0 or self.scale.z <= 0:
            raise ValueError("Transform scale components must be strictly positive non-zero numbers")
        return self


class BoundingBox2D(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized left boundary [0, 1]")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized top boundary [0, 1]")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Normalized right boundary [0, 1]")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom boundary [0, 1]")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox2D":
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})")
        return self


class CoordinateSystemInfo(BaseModel):
    handedness: Literal["right-handed"] = "right-handed"
    up_axis: Literal["+Y"] = "+Y"
    forward_axis: Literal["+Z"] = "+Z"
    right_axis: Literal["+X"] = "+X"
    unit: Literal["meters"] = "meters"
