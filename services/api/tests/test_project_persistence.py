"""DRISHTI - Project Persistence Tests
Phase 09: Interaction, Persistence & Export
"""

import json
from pathlib import Path
import pytest
from app.project.manager import ProjectManager
from app.scene.manager import SceneManager
from app.schemas.assets import AssetDimensions, AssetMetadata
from app.schemas.project import Project, ProjectSettings
from app.schemas.scene import Scene


@pytest.fixture
def temp_project_dir(tmp_path: Path):
    proj_dir = tmp_path / "projects"
    proj_dir.mkdir(parents=True, exist_ok=True)
    return proj_dir


@pytest.fixture
def temp_scene_dir(tmp_path: Path):
    sc_dir = tmp_path / "scenes"
    sc_dir.mkdir(parents=True, exist_ok=True)
    return sc_dir


@pytest.fixture
def sample_scene(temp_scene_dir: Path) -> Scene:
    sm = SceneManager(storage_dir=temp_scene_dir)
    asset = AssetMetadata(
        asset_id="asset-photo-100",
        category="original_photo",
        filename="test_photo.jpg",
        relative_path="photos/test_photo.jpg",
        mime_type="image/jpeg",
        size_bytes=2048,
        checksum_sha256="a" * 64,
        dimensions=AssetDimensions(width=800, height=600),
    )
    scene = sm.create_scene_from_photo(asset, name="Test Photo Scene")
    return scene


def test_create_and_load_project(temp_project_dir: Path, sample_scene: Scene):
    pm = ProjectManager(storage_dir=temp_project_dir)
    project = pm.create_project_from_scene(
        scene=sample_scene,
        name="My Workspace Project",
        settings=ProjectSettings(active_representation="combined")
    )

    assert project.id.startswith("proj-")
    assert project.name == "My Workspace Project"
    assert project.scene_id == sample_scene.id
    assert project.settings.active_representation == "combined"
    assert len(project.asset_manifest) >= 1
    assert project.asset_manifest[0].asset_id == "asset-photo-100"

    # Verify atomic disk persistence
    proj_file = temp_project_dir / f"{project.id}.json"
    assert proj_file.is_file()

    # Clear memory cache and reload
    pm2 = ProjectManager(storage_dir=temp_project_dir)
    reloaded = pm2.get_project(project.id)
    assert reloaded.id == project.id
    assert reloaded.name == "My Workspace Project"
    assert reloaded.scene_id == sample_scene.id


def test_project_atomic_save(temp_project_dir: Path, sample_scene: Scene):
    pm = ProjectManager(storage_dir=temp_project_dir)
    project = pm.create_project_from_scene(sample_scene, name="Initial Name")

    updated = pm.save_project(
        project_id=project.id,
        name="Renamed Project",
        settings=ProjectSettings(show_grid=False)
    )

    assert updated.name == "Renamed Project"
    assert updated.settings.show_grid is False

    # Check file on disk
    with open(temp_project_dir / f"{project.id}.json", "r") as f:
        data = json.load(f)
    assert data["name"] == "Renamed Project"
    assert data["settings"]["show_grid"] is False


def test_list_projects_order(temp_project_dir: Path, sample_scene: Scene):
    pm = ProjectManager(storage_dir=temp_project_dir)
    p1 = pm.create_project_from_scene(sample_scene, name="Project 1")
    p2 = pm.create_project_from_scene(sample_scene, name="Project 2")

    projects = pm.list_projects()
    assert len(projects) >= 2
    # Verify both IDs present
    p_ids = [p.id for p in projects]
    assert p1.id in p_ids
    assert p2.id in p_ids


def test_delete_project(temp_project_dir: Path, sample_scene: Scene):
    pm = ProjectManager(storage_dir=temp_project_dir)
    project = pm.create_project_from_scene(sample_scene, name="To Delete")
    p_path = temp_project_dir / f"{project.id}.json"
    assert p_path.is_file()

    pm.delete_project(project.id)
    assert not p_path.is_file()
