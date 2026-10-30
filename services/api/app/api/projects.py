"""DRISHTI - Project API Endpoints
Version 1.0.0 (Phase 09: Interaction, Persistence & Export)
"""

from typing import List, Optional
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from app.project.manager import project_manager
from app.scene.manager import scene_manager
from app.schemas.project import Project, ProjectSettings

router = APIRouter(prefix="/projects", tags=["Projects"])


class CreateProjectRequest(BaseModel):
    scene_id: str
    name: Optional[str] = None
    settings: Optional[ProjectSettings] = None


class SaveProjectRequest(BaseModel):
    name: Optional[str] = None
    settings: Optional[ProjectSettings] = None


class ProjectListResponse(BaseModel):
    projects: List[Project]
    total: int


@router.get("", response_model=ProjectListResponse)
async def list_projects() -> ProjectListResponse:
    projects = project_manager.list_projects()
    return ProjectListResponse(projects=projects, total=len(projects))


@router.post("", response_model=Project)
async def create_project(request: CreateProjectRequest) -> Project:
    scene = scene_manager.get_scene(request.scene_id)
    return project_manager.create_project_from_scene(
        scene=scene,
        name=request.name,
        settings=request.settings,
    )


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str) -> Project:
    return project_manager.get_project(project_id)


@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, request: SaveProjectRequest) -> Project:
    return project_manager.save_project(
        project_id=project_id,
        name=request.name,
        settings=request.settings,
    )


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    success = project_manager.delete_project(project_id)
    return {"status": "ok", "deleted": success, "project_id": project_id}


@router.post("/import", response_model=Project)
async def import_project(
    file: UploadFile = File(...),
    new_name: Optional[str] = Form(default=None)
) -> Project:
    from app.export.engine import export_engine
    content = await file.read()
    return export_engine.import_portable_project_archive(content, new_name=new_name)

