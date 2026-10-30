"""DRISHTI - Export API Endpoints
Version 1.0.0 (Phase 09: Interaction, Persistence & Export)
"""

from fastapi import APIRouter, Query, Response
from fastapi.responses import Response

from app.export.engine import export_engine

router = APIRouter(prefix="/export", tags=["Export"])


@router.get("/mesh/{scene_id}/{entity_id}")
async def export_entity_mesh(
    scene_id: str,
    entity_id: str,
    apply_transform: bool = Query(default=False)
):
    glb_bytes = export_engine.export_entity_mesh_glb(
        scene_id=scene_id,
        entity_id=entity_id,
        apply_transform=apply_transform,
    )
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'attachment; filename="{entity_id}_mesh.glb"'
        }
    )


@router.get("/scene/{scene_id}")
async def export_assembled_scene(scene_id: str):
    glb_bytes = export_engine.export_assembled_scene_glb(scene_id=scene_id)
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'attachment; filename="assembled_scene_{scene_id[:8]}.glb"'
        }
    )


@router.get("/metadata/{scene_id}")
async def export_scene_metadata(scene_id: str):
    json_str = export_engine.export_scene_metadata_json(scene_id=scene_id)
    return Response(
        content=json_str,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="scene_{scene_id[:8]}_metadata.json"'
        }
    )


@router.get("/project/{project_id}/archive")
async def export_project_archive(project_id: str):
    zip_bytes = export_engine.create_portable_project_archive(project_id=project_id)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="drishti_project_{project_id[:8]}.zip"'
        }
    )
