import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid


class Project:
    def __init__(self):
        self.client = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
        )
        self.project_collection = "projects"
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.project_collection not in existing:
            self.client.create_collection(
                collection_name=self.project_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    def add_project(self, name: str) -> str:
        project_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.project_collection,
            points=[
                PointStruct(
                    id=project_id,
                    vector=[0.0],
                    payload={"name": name, "id": project_id},
                )
            ],
        )
        return project_id

    def get_all_projects(self) -> list[dict]:
        results = self.client.scroll(
            collection_name=self.project_collection,
            with_payload=True,
            limit=100,
        )
        return [
            {"name": p.payload["name"].split("/")[-1], "id": p.payload["id"], "file_summaries": p.payload.get("file_summaries", {})}
            for p in results[0]
        ]

    def get_project(self, name: str) -> dict | None:
        results = self.client.scroll(
            collection_name=self.project_collection,
            scroll_filter={"must": [{"key": "name", "match": {"value": name}}]},
            with_payload=True,
            limit=1,
        )
        if not results[0]:
            return None
        p = results[0][0]
        return {
            "name": p.payload["name"],
            "id": p.payload["id"],
            "file_summaries": p.payload.get("file_summaries", {}),
        }

    def get_project_by_id(self, project_id: str) -> dict | None:
        results = self.client.retrieve(
            collection_name=self.project_collection,
            ids=[project_id],
            with_payload=True,
        )
        if not results:
            return None
        p = results[0]
        return {
            "name": p.payload["name"],
            "id": p.payload["id"],
            "file_summaries": p.payload.get("file_summaries", {}),
        }

    def update_project_summaries(self, project_id: str, new_summaries: dict[str, str]):
        existing_points = self.client.retrieve(
            collection_name=self.project_collection,
            ids=[project_id],
            with_payload=True,
        )
        existing = existing_points[0].payload.get("file_summaries", {}) if existing_points else {}
        self.client.set_payload(
            collection_name=self.project_collection,
            payload={"file_summaries": {**existing, **new_summaries}},
            points=[project_id],
        )
