import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_openai import OpenAIEmbeddings
import uuid


class Intelligence:
    def __init__(self):
        self.client = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.unstructured_data_chunks_collection = "unstructured_data_chunks"
        self.unstructured_entities_collection = "unstructured_entities"
        self.unstructured_relationships_collection = "unstructured_relationships"
        self.structured_collection = "structured_data"

    def create_collections(self):
        existing = {c.name for c in self.client.get_collections().collections}
        for name in [
            self.unstructured_data_chunks_collection,
            self.unstructured_entities_collection,
            self.unstructured_relationships_collection,
        ]:
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
        if self.structured_collection not in existing:
            self.client.create_collection(
                collection_name=self.structured_collection,
                vectors_config={},
            )

    def store_text_chunk(self, chunk_id: str, payload: dict):
        vector = self.embeddings.embed_query(payload["content"])
        self.client.upsert(
            collection_name=self.unstructured_data_chunks_collection,
            points=[PointStruct(id=chunk_id, vector=vector, payload=payload)],
        )

    def store_entity(self, entity_id: str, payload: dict):
        vector = self.embeddings.embed_query(payload["description"])
        self.client.upsert(
            collection_name=self.unstructured_entities_collection,
            points=[PointStruct(id=entity_id, vector=vector, payload=payload)],
        )

    def store_relationship(self, rel_id: str, payload: dict):
        vector = self.embeddings.embed_query(payload["description"])
        self.client.upsert(
            collection_name=self.unstructured_relationships_collection,
            points=[PointStruct(id=rel_id, vector=vector, payload=payload)],
        )

    def store_structured_data(
        self, structured_data_information: dict[str, str], datasource_id: str
    ):
        structured_data_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.structured_collection,
            points=[
                PointStruct(
                    id=structured_data_id,
                    vector={},
                    payload={"structured_data_information": structured_data_information, "datasource_id": datasource_id},
                )
            ],
        )
        

    def _project_filter(self, project_id: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="datasource_id",
                    match=MatchValue(value=project_id),
                )
            ]
        )

    def query_text_chunks(self, query: str, project_id: str):
        results = self.client.query_points(
            collection_name=self.unstructured_data_chunks_collection,
            query=self.embeddings.embed_query(query),
            query_filter=self._project_filter(project_id),
            limit=10,
        )
        return [
            {
                "content": p.payload.get("content", ""),
                "file": p.payload.get("file", ""),
            }
            for p in results.points
        ]

    def query_entities(self, query: str, project_id: str):
        results = self.client.query_points(
            collection_name=self.unstructured_entities_collection,
            query=self.embeddings.embed_query(query),
            query_filter=self._project_filter(project_id),
            limit=10,
        )

        all_chunk_ids = []
        for point in results.points:
            source_ids = point.payload.get("source_id", [])
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            all_chunk_ids.extend(source_ids)

        chunks_by_id = {}
        if all_chunk_ids:
            chunk_points = self.client.retrieve(
                collection_name=self.unstructured_data_chunks_collection,
                ids=list(set(all_chunk_ids)),
                with_payload=True,
            )
            chunks_by_id = {str(p.id): p.payload.get("content", "") for p in chunk_points}

        enriched = []
        for point in results.points:
            source_ids = point.payload.get("source_id", [])
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            enriched.append({
                "name": point.payload.get("name"),
                "type": point.payload.get("type"),
                "description": point.payload.get("description"),
                "text_chunks": [chunks_by_id[cid] for cid in source_ids if cid in chunks_by_id],
            })

        return enriched

    def query_relationships(self, query: str, project_id: str):
        results = self.client.query_points(
            collection_name=self.unstructured_relationships_collection,
            query=self.embeddings.embed_query(query),
            query_filter=self._project_filter(project_id),
            limit=10,
        )

        # Collect all entity IDs across all relationship results
        all_entity_ids = set()
        for point in results.points:
            p = point.payload
            if p.get("src_id"):
                all_entity_ids.add(p["src_id"])
            if p.get("tgt_id"):
                all_entity_ids.add(p["tgt_id"])

        # Fetch entities and build entity_id -> [chunk_ids] map
        entity_chunks_map: dict[str, list[str]] = {}
        if all_entity_ids:
            entity_points = self.client.retrieve(
                collection_name=self.unstructured_entities_collection,
                ids=list(all_entity_ids),
                with_payload=True,
            )
            for ep in entity_points:
                source_ids = ep.payload.get("source_id", [])
                if isinstance(source_ids, str):
                    source_ids = [source_ids]
                entity_chunks_map[str(ep.id)] = source_ids

        # Gather all chunk IDs needed across all relationships
        all_chunk_ids = set()
        for point in results.points:
            p = point.payload
            if p.get("source_id"):
                all_chunk_ids.add(p["source_id"])
            for eid in (p.get("src_id"), p.get("tgt_id")):
                if eid:
                    all_chunk_ids.update(entity_chunks_map.get(eid, []))

        # Fetch all chunks in one shot
        chunks_by_id = {}
        if all_chunk_ids:
            chunk_points = self.client.retrieve(
                collection_name=self.unstructured_data_chunks_collection,
                ids=list(all_chunk_ids),
                with_payload=True,
            )
            chunks_by_id = {str(p.id): p.payload.get("content", "") for p in chunk_points}

        # Build per-relationship result
        enriched = []
        for point in results.points:
            p = point.payload
            ids_for_rel = set()
            if p.get("source_id"):
                ids_for_rel.add(p["source_id"])
            for eid in (p.get("src_id"), p.get("tgt_id")):
                if eid:
                    ids_for_rel.update(entity_chunks_map.get(eid, []))

            enriched.append({
                "description": p.get("description"),
                "text_chunks": [chunks_by_id[cid] for cid in ids_for_rel if cid in chunks_by_id],
            })

        return enriched

    def get_structured_data_information(self, project_id: str) -> dict:
        results, _ = self.client.scroll(
            collection_name=self.structured_collection,
            scroll_filter=self._project_filter(project_id),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        data = {}
        for point in results:
            data.update(point.payload.get("structured_data_information", {}))
        return data
