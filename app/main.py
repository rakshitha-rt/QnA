import os
import sys
import shutil
import random
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from pyvis.network import Network
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from stores.project import Project
from stores.intelligence import Intelligence
from utils.ingestion import process_structured, process_unstructured
from handlers.node import build_graph
from model import QueryRequest, ClarificationHistory
from dotenv import load_dotenv
load_dotenv()

def _retry(fn, label, retries=10, delay=3):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries:
                raise
            logger.warning("%s not ready (attempt %d/%d): %s — retrying in %ds", label, attempt, retries, exc, delay)
            time.sleep(delay)

app = FastAPI()
project_store = _retry(Project, "Qdrant/Project")
intelligence = _retry(Intelligence, "Qdrant/Intelligence")
intelligence.create_collections()

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "../templates/index.html")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "../temp"))

STRUCTURED_EXTENSIONS   = {".csv", ".xlsx", ".xls", ".parquet"}
UNSTRUCTURED_EXTENSIONS = {".txt", ".pdf"}


class CreateProjectRequest(BaseModel):
    name: str


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(TEMPLATE_PATH) as f:
        return f.read()


@app.get("/projects")
async def list_projects():
    return project_store.get_all_projects()


@app.post("/projects")
async def create_project(body: CreateProjectRequest):
    project_id = project_store.add_project(body.name)
    return {"id": project_id, "name": body.name}


@app.post("/projects/{project_id}/upload")
async def upload_file(project_id: str, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in STRUCTURED_EXTENSIONS | UNSTRUCTURED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: csv, xlsx, xls, parquet, txt, pdf."
        )

    project_tmp_dir = os.path.join(UPLOAD_DIR, project_id)
    os.makedirs(project_tmp_dir, exist_ok=True)
    tmp_path = os.path.join(project_tmp_dir, file.filename)

    with open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    if ext in STRUCTURED_EXTENSIONS:
        summaries = process_structured(project_tmp_dir, project_id)
    else:
        summaries = process_unstructured(project_tmp_dir, project_id)

    project_store.update_project_summaries(project_id, summaries)

    return {"filename": file.filename, "type": "structured" if ext in STRUCTURED_EXTENSIONS else "unstructured"}


@app.post("/query")
async def query_project(body: QueryRequest):
    graph = build_graph()
    structured_data_information = intelligence.get_structured_data_information(body.project_id)
    project = project_store.get_project_by_id(body.project_id)
    file_summaries = project.get("file_summaries", {}) if project else {}
    result = graph.invoke({
        "user_query": body.query,
        "project_id": body.project_id,
        "query_type": body.query_type,
        "clarification_history": ClarificationHistory(clarifications=body.clarification_history),
        "number_of_clarifications": len(body.clarification_history),
        "structured_data_information": structured_data_information,
        "file_summaries": file_summaries,
    })

    clarifications_request = result.get("clarifications_request")
    needs_clarification = (
        clarifications_request is not None and
        any(t.ambiguous for t in clarifications_request.ambiguous_terms)
    )

    if needs_clarification:
        c = clarifications_request.clarification
        return {
            "type": "clarification",
            "term": c.term,
            "question": c.question,
            "options": c.options,
        }

    raw_citations = result.get("citations", [])
    citations = [c.model_dump() if hasattr(c, "model_dump") else c for c in raw_citations]
    return {"type": "answer", "response": result.get("planner_request"), "citations": citations}


@app.get("/graph/{project_id}", response_class=HTMLResponse)
async def get_graph(project_id: str):
    graph_path = os.path.join(UPLOAD_DIR, project_id, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="Graph not found for this project.")

    G = nx.read_graphml(graph_path)

    net = Network(height="100vh", width="100%", notebook=False)
    net.from_nx(G)

    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if "description" in node:
            node["title"] = node["description"]

    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]

    return net.generate_html()