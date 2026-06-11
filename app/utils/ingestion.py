import os
import uuid
from tiktoken import encoding_for_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from stores.intelligence import Intelligence
from agents import (
    Entity_Extraction_System_Prompt,
    Entity_Extraction_User_Prompt,
    Summarize_Entity_Descriptions_System_Prompt,
    Summarize_Entity_Descriptions_User_Prompt,
    File_Summary_System_Prompt,
    File_Summary_User_Prompt,
)
from utils.graph import build_and_save_graphml
from dotenv import load_dotenv
import json
import pandas as pd
load_dotenv()

TUPLE_DELIMITER      = "<|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
ENTITY_TYPES         = "person, organization, location, event, concept, product, technology, other"
LANGUAGE             = "English"
SUMMARY_LENGTH       = 500

llm = ChatOpenAI(model="gpt-4o", temperature=0)
qdrant = Intelligence()

def text_chunking(text: str, chunk_size: int = 1200, overlap: int = 100) -> list[dict]:
    tokenizer = encoding_for_model("gpt-4o")
    tokens = tokenizer.encode(text)
    results = []
    for index, start in enumerate(range(0, len(tokens), chunk_size - overlap)):
        chunk_tokens = tokens[start : start + chunk_size]
        results.append(
            {
                "tokens": len(chunk_tokens),
                "content": tokenizer.decode(chunk_tokens).strip(),
                "chunk_order_index": index,
            }
        )
    return results
    
def parse_extraction_output(text: str) -> tuple[list[dict], list[dict]]:
    entities = []
    relationships = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line == COMPLETION_DELIMITER:
            continue
        parts = [p.strip() for p in line.split(TUPLE_DELIMITER)]
        if parts[0] == "entity" and len(parts) == 4:
            entities.append({
                "name": parts[1],
                "type": parts[2],
                "description": parts[3],
            })
        elif parts[0] == "relation" and len(parts) == 5:
            relationships.append({
                "source": parts[1],
                "target": parts[2],
                "keywords": parts[3],
                "description": parts[4],
            })
    return entities, relationships


def _merge_descriptions(name: str, descriptions: list[str]) -> str:
    description_list = "\n".join(json.dumps({"description": d}) for d in descriptions)
    system_msg = SystemMessage(content=Summarize_Entity_Descriptions_System_Prompt.format(
        summary_length=SUMMARY_LENGTH,
        language=LANGUAGE,
    ))
    human_msg = HumanMessage(content=Summarize_Entity_Descriptions_User_Prompt.format(
        description_type="Entity",
        description_name=name,
        description_list=description_list,
    ))
    return llm.invoke([system_msg, human_msg]).content.strip()

def _summarize_file(filename: str, file_type: str, content: str) -> str:
    system_msg = SystemMessage(content=File_Summary_System_Prompt)
    human_msg = HumanMessage(content=File_Summary_User_Prompt.format(
        filename=filename,
        file_type=file_type,
        content=content,
    ))
    return llm.invoke([system_msg, human_msg]).content.strip()


def process_unstructured(path: str, project_id: str) -> dict[str, str]:
    UNSTRUCTURED_EXTENSIONS = {".txt", ".pdf"}
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
        and os.path.splitext(f)[1].lower() in UNSTRUCTURED_EXTENSIONS
    ]

    # name -> list of {type, description, source_id}
    entity_occurrences: dict[str, list[dict]] = {}
    # list of {source, target, keywords, description, source_id}
    all_relationships: list[dict] = []
    file_summaries: dict[str, str] = {}

    # Phase 1: extract entities/relationships from every chunk across all files
    for file in files:
        with open(file, "r") as f:
            text = f.read()

        basename = os.path.basename(file)
        file_summaries[basename] = _summarize_file(basename, "unstructured text", text[:3000])

        chunks = text_chunking(text)

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            qdrant.store_text_chunk(chunk_id, {
                "content": chunk["content"],
                "tokens": chunk["tokens"],
                "chunk_order_index": chunk["chunk_order_index"],
                "datasource_id": project_id,
                "source_id": chunk_id,
                "file": basename,
            })

            system_msg = SystemMessage(content=Entity_Extraction_System_Prompt.format(
                tuple_delimiter=TUPLE_DELIMITER,
                completion_delimiter=COMPLETION_DELIMITER,
                entity_types=ENTITY_TYPES,
                language=LANGUAGE,
                examples="",
            ))
            human_msg = HumanMessage(content=Entity_Extraction_User_Prompt.format(
                tuple_delimiter=TUPLE_DELIMITER,
                completion_delimiter=COMPLETION_DELIMITER,
                entity_types=ENTITY_TYPES,
                language=LANGUAGE,
                input_text=chunk["content"],
            ))
            response = llm.invoke([system_msg, human_msg])
            entities, relationships = parse_extraction_output(response.content)

            for entity in entities:
                name = entity["name"]
                if name not in entity_occurrences:
                    entity_occurrences[name] = []
                entity_occurrences[name].append({
                    "type": entity["type"],
                    "description": entity["description"],
                    "source_id": chunk_id,
                })

            for rel in relationships:
                all_relationships.append({
                    "source": rel["source"],
                    "target": rel["target"],
                    "keywords": rel["keywords"],
                    "description": rel["description"],
                    "source_id": chunk_id,
                })

    # Phase 2: deduplicate entities and store
    entity_name_to_id: dict[str, str] = {}
    graph_entities: list[dict] = []

    for name, occurrences in entity_occurrences.items():
        entity_id = str(uuid.uuid4())
        entity_name_to_id[name] = entity_id

        if len(occurrences) == 1:
            description = occurrences[0]["description"]
        else:
            description = _merge_descriptions(name, [o["description"] for o in occurrences])

        source_ids = [o["source_id"] for o in occurrences]
        qdrant.store_entity(entity_id, {
            "entity_id": entity_id,
            "name": name,
            "type": occurrences[0]["type"],
            "description": description,
            "source_id": source_ids,
            "datasource_id": project_id,
        })
        graph_entities.append({
            "name": name,
            "type": occurrences[0]["type"],
            "description": description,
            "source_ids": source_ids,
        })

    # Phase 3: resolve relationships to canonical entity IDs and store
    for rel in all_relationships:
        src_id = entity_name_to_id.get(rel["source"])
        tgt_id = entity_name_to_id.get(rel["target"])
        if not src_id or not tgt_id:
            continue
        rel_id = str(uuid.uuid4())
        qdrant.store_relationship(rel_id, {
            "src_id": src_id,
            "tgt_id": tgt_id,
            "keywords": rel["keywords"],
            "description": rel["description"],
            "source_id": rel["source_id"],
            "datasource_id": project_id,
        })

    build_and_save_graphml(graph_entities, all_relationships, path)
    return file_summaries


def _read_csv_sections(filepath: str) -> dict[str, pd.DataFrame]:
    """Parse a CSV that may contain multiple tables separated by blank rows."""
    import io

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    # Split into sections at blank/all-comma lines
    raw_sections: list[list[str]] = []
    current: list[str] = []
    for line in raw_lines:
        if not line.strip().strip(","):
            if current:
                raw_sections.append(current)
                current = []
        else:
            current.append(line.rstrip("\n"))
    if current:
        raw_sections.append(current)

    result: dict[str, pd.DataFrame] = {}
    for section_lines in raw_sections:
        # Try parsing from each line until we get a multi-column table
        for start in range(len(section_lines)):
            try:
                content = "\n".join(section_lines[start:])
                df = pd.read_csv(io.StringIO(content))
                if len(df.columns) > 1 and len(df) > 0:
                    # Use any preceding title line as the section name
                    name = section_lines[0].strip().strip(",") if start > 0 else f"Section {len(result) + 1}"
                    result[name] = df
                    break
            except Exception:
                continue

    return result


def process_structured(path: str, project_id: str) -> dict[str, str]:

    STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
        and os.path.splitext(f)[1].lower() in STRUCTURED_EXTENSIONS
    ]
    structured_data_information = {}
    file_summaries: dict[str, str] = {}

    for file in files:
        basename = os.path.basename(file)

        if file.endswith(".csv"):
            sections = _read_csv_sections(file)
            if not sections:
                continue

            section_info_parts: list[str] = []
            summary_parts: list[str] = []
            for section_name, df in sections.items():
                cols = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
                section_info_parts.append(f"[{section_name}]\n{cols}")
                summary_parts.append(
                    f"[{section_name}] {df.shape[0]} rows x {df.shape[1]} columns\n"
                    f"Columns:\n{cols}\n"
                    f"Sample rows:\n{df.head(3).to_string(index=False)}"
                )

            columns_and_types = "\n\n".join(section_info_parts)
            summary_content = "\n\n".join(summary_parts)

        elif file.endswith(".xlsx") or file.endswith(".xls"):
            df = pd.read_excel(file)
            columns_and_types = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
            summary_content = (
                f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                f"Columns:\n{columns_and_types}\n"
                f"Sample rows:\n{df.head(3).to_string(index=False)}"
            )
        elif file.endswith(".parquet"):
            df = pd.read_parquet(file)
            columns_and_types = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
            summary_content = (
                f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                f"Columns:\n{columns_and_types}\n"
                f"Sample rows:\n{df.head(3).to_string(index=False)}"
            )
        else:
            continue

        structured_data_information[file] = columns_and_types
        file_summaries[basename] = _summarize_file(basename, "structured tabular data", summary_content)

    qdrant.store_structured_data(structured_data_information, project_id)
    return file_summaries
