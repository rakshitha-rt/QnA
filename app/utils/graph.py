import os
import time
import networkx as nx

SEP = "<SEP>"


def build_and_save_graphml(
    entities: list[dict],
    relationships: list[dict],
    output_dir: str,
) -> str:
    """
    entities:      [{name, type, description, source_ids: list[str]}]
    relationships: [{source, target, keywords, description, source_id: str}]
    """
    G = nx.Graph()

    for entity in entities:
        source_id = SEP.join(entity["source_ids"]) if isinstance(entity["source_ids"], list) else entity["source_ids"]
        G.add_node(
            entity["name"],
            entity_id=entity["name"],
            entity_type=entity["type"],
            description=entity["description"],
            source_id=source_id,
            file_path=entity.get("file_path", "unknown_source"),
            created_at=int(time.time()),
            truncate="",
        )

    for rel in relationships:
        if not G.has_node(rel["source"]) or not G.has_node(rel["target"]):
            continue
        G.add_edge(
            rel["source"],
            rel["target"],
            weight=1.0,
            description=rel["description"],
            keywords=rel["keywords"],
            source_id=rel["source_id"],
            file_path=rel.get("file_path", "unknown_source"),
            created_at=int(time.time()),
            truncate="",
        )

    output_path = os.path.join(output_dir, "graph_chunk_entity_relation.graphml")
    nx.write_graphml(G, output_path)
    return output_path
