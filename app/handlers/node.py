import sys
import os
import logging

logger = logging.getLogger(__name__)
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stores.intelligence import Intelligence
from model import query_state, Citation
from agents import CLARIFICATION_PROMPT, PLANNER_PROMPT, Keywords_Extraction_System_Prompt, Keywords_Extraction_User_Prompt, CODER_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from model import ClarificationsRequest, Keywords
from dotenv import load_dotenv
import httpx
import json

load_dotenv()
def extract_keywords(query: str):
    llm = ChatOpenAI(model="o4-mini")
    system_prompt = Keywords_Extraction_System_Prompt
    user_prompt =Keywords_Extraction_User_Prompt.format(query=query)
    structured_llm = llm.with_structured_output(Keywords)
    response = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return response.model_dump_json()

def clarify_question(query_state: query_state):
    llm = ChatOpenAI(model="o4-mini")
    system_prompt = CLARIFICATION_PROMPT

    files_gist = ""
    if query_state.file_summaries:
        lines = "\n".join(
            f"- {fname}: {summary}"
            for fname, summary in query_state.file_summaries.items()
        )
        files_gist = f"\nAvailable Files:\n{lines}\n"

    user_prompt = f"""
    User Query: {query_state.user_query}
    {files_gist}
    Clarification History: {query_state.clarification_history}
    Number of Clarifications: {query_state.number_of_clarifications}
    """
    structured_ll = llm.with_structured_output(ClarificationsRequest)
    response = structured_ll.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return {"clarifications_request": response}

def planner(state: query_state):
    llm = ChatOpenAI(model="o4-mini", max_tokens=4096)

    _query_type = state.query_type
    _project_id = state.project_id
    _structured_data_information = state.structured_data_information

    citations: list[Citation] = []

    @tool
    def query_structured(query: str):
        """Analyze structured data (CSV/tabular files) by running Python code with pandas."""
        logger.info("[query_structured] called | query=%r", query)
        llm = ChatOpenAI(model="o4-mini")
        system_prompt = CODER_PROMPT
        user_prompt = f"""
        User Query: {query}
        Structured Data Information: {_structured_data_information}
        """
        @tool
        def python_repl(code: str) -> str:
            """Execute Python code. pandas and numpy are available. /data/uploads is readable. Returns stdout."""
            sandbox_url = os.environ.get("SANDBOX_URL", "http://sandbox:8001")
            try:
                r = httpx.post(
                    f"{sandbox_url}/run",
                    json={"code": code, "timeout": 30},
                    timeout=35,
                )
                r.raise_for_status()
                result = r.json()
                if result.get("error"):
                    return f"Error: {result['error']}\n{result.get('stderr', '')}"
                return result.get("stdout") or result.get("stderr") or ""
            except Exception as exc:
                return f"Sandbox unavailable: {exc}"

        agent = create_react_agent(llm, [python_repl], prompt=system_prompt)
        response = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
        result = response["messages"][-1].content
        logger.info("[query_structured] returned %d chars", len(result))

        for path in (_structured_data_information or {}).keys():
            citations.append(Citation(
                type="structured",
                file=os.path.basename(path),
                snippet=f"Query: {query}",
            ))

        return result

    @tool
    def query_unstructured(query: str):
        """Search unstructured documents (text chunks, entities, or relationships)."""
        logger.info("[query_unstructured] called | query_type=%r project_id=%r query=%r", _query_type, _project_id, query)
        intelligence = Intelligence()
        if _query_type == "local":
            _keywords = json.loads(extract_keywords(query))
            kw = ", ".join(_keywords["low_level_keywords"])
            if kw.strip():
                logger.info("[query_unstructured] local entity search | keywords=%r", kw)
                result = intelligence.query_entities(kw, _project_id)
            else:
                logger.info("[query_unstructured] empty keywords — falling back to text chunk search | query=%r", query)
                result = intelligence.query_text_chunks(query, _project_id)
        elif _query_type == "global":
            _keywords = json.loads(extract_keywords(query))
            kw = ", ".join(_keywords["high_level_keywords"])
            if kw.strip():
                logger.info("[query_unstructured] global relationship search | keywords=%r", kw)
                result = intelligence.query_relationships(kw, _project_id)
            else:
                logger.info("[query_unstructured] empty keywords — falling back to text chunk search | query=%r", query)
                result = intelligence.query_text_chunks(query, _project_id)
        else:
            logger.info("[query_unstructured] text chunk search")
            result = intelligence.query_text_chunks(query, _project_id)

        result_len = len(result) if isinstance(result, list) else 0
        logger.info("[query_unstructured] returned type=%s len=%d", type(result).__name__, result_len)

        for item in (result or []):
            if "content" in item:
                citations.append(Citation(
                    type="chunk",
                    file=item.get("file", ""),
                    snippet=item["content"][:300],
                ))
            elif "name" in item:
                citations.append(Citation(
                    type="entity",
                    file="",
                    snippet=f"{item['name']} ({item.get('type', '')}): {item.get('description', '')[:250]}",
                ))
            elif "description" in item:
                citations.append(Citation(
                    type="relationship",
                    file="",
                    snippet=item["description"][:300],
                ))

        if result:
            for i, item in enumerate(result[:3]):
                if isinstance(item, dict) and "name" in item:
                    logger.info("[query_unstructured] item[%d] name=%r desc=%r chunks=%d",
                                i, item.get("name"), item.get("description", "")[:120], len(item.get("text_chunks", [])))
                elif isinstance(item, dict) and "content" in item:
                    logger.info("[query_unstructured] item[%d] content=%r", i, item.get("content", "")[:150])
                else:
                    logger.info("[query_unstructured] item[%d]=%r", i, str(item)[:200])
            return result
        else:
            logger.info("[query_unstructured] no relevant results above score threshold")
            return "NO_RELEVANT_DATA: The knowledge store contains no information relevant to this query."

    tools = [query_unstructured, query_structured]
    user_prompt = f"""
User Query: {state.user_query}
Clarification History: {state.clarification_history}
Structured Data Available: { state.structured_data_information if state.structured_data_information else "Not Available"}
"""
    agent = create_react_agent(llm, tools, prompt=PLANNER_PROMPT)
    response = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
    return {"planner_request": response["messages"][-1].content, "citations": citations}

MAX_CLARIFICATIONS = 3

def _route_after_clarification(state: query_state) -> str:
    req = state.clarifications_request
    if (req and any(t.ambiguous for t in req.ambiguous_terms)
            and state.number_of_clarifications < MAX_CLARIFICATIONS):
        return "end"
    return "planner"

def build_graph():
    graph = StateGraph(query_state)
    graph.add_node("clarification", clarify_question)
    graph.add_node("planner", planner)
    graph.add_edge(START, "clarification")
    graph.add_conditional_edges("clarification", _route_after_clarification, {"end": END, "planner": "planner"})
    graph.add_edge("planner", END)
    return graph.compile()


