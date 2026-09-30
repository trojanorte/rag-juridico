"""Extract DOCX and create stable, clause-aware chunks."""
from __future__ import annotations
import hashlib
import logging
import re
from pathlib import Path
import docx2txt
from core.config import MAX_CHUNK_TOKENS

log = logging.getLogger(__name__)
PARSER_VERSION = "2"
CLAUSE = re.compile(r"(?im)^\s*(CL[ÁA]USULA\s+([^\n]{1,160}))")
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ \xa0]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line.strip() for line in text.splitlines())).strip()

def extract_text_from_docx(path: str) -> str:
    return normalize_text(docx2txt.process(path) or "")

def token_count(text: str) -> int:
    return len(TOKEN.findall(text))

def split_by_clausula(text: str) -> list[dict]:
    matches = list(CLAUSE.finditer(text))
    if not matches:
        return [{"titulo": "Texto sem cláusula identificada", "conteudo": text, "start": 0, "clause_number": None}] if text else []
    sections = []
    if text[:matches[0].start()].strip():
        sections.append({"titulo": "Preâmbulo", "conteudo": text[:matches[0].start()].strip(), "start": 0, "clause_number": None})
    for n, match in enumerate(matches):
        end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        sections.append({"titulo": match.group(1).strip(), "conteudo": text[match.start():end].strip(), "start": match.start(), "clause_number": match.group(2).split()[0]})
    return sections

def _pieces(text: str, limit: int):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]
    cursor, group, group_start = 0, [], 0
    for sentence in sentences:
        position = text.find(sentence, cursor)
        if position < 0:
            position = cursor
        cursor = position + len(sentence)
        if group and token_count(" ".join(group + [sentence])) > limit:
            yield " ".join(group), group_start, position
            group = []
        if not group:
            group_start = position
        if token_count(sentence) > limit:
            if group:
                yield " ".join(group), group_start, position
                group = []
            # An unpunctuated legal paragraph still needs to fit the E5 window.
            words = list(re.finditer(r"\S+", sentence))
            block, block_start = [], 0
            for word in words:
                if block and token_count(" ".join(block + [word.group()])) > limit:
                    yield " ".join(block), position + block_start, position + word.start()
                    block = []
                if not block:
                    block_start = word.start()
                block.append(word.group())
            if block:
                yield " ".join(block), position + block_start, cursor
        else:
            group.append(sentence)
    if group:
        yield " ".join(group), group_start, cursor

def chunk_document(text: str, filename: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[dict]:
    text = normalize_text(text)
    if not text:
        return []
    document_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = document_hash[:16]
    title = next((line for line in text.splitlines() if line and len(line) <= 180), Path(filename).stem)
    chunks = []
    for clause_index, clause in enumerate(split_by_clausula(text), start=1):
        clause_key = str(clause_index)
        for part, (content, relative_start, relative_end) in enumerate(_pieces(clause["conteudo"], max_tokens), start=1):
            chunks.append({
                "chunk_id": f"{document_id}:clause_{clause_key}:part_{part}",
                "document_id": document_id, "document_hash": document_hash,
                "document_version": document_hash[:12], "filename": filename,
                "document_title": title, "categoria": None, "sindicato_laboral": None,
                "sindicato_patronal": None, "abrangencia_territorial": None,
                "vigencia_inicio": None, "vigencia_fim": None,
                "clause_number": clause["clause_number"], "clause_title": clause["titulo"],
                "section": None, "page": None,
                "start_position": clause["start"] + relative_start,
                "end_position": clause["start"] + relative_end, "content": content,
            })
    return chunks

def load_and_chunk_documents(folder_path: str) -> list[dict]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(folder_path)
    chunks, seen_hashes = [], set()
    for path in sorted(folder.glob("*.docx")):
        try:
            text = extract_text_from_docx(str(path))
            if not text:
                log.warning("Ignored empty document: %s", path.name)
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                log.warning("Duplicate document content ignored: %s", path.name)
                continue
            seen_hashes.add(digest)
            produced = chunk_document(text, path.name)
            if not produced:
                log.warning("No chunks produced: %s", path.name)
            chunks.extend(produced)
        except Exception:
            log.exception("Document processing failed: %s", path.name)
            raise
    return chunks
