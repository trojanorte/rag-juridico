import os
import re
import docx2txt


MIN_CONTENT_LEN = 80


def extract_text_from_docx(path: str) -> str:
    text = docx2txt.process(path)
    return text or ""


def normalize_text(text: str) -> str:
    if not text:
        return ""

    # normaliza quebras de linha e espaços
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    # remove espaços duplicados
    text = re.sub(r"[ \xa0]+", " ", text)

    # reduz quebras excessivas
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove espaços antes/depois das linhas
    text = "\n".join(line.strip() for line in text.splitlines())

    # remove linhas vazias repetidas de novo após strip
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_valid_clause_content(content: str) -> bool:
    content = (content or "").strip()
    return len(content) >= MIN_CONTENT_LEN


def split_by_clausula(text: str) -> list[dict]:
    """
    Divide o texto por cláusulas, preservando título e conteúdo.
    Aceita variações como:
    - CLÁUSULA PRIMEIRA
    - CLAUSULA PRIMEIRA
    - CLÁUSULA 1
    - CLAUSULA 1ª
    """

    if not text:
        return []

    # ancora no início de linha para evitar matches no meio do texto
    clause_pattern = re.compile(
        r"(?im)^(CL[ÁA]USULA\s+[^\n]{3,150})"
    )

    matches = list(clause_pattern.finditer(text))
    clausulas = []

    if not matches:
        return clausulas

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        titulo = match.group(1).strip(" -:\n\t")
        conteudo = text[start:end].strip()

        if not is_valid_clause_content(conteudo):
            continue

        clausulas.append({
            "titulo": titulo,
            "conteudo": conteudo,
        })

    return clausulas


def fallback_split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[dict]:
    """
    Fallback para documentos que não possuem marcação clara de cláusulas.
    Divide em blocos por tamanho com sobreposição.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    idx = 1
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()

        if len(piece) >= MIN_CONTENT_LEN:
            chunks.append({
                "titulo": f"BLOCO {idx}",
                "conteudo": piece,
            })

        if end == text_len:
            break

        start = max(end - overlap, start + 1)
        idx += 1

    return chunks


def load_and_chunk_documents(folder_path: str) -> list[dict]:
    all_chunks = []

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".docx"):
            continue

        full_path = os.path.join(folder_path, file)

        try:
            raw_text = extract_text_from_docx(full_path)
            text = normalize_text(raw_text)

            if not text:
                print(f"[AVISO] Documento vazio ou não lido corretamente: {file}")
                continue

            clausulas = split_by_clausula(text)

            # fallback se não encontrar cláusulas
            if not clausulas:
                print(f"[AVISO] Nenhuma cláusula encontrada em {file}. Usando fallback por blocos.")
                clausulas = fallback_split_text(text)

            for clausula in clausulas:
                titulo = (clausula.get("titulo") or "trecho_sem_titulo").strip()
                conteudo = (clausula.get("conteudo") or "").strip()

                if not is_valid_clause_content(conteudo):
                    continue

                all_chunks.append({
                    "filename": file,
                    "titulo": titulo,
                    "content": conteudo,
                })

        except Exception as e:
            print(f"[ERRO] Falha ao processar {file}: {e}")

    return all_chunks