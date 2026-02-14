import os
import re
import docx2txt


def extract_text_from_docx(path):
    return docx2txt.process(path)


def split_by_clausula(text):
    """
    Divide o texto por qualquer variação de CLÁUSULA
    """
    pattern = r'(CL[ÁA]USULA\s+[^\n]+)'

    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    clausulas = []

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        titulo = matches[i].group()
        conteudo = text[start:end]

        clausulas.append({
            "titulo": titulo.strip(),
            "conteudo": conteudo.strip()
        })

    return clausulas


def load_and_chunk_documents(folder_path):
    all_chunks = []

    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            full_path = os.path.join(folder_path, file)
            text = extract_text_from_docx(full_path)

            clausulas = split_by_clausula(text)

            for clausula in clausulas:
                all_chunks.append({
                    "filename": file,
                    "titulo": clausula["titulo"],
                    "content": clausula["conteudo"]
                })

    return all_chunks
