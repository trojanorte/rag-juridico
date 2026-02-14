from ingest.parser import load_and_chunk_documents

chunks = load_and_chunk_documents("convencoes coletivas")

print("Total de chunks:", len(chunks))

if len(chunks) > 0:
    print("\nExemplo de chunk:\n")
    print("Arquivo:", chunks[0]["filename"])
    print("Título:", chunks[0]["titulo"])
    print("Conteúdo (primeiros 500 caracteres):\n")
    print(chunks[0]["content"][:500])
