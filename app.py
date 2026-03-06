import streamlit as st

from rag_generator import answer_question
from observability.telemetry import telemetry


st.set_page_config(
    page_title="Assistente de Convenções Coletivas",
    layout="wide"
)

st.title("📄 Assistente de Convenções Coletivas")
st.write("Pergunte algo sobre os acordos coletivos.")

question = st.text_input("Digite sua pergunta:")

if st.button("Consultar"):
    telemetry.reset()

    start = telemetry.start_timer()

    with st.spinner("Consultando documentos..."):
        answer, sources = answer_question(question)

    telemetry.metrics["total_time"] = telemetry.stop_timer(start)

    st.subheader("Resposta")
    st.write(answer)

    st.subheader("Fontes")
    if sources:
        for arquivo, titulo in sources:
            st.write(f"- **{arquivo}** | {titulo}")
    else:
        st.write("Nenhuma fonte retornada.")