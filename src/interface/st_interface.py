import streamlit as st

def streamlit_interface(chain_rag):
    st.title('Perguntas sobre a Hotmart')
    st.write('Digite sua pergunta abaixo para obter uma resposta baseada no conteúdo.')

    user_question = st.text_input('Qual é a sua pergunta?')

    if user_question:
        with st.spinner('Gerando resposta...'):
            answer = chain_rag.invoke(user_question)  
        
        st.write('### Resposta:')
        st.write(answer)