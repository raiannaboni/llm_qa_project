import streamlit as st


def streamlit_interface(chain_rag):
          
    with st.sidebar:
        st.image('hotmart.svg',
                 width=200)
        st.title('Quer saber tudo sobre a :red[Hotmart]?')
        st.header('A assistente virtual te ajuda!')
       

    # Inicializar histórico de mensagens
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Entrada do usuário
    if prompt := st.chat_input('Faça uma pergunta sobre a Hotmart!'):
        with st.chat_message('user'):
            st.markdown(prompt)

        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Obter resposta da LLM usando o VectorDB
        with st.chat_message('assistant'):
            response = chain_rag.invoke(prompt)
            st.markdown(response)

        # Adicionar resposta ao histórico
        st.session_state.messages.append({'role': 'assistant', 'content': response})



