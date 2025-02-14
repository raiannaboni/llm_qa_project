# Hotmart Q&A - Assistente Virtual

Este projeto é uma assistente virtual que responde perguntas sobre a empresa **Hotmart**, utilizando um modelo de **LLM** e **Vector Database** para armazenar e recuperar informações extraídas blog da Hotmart.

## 🛠 Tecnologias Utilizadas
- **Python 3.11.9**
- **LangChain** (para processamento e recuperação de dados)
- **Hugging Face** (para embeddings e LLM)
- **ChromaDB** (para armazenamento vetorial)
- **Streamlit** (para interface de usuário)
- **Docker** (para containerização)

## 🚀 Como Executar o Projeto

### 1️⃣ Clonar o repositório
```sh
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 3️⃣ Construir a imagem Docker
```sh
docker build -t hotmart_project .
```

### 4️⃣ Executar o container
```sh
docker run -p 8501:8501 --env-file .env hotmart_project
```

### 5️⃣ Acessar a aplicação
Abra o navegador e acesse:
👉 **http://localhost:8501**

## 📌 Funcionamento
1. O código carrega o conteúdo do site da Hotmart.
2. O texto é dividido em trechos menores para facilitar a recuperação.
3. Os trechos são armazenados em um **Vector Database** (ChromaDB).
4. Quando uma pergunta é feita, o sistema recupera os trechos mais relevantes.
5. A LLM (Meta-Llama 3-8B-Instruct) gera uma resposta baseada nos dados recuperados.



