FROM python:3.11.9

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos necessários
COPY requirements.txt ./
COPY . ./

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta do Streamlit
EXPOSE 8501

# Comando para rodar a aplicação
CMD ["streamlit", "run", "./src/main.py"]
