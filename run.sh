#!/bin/bash

# Colores para los mensajes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════╗"
echo "║     Sistema Greylabs EC RAG Bíblico        ║"
echo "║          Powered by Tiny Llama             ║"
echo "╚════════════════════════════════════════════╝"
echo -e "${NC}"

# Configuración del modelo
MODEL="llama3.2"

# Verificar Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Error: Ollama no está instalado${NC}"
    echo -e "${YELLOW}Por favor, instala Ollama primero: curl https://ollama.ai/install.sh | sh${NC}"
    exit 1
fi

# Iniciar Ollama si no está corriendo
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Iniciando Ollama...${NC}"
    ollama serve &
    sleep 5
fi

# Verificar modelo
if ! ollama list | grep -q "$MODEL"; then
    echo -e "${YELLOW}Descargando modelo tinyllama...${NC}"
    ollama pull $MODEL
fi

echo -e "${GREEN}Iniciando aplicación...${NC}"

# Ejecutar la aplicación
python app/main.py
