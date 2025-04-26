#!/bin/bash

# Exit on error
set -e

echo "================================================"
echo "QuickContractAI Setup Script"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is required but not installed. Please install pip3 and try again."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is required but not installed."
    echo "Please visit https://ollama.com/download to install Ollama and try again."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit the .env file with your API keys and Snowflake credentials."
fi

# Pull Ollama models
echo "Pulling required Ollama models..."
echo "This may take some time depending on your internet connection."

echo "Pulling TinyLlama (1.1B)..."
ollama pull tinyllama

echo "Pulling ShieldGemma (2B)..."
ollama pull shieldgemma

echo "Pulling Phi4 (14B)..."
ollama pull phi4

echo "================================================"
echo "Setup completed successfully!"
echo "To start the application, run:"
echo "source venv/bin/activate"
echo "streamlit run app.py"
echo "================================================"