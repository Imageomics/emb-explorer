#!/bin/bash
# Convenience script to setup and run the embedding explorer

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Embedding Explorer Setup Script${NC}"
echo "================================="

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo -e "${RED}Error: pyproject.toml not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Function to install dependencies
install_deps() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    
    # Check if uv is available
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}Using uv for fast installation...${NC}"
        uv pip install -e .
    else
        echo -e "${YELLOW}uv not found, falling back to pip...${NC}"
        # Check if pip is available
        if ! command -v pip &> /dev/null; then
            echo -e "${RED}Error: Neither uv nor pip is installed or not in PATH${NC}"
            echo -e "${YELLOW}To install uv (recommended): curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
            exit 1
        fi
        
        # Install the project in editable mode
        pip install -e .
    fi
    
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

# Function to list models
list_models() {
    echo -e "${YELLOW}Available models:${NC}"
    python list_models.py --format table
}

# Function to run the streamlit app
run_app() {
    echo -e "${YELLOW}Starting Streamlit application...${NC}"
    streamlit run app.py
}

# Main menu
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install    Install dependencies"
    echo "  models     List available models"
    echo "  run        Run the Streamlit app"
    echo "  help       Show this help message"
    echo ""
    echo "If no command is provided, the script will install dependencies and run the app."
}

# Parse command line arguments
case "${1:-}" in
    install)
        install_deps
        ;;
    models)
        list_models
        ;;
    run)
        run_app
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        # Default behavior: install and run
        install_deps
        echo ""
        echo -e "${GREEN}Setup complete! Starting the application...${NC}"
        echo ""
        run_app
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
