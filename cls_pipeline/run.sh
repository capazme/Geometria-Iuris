#!/bin/bash
# CLS Pipeline v2.0 - Run Script
# ===============================
# Usage: ./run.sh [OPTIONS]
#
# Options:
#   --rebuild     Force rebuild Docker image
#   --no-docker   Run locally without Docker
#   --open        Open visualization in browser after completion
#   --dev         Use development mode (mount source)
#   --info        Show pipeline info
#   --clear-cache Clear embedding cache
#   -h, --help    Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
REBUILD=false
NO_DOCKER=false
OPEN_BROWSER=false
DEV_MODE=false
COMMAND="run"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --no-docker)
            NO_DOCKER=true
            shift
            ;;
        --open)
            OPEN_BROWSER=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --info)
            COMMAND="info"
            shift
            ;;
        --clear-cache)
            COMMAND="clear-cache"
            shift
            ;;
        -h|--help)
            echo "CLS Pipeline v2.0 - Run Script"
            echo ""
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rebuild     Force rebuild Docker image"
            echo "  --no-docker   Run locally without Docker"
            echo "  --open        Open visualization in browser after completion"
            echo "  --dev         Use development mode (mount source)"
            echo "  --info        Show pipeline info"
            echo "  --clear-cache Clear embedding cache"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                   CLS Pipeline v2.0                       ║"
echo "║         Cross-Lingual Semantics Analysis                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for .env file
if [[ ! -f ".env" && -f ".env.example" ]]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env.example${NC}"
    cp .env.example .env
fi

# Source .env if exists
if [[ -f ".env" ]]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create required directories
mkdir -p data output cache models output/plots

# Check for legal_terms.json
if [[ ! -f "data/legal_terms.json" ]]; then
    echo -e "${RED}Error: data/legal_terms.json not found${NC}"
    echo "Please ensure the legal terms dataset is in the data directory."
    exit 1
fi

# Run without Docker
if [[ "$NO_DOCKER" == true ]]; then
    echo -e "${GREEN}Running locally without Docker...${NC}"

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found${NC}"
        exit 1
    fi

    # Check for virtual environment
    if [[ ! -d ".venv" ]]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv .venv
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt

    # Run pipeline
    echo -e "${GREEN}Running pipeline...${NC}"
    PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" python -m src.cli $COMMAND

    # Open browser if requested
    if [[ "$OPEN_BROWSER" == true && "$COMMAND" == "run" ]]; then
        VIZ_PATH="$SCRIPT_DIR/output/visualization.html"
        if [[ -f "$VIZ_PATH" ]]; then
            echo -e "${GREEN}Opening visualization...${NC}"
            if [[ "$OSTYPE" == "darwin"* ]]; then
                open "$VIZ_PATH"
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                xdg-open "$VIZ_PATH"
            fi
        fi
    fi

    exit 0
fi

# Docker mode
echo -e "${GREEN}Running with Docker...${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found. Install Docker or use --no-docker${NC}"
    exit 1
fi

# Build if needed
if [[ "$REBUILD" == true ]]; then
    echo -e "${YELLOW}Rebuilding Docker image...${NC}"
    docker-compose build --no-cache cls-pipeline
else
    # Build only if image doesn't exist
    if [[ -z "$(docker images -q cls-pipeline 2> /dev/null)" ]]; then
        echo -e "${YELLOW}Building Docker image...${NC}"
        docker-compose build cls-pipeline
    fi
fi

# Run container
SERVICE="cls-pipeline"
if [[ "$DEV_MODE" == true ]]; then
    SERVICE="cls-pipeline-dev"
    echo -e "${YELLOW}Using development mode...${NC}"
fi

echo -e "${GREEN}Starting container...${NC}"
docker-compose run --rm $SERVICE python -m src.cli $COMMAND

# Open browser if requested
if [[ "$OPEN_BROWSER" == true && "$COMMAND" == "run" ]]; then
    VIZ_PATH="$SCRIPT_DIR/output/visualization.html"
    if [[ -f "$VIZ_PATH" ]]; then
        echo -e "${GREEN}Opening visualization...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open "$VIZ_PATH"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xdg-open "$VIZ_PATH"
        fi
    fi
fi

echo -e "${GREEN}Done!${NC}"
