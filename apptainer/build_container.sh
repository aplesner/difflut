#!/bin/bash
###############################################################################
# DiffLUT Container Build Script
#
# Simple wrapper for building the Apptainer/Singularity container.
# Handles CUDA architecture selection and validates the build.
#
# IMPORTANT: Must be run from the project root directory, not from apptainer/
#
# Usage (from project root):
#   bash apptainer/build_container.sh                    # Build with default arch (8.6)
#   bash apptainer/build_container.sh --arch 8.0         # Build for A100 (sm_80)
#   bash apptainer/build_container.sh --arch "7.5;8.6"   # Build for multiple architectures
#   bash apptainer/build_container.sh --clean            # Clean build (remove old container)
#
###############################################################################

set -e  # Exit on error

# Check that script is run from project root
if [ ! -f "setup.py" ] || [ ! -d "difflut" ] || [ ! -d "apptainer" ]; then
    echo "ERROR: This script must be run from the project root directory."
    echo ""
    echo "Current directory: $(pwd)"
    echo ""
    echo "Please run from project root:"
    echo "  cd /path/to/difflut"
    echo "  bash apptainer/build_container.sh"
    exit 1
fi

# Default settings
CUDA_ARCH="8.6"  # Default: RTX 3090, A6000
APPTAINER_DIR="apptainer"
CONTAINER_NAME="apptainer/difflut.sif"
DEF_FILE="apptainer/difflut.def"
CLEAN_BUILD=false
BUILD_LOG_DIR="apptainer/build_logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --arch ARCH    CUDA architecture(s) to build for (default: 8.6)"
            echo "                 Examples: 8.6, \"7.5;8.6\", \"6.1;7.0;7.5;8.0;8.6\""
            echo "  --clean        Remove existing container before building"
            echo "  --help         Show this help message"
            echo ""
            echo "CUDA Architecture Reference:"
            echo "  6.1 - Titan XP, GTX 1080 Ti"
            echo "  7.0 - Tesla V100"
            echo "  7.5 - RTX 2080 Ti, Titan RTX"
            echo "  8.0 - A100"
            echo "  8.6 - RTX 3090, A6000"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create build log directory
mkdir -p "$BUILD_LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${BUILD_LOG_DIR}/build_${TIMESTAMP}.log"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}  DiffLUT Apptainer Container Build${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "  ${GREEN}Container:${NC}     $CONTAINER_NAME"
echo -e "  ${GREEN}Definition:${NC}    $DEF_FILE"
echo -e "  ${GREEN}CUDA Arch:${NC}     $CUDA_ARCH"
echo -e "  ${GREEN}Build Log:${NC}     $LOG_FILE"
echo ""

# Check if apptainer/apptainer is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    echo -e "  ${GREEN}Tool:${NC}          apptainer ($(apptainer --version))"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    echo -e "  ${GREEN}Tool:${NC}          singularity ($(singularity --version))"
else
    echo -e "${RED}Error: Neither apptainer nor singularity found in PATH${NC}"
    echo "Please install apptainer or singularity to build containers."
    exit 1
fi

echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Clean existing container if requested
if [ "$CLEAN_BUILD" = true ]; then
    if [ -f "$CONTAINER_NAME" ]; then
        echo -e "${YELLOW}Removing existing container: $CONTAINER_NAME${NC}"
        rm -f "$CONTAINER_NAME"
    fi
fi

# Check if container already exists
if [ -f "$CONTAINER_NAME" ]; then
    echo -e "${YELLOW}Warning: Container $CONTAINER_NAME already exists.${NC}"
    echo -e "${YELLOW}Use --clean to remove it before building.${NC}"
    read -p "Continue anyway? This will overwrite the existing container. [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Build cancelled."
        exit 0
    fi
fi

# Export CUDA architecture for the build
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
echo -e "${GREEN}Building container with CUDA architectures: $CUDA_ARCH${NC}"
echo ""

# Build the container
echo -e "${BLUE}Starting build...${NC}"
echo -e "${BLUE}This may take 30-60 minutes depending on your system.${NC}"
echo -e "${BLUE}Build output is being logged to: $LOG_FILE${NC}"
echo ""

# Run build and tee output to log file
if $CONTAINER_CMD build "$CONTAINER_NAME" "$DEF_FILE" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${GREEN}✓ Container built successfully!${NC}"
    echo -e "${BLUE}=============================================================================${NC}"

    # Get container size
    CONTAINER_SIZE=$(du -h "$CONTAINER_NAME" | cut -f1)
    echo -e "  ${GREEN}Container:${NC}     $CONTAINER_NAME"
    echo -e "  ${GREEN}Size:${NC}          $CONTAINER_SIZE"
    echo -e "  ${GREEN}Build Log:${NC}     $LOG_FILE"
    echo ""

    # Validate the build
    echo -e "${BLUE}Validating container...${NC}"
    if $CONTAINER_CMD exec "$CONTAINER_NAME" python3 -c "import difflut; from difflut.nodes import LinearLUTNode; from difflut.layers import ConvolutionalLayer" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Container validation passed!${NC}"
        echo ""
        echo -e "${BLUE}=============================================================================${NC}"
        echo -e "${GREEN}Next Steps:${NC}"
        echo -e "${BLUE}=============================================================================${NC}"
        echo ""
        echo -e "  ${GREEN}1. Test the container locally:${NC}"
        echo "     $CONTAINER_CMD exec $CONTAINER_NAME python3 -c 'import difflut; print(difflut.__version__)'"
        echo ""
        echo -e "  ${GREEN}2. Test imports:${NC}"
        echo "     $CONTAINER_CMD exec $CONTAINER_NAME python3 -c 'from difflut.nodes import LinearLUTNode; from difflut.layers import ConvolutionalLayer; print(\"All imports successful\")'"
        echo ""
        echo -e "  ${GREEN}3. Sync to remote server:${NC}"
        echo "     cd .. && source helper_scripts/project_variables.sh"
        echo "     bash helper_scripts/sync_container_to_remote.sh"
        echo ""
    else
        echo -e "${RED}✗ Container validation failed!${NC}"
        echo -e "${YELLOW}Check the build log for errors: $LOG_FILE${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${RED}✗ Container build failed!${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${YELLOW}Check the build log for details: $LOG_FILE${NC}"
    exit 1
fi
