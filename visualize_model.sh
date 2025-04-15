#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}     SVM MODEL VISUALIZATION TOOL     ${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}This will create visualizations in the 'visualizations' directory${NC}"
echo

# Check if the model file exists
if [ -f "claim_verifier_model.joblib" ]; then
    echo -e "${GREEN}Found model file. Starting visualization...${NC}"
    # Run the visualization script
    python model_visualizer.py
    
    # Check if visualization was successful by checking if the directory has files
    if [ "$(ls -A visualizations 2>/dev/null)" ]; then
        echo -e "\n${GREEN}Visualization complete! View your plots in the 'visualizations' directory.${NC}"
    else
        echo -e "\n${RED}Visualization may have failed. No files found in the 'visualizations' directory.${NC}"
    fi
else
    echo -e "${RED}Model file 'claim_verifier_model.joblib' not found.${NC}"
    echo -e "${YELLOW}Please run 'python claim_verifier.py' to train the model first.${NC}"
fi 