BLUE='\033[0;34m'
NC='\033[0m'

lint:
        @echo "\n${BLUE}Running Flake8 against source...${NC}\n"
	    @flake8