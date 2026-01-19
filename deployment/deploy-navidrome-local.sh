#!/bin/bash
#===============================================================================
# AudioMuse-AI Quick Start Script for N97 NAS
# Target: Intel N97 NAS with Navidrome and Intel QuickSync
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose-navidrome-local.yaml"
ENV_FILE=".env.navidrome-local"
ENV_TARGET=".env"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  AudioMuse-AI Quick Start Script${NC}"
echo -e "${BLUE}  N97 NAS + Navidrome + Intel QuickSync${NC}"
echo -e "${BLUE}============================================${NC}"
echo

#-------------------------------------------------------------------------------
# Check prerequisites
#-------------------------------------------------------------------------------
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is installed${NC}"

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}Error: $COMPOSE_FILE not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose file found${NC}"

echo

#-------------------------------------------------------------------------------
# Setup environment file
#-------------------------------------------------------------------------------
echo -e "${YELLOW}Setting up environment file...${NC}"

if [ -f "$ENV_TARGET" ]; then
    echo -e "${YELLOW}Warning: $ENV_TARGET already exists, backing up...${NC}"
    cp "$ENV_TARGET" "$ENV_TARGET.backup.$(date +%Y%m%d%H%M%S)"
fi

if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_TARGET"
    echo -e "${GREEN}✓ Copied $ENV_FILE to $ENV_TARGET${NC}"
else
    echo -e "${RED}Error: $ENV_FILE not found${NC}"
    exit 1
fi

echo

#-------------------------------------------------------------------------------
# User configuration
#-------------------------------------------------------------------------------
echo -e "${YELLOW}Configuration required:${NC}"
echo

read -p "Enter your Navidrome URL (e.g., http://192.168.1.100:4533): " NAVIDROME_URL
read -p "Enter your Navidrome username: " NAVIDROME_USER
read -s -p "Enter your Navidrome password: " NAVIDROME_PASSWORD
echo

read -s -p "Enter your DeepSeek API Key: " DEEPSEEK_API_KEY
echo

# Update .env file
sed -i "s|http://your-navidrome-host:4533|$NAVIDROME_URL|g" "$ENV_TARGET"
sed -i "s|your-navidrome-username|$NAVIDROME_USER|g" "$ENV_TARGET"
sed -i "s|your-navidrome-password|$NAVIDROME_PASSWORD|g" "$ENV_TARGET"
sed -i "s|your-deepseek-api-key|$DEEPSEEK_API_KEY|g" "$ENV_TARGET"

echo -e "${GREEN}✓ Environment file updated${NC}"
echo

#-------------------------------------------------------------------------------
# Pull latest images
#-------------------------------------------------------------------------------
echo -e "${YELLOW}Pulling latest Docker images...${NC}"
docker compose -f "$COMPOSE_FILE" pull
echo -e "${GREEN}✓ Images pulled${NC}"
echo

#-------------------------------------------------------------------------------
# Start services
#-------------------------------------------------------------------------------
echo -e "${YELLOW}Starting AudioMuse-AI services...${NC}"
docker compose -f "$COMPOSE_FILE" up -d
echo

#-------------------------------------------------------------------------------
# Wait for services to be healthy
#-------------------------------------------------------------------------------
echo -e "${YELLOW}Waiting for services to be ready...${NC}"

# Wait for Redis
echo -n "  Redis: "
until docker exec audiomuse-redis redis-cli ping &> /dev/null; do
    echo -n "."
    sleep 2
done
echo -e "${GREEN}Ready${NC}"

# Wait for PostgreSQL
echo -n "  PostgreSQL: "
until docker exec audiomuse-postgres pg_isready -U audiomuse &> /dev/null; do
    echo -n "."
    sleep 2
done
echo -e "${GREEN}Ready${NC}"

# Wait for Flask app
echo -n "  Flask App: "
until docker exec audiomuse-ai-flask-app curl -sf http://localhost:8000/api/config &> /dev/null; do
    echo -n "."
    sleep 5
done
echo -e "${GREEN}Ready${NC}"

echo

#-------------------------------------------------------------------------------
# Display status
#-------------------------------------------------------------------------------
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  AudioMuse-AI is now running!${NC}"
echo -e "${GREEN}============================================${NC}"
echo

echo -e "${BLUE}Access the web interface:${NC}"
echo -e "  URL: http://localhost:8000"
echo

echo -e "${BLUE}Service Status:${NC}"
docker compose -f "$COMPOSE_FILE" ps
echo

echo -e "${BLUE}View logs:${NC}"
echo -e "  docker compose -f $COMPOSE_FILE logs -f"
echo

echo -e "${BLUE}Stop services:${NC}"
echo -e "  docker compose -f $COMPOSE_FILE down"
echo

echo -e "${YELLOW}Note:${NC}"
echo -e "  - First analysis may take several minutes"
echo -e "  - Check logs if services don't start properly"
echo -e "  - Configure AI provider in .env file if needed"
echo
