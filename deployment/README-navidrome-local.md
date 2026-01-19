# AudioMuse-AI Deployment for Intel N97 NAS

## Overview

This guide covers deploying AudioMuse-AI on an Intel N97 NAS with:
- Navidrome as the media server
- Intel QuickSync hardware acceleration
- Resource constraints (1 CPU core, 2GB RAM)

## Prerequisites

### System Requirements
- Intel N97 NAS with Docker support
- At least 4GB RAM available for containers
- At least 10GB storage space
- Docker and Docker Compose installed

### Network Requirements
- NAS accessible on the network
- Navidrome instance running and accessible
- Internet connection for DeepSeek API (cloud AI)

## Quick Start

### 1. Prepare the Configuration

```bash
# Navigate to deployment directory
cd deployment

# Copy the environment template
cp .env.navidrome-local .env

# Edit the environment file
nano .env
```

### 2. Configure Required Settings

Edit `.env` and set the following values:

```bash
# Navidrome Configuration
NAVIDROME_URL=http://your-navidrome-ip:4533
NAVIDROME_USER=your-username
NAVIDROME_PASSWORD=your-password

# DeepSeek API (for AI playlist naming)
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### 3. Deploy

```bash
# Start the services
docker compose -f docker-compose-navidrome-local.yaml up -d

# Check status
docker compose -f docker-compose-navidrome-local.yaml ps

# View logs
docker compose -f docker-compose-navidrome-local.yaml logs -f
```

### 4. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

## Configuration Details

### Resource Limits

Each service is configured with resource limits:

| Service | CPU | Memory |
|---------|-----|--------|
| Redis | 0.25 core | 256 MB |
| PostgreSQL | 0.25 core | 512 MB |
| Flask App | 1 core | 2 GB |
| Worker | 1 core | 2 GB |

### Music Library

The music library is mounted read-only at:
```
/volume1/music:/music:ro
```

**Important:** Ensure this path matches your NAS music directory.

### Intel QuickSync

Intel QuickSync is enabled for hardware-accelerated transcoding. The device is mapped:
```
/dev/dri:/dev/dri
```

Verify QuickSync is working:
```bash
# On the NAS
ls -la /dev/dri

# Inside the container
docker exec audiomuse-ai-flask-app vainfo
```

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker compose -f docker-compose-navidrome-local.yaml logs

# Common issues:
# 1. Port already in use - change FRONTEND_PORT in .env
# 2. Invalid Navidrome URL - verify NAVIDROME_URL
# 3. Missing API keys - check DEEPSEEK_API_KEY
```

### Poor Performance

If the NAS is slow:

1. **Disable CLAP text search** (saves ~1GB RAM):
   ```bash
   CLAP_ENABLED=false
   ```

2. **Reduce clustering runs**:
   ```bash
   CLUSTERING_RUNS=1000
   ```

3. **Fewer playlists**:
   ```bash
   TOP_N_PLAYLISTS=5
   ```

### Cannot Access Web Interface

```bash
# Check if Flask is running
docker exec audiomuse-ai-flask-app curl localhost:8000

# Check port mapping
docker port audiomuse-ai-flask-app
```

### Music Not Found

1. Verify music library path:
   ```bash
   ls -la /volume1/music
   ```

2. Check Navidrome is scanning the same library

3. Verify read permissions:
   ```bash
   docker exec -u root audiomuse-ai-flask-app ls /music
   ```

## Maintenance

### Update to Latest Version

```bash
# Pull latest images
docker compose -f docker-compose-navidrome-local.yaml pull

# Restart with new images
docker compose -f docker-compose-navidrome-local.yaml down
docker compose -f docker-compose-navidrome-local.yaml up -d
```

### Backup Data

```bash
# Backup PostgreSQL
docker exec audiomuse-postgres pg_dump -U audiomuse audiomusedb > backup.sql

# Backup Redis (optional)
docker exec audiomuse-redis redis-cli BGSAVE
docker cp audiomuse-redis:/data/dump.rdb ./redis-backup.rdb
```

### View Logs

```bash
# All services
docker compose -f docker-compose-navidrome-local.yaml logs

# Specific service
docker compose -f docker-compose-navidrome-local.yaml logs audiomuse-ai-flask-app

# Follow mode
docker compose -f docker-compose-navidrome-local.yaml logs -f
```

### Stop Services

```bash
# Stop but keep data
docker compose -f docker-compose-navidrome-local.yaml down

# Stop and remove data volumes
docker compose -f docker-compose-navidrome-local.yaml down -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Intel N97 NAS                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Navidrome │  │  Music Lib  │  │ AudioMuse-AI    │  │
│  │   :4533     │  │/volume1/music││  Flask :8000    │  │
│  └─────────────┘  └─────────────┘  │  Worker         │  │
│                                    └────────┬────────┘  │
│                     ┌───────────────────────┼────────┐  │
│                     │                       │        │  │
│               ┌─────┴─────┐          ┌──────┴────┐   │  │
│               │   Redis   │          │PostgreSQL │   │  │
│               │   :6379   │          │  :5432    │   │  │
│               └───────────┘          └───────────┘   │  │
└─────────────────────────────────────────────────────────┘
```

## Support

- GitHub Issues: https://github.com/neptunehub/AudioMuse-AI/issues
- Documentation: https://github.com/neptunehub/AudioMuse-AI/wiki
