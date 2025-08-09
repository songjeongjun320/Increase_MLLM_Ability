#!/bin/bash
# ToW Model Server Startup Script

set -e

echo "Starting ToW Model Server..."
echo "Environment: $MLOPS_ENV"
echo "Host: $MODEL_SERVER_HOST"
echo "Port: $MODEL_SERVER_PORT"

# Wait for dependencies
echo "Waiting for dependencies..."

# Wait for database
if [ ! -z "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    python3 -c "
import time
import psycopg2
import os
from urllib.parse import urlparse

url = os.environ['DATABASE_URL']
parsed = urlparse(url)

for i in range(30):
    try:
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            database=parsed.path[1:],
            user=parsed.username,
            password=parsed.password
        )
        conn.close()
        print('Database connected successfully')
        break
    except Exception as e:
        print(f'Attempt {i+1}/30: Database not ready: {e}')
        time.sleep(5)
else:
    print('Failed to connect to database after 30 attempts')
    exit(1)
"
fi

# Wait for Redis
if [ ! -z "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    python3 -c "
import time
import redis
import os
from urllib.parse import urlparse

url = os.environ['REDIS_URL']
parsed = urlparse(url)

for i in range(30):
    try:
        r = redis.Redis(
            host=parsed.hostname,
            port=parsed.port,
            password=parsed.password,
            socket_timeout=5
        )
        r.ping()
        print('Redis connected successfully')
        break
    except Exception as e:
        print(f'Attempt {i+1}/30: Redis not ready: {e}')
        time.sleep(2)
else:
    print('Failed to connect to Redis after 30 attempts')
    exit(1)
"
fi

# Initialize MLOps configuration
echo "Initializing MLOps configuration..."
python3 -c "
from mlops.config import load_config
config = load_config()
print(f'Configuration loaded for environment: {config.environment.value}')
"

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)')
"

# Create necessary directories
mkdir -p /app/logs /app/models /app/monitoring_charts

# Set default configuration file if not provided
if [ -z "$MLOPS_CONFIG_FILE" ]; then
    export MLOPS_CONFIG_FILE="/app/mlops/configs/${MLOPS_ENV}.yaml"
fi

# Start the model server
echo "Starting ToW Model Server..."

# Choose startup method based on configuration
if [ "$MLOPS_ENV" = "production" ]; then
    # Production: Use Gunicorn with multiple workers
    exec gunicorn \
        --bind $MODEL_SERVER_HOST:$MODEL_SERVER_PORT \
        --workers ${MODEL_SERVER_WORKERS:-4} \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 300 \
        --keep-alive 5 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --log-level info \
        "mlops.deployment:create_model_server(load_config()).app"
else
    # Development: Use Uvicorn directly
    exec python3 -c "
import asyncio
import uvicorn
from mlops.config import load_config
from mlops.deployment import create_model_server

config = load_config()
server = create_model_server(config)

uvicorn.run(
    server.app,
    host='$MODEL_SERVER_HOST',
    port=int('$MODEL_SERVER_PORT'),
    log_level='info',
    access_log=True,
    reload=False
)
"
fi