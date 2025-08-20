#!/bin/bash

# Optimized Docker Hub push script using build-optimized.sh
# Usage: ./push-to-docker-hub.sh [image-name] [tag]

# Stop on error
set -e

# Load environment variables from .env file
if [ -f .env ]; then
    echo "🔧 Loading variables from .env..."
    # Enable automatic variable export
    set -o allexport
    source .env
    set +o allexport
    echo "✅ Variables loaded: IMAGE_NAME=$IMAGE_NAME, TAG=$TAG"
else
    echo "⚠️  .env file not found, using defaults"
fi

# Set default values if not defined in .env
IMAGE_NAME="${IMAGE_NAME:-runpod-comfyui}"
TAG="${TAG:-latest}"
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-}"
DOCKER_HUB_PASSWORD="${DOCKER_HUB_PASSWORD:-}"
HF_TOKEN="${HF_TOKEN:-}"

# Override with command line arguments if provided
if [ "$#" -eq 2 ]; then
    IMAGE_NAME="$1"
    TAG="$2"
    echo "📝 Command-line args: IMAGE_NAME=$IMAGE_NAME, TAG=$TAG"
elif [ "$#" -eq 1 ]; then
    IMAGE_NAME="$1"
    echo "📝 Command-line arg: IMAGE_NAME=$IMAGE_NAME"
fi

# Validate required variables
if [ -z "$DOCKER_HUB_USERNAME" ]; then
    echo "❌ DOCKER_HUB_USERNAME not set in .env"
    exit 1
fi

if [ -z "$DOCKER_HUB_PASSWORD" ]; then
    echo "❌ DOCKER_HUB_PASSWORD not set in .env"
    exit 1
fi

echo "🚀 Starting build & push process for $DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG"

# Check if already logged into Docker Hub
if docker info 2>/dev/null | grep -q "Username: $DOCKER_HUB_USERNAME"; then
    echo "✅ Already authenticated to Docker Hub as $DOCKER_HUB_USERNAME"
else
    echo "🔐 Authenticating on Docker Hub..."
    echo "$DOCKER_HUB_PASSWORD" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
fi

# Build the Docker image using the optimized build script
echo "🏗️  Building optimized Docker image..."

# Export environment variables for the build script
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Create temporary build script args
BUILD_ARGS=""
if [ -n "$HF_TOKEN" ]; then
    BUILD_ARGS="--build-arg HF_TOKEN=$HF_TOKEN"
fi

# Use the optimized build script approach but with our specific requirements
echo "📦 Building image: $DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG"

# Use BuildKit plain progress for more predictable, non-interactive logs.
docker build \
    --build-arg MODEL_DIFF="$MODEL_DIFF" \
    $BUILD_ARGS \
    --tag "$DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG" \
    --tag "$DOCKER_HUB_USERNAME/$IMAGE_NAME:latest" \
    --progress=plain \
    --platform linux/amd64 \
    .

# Verify the build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
else
    echo "❌ Build failed!"
    exit 1
fi

# Run a quick health check before pushing
echo "🧪 Running quick health check (no GPU binding to avoid host-specific hangs)..."
# Run without explicit GPU binding to avoid hangs on hosts where --gpus is unsupported.
if docker run --rm "$DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG" python3 -c "import torch,sys; print(f'PyTorch version: {torch.__version__}'); print('CUDA available:', torch.cuda.is_available()); sys.exit(0)" 2>/dev/null; then
    echo "✅ Health check passed!"
else
    echo "⚠️  Health check failed or produced errors; proceeding with push anyway..."
fi

# Push the Docker image to Docker Hub
echo "📤 Pushing Docker image to Docker Hub..."
docker push "$DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG"

# Also push the latest tag if we're not already using it
if [ "$TAG" != "latest" ]; then
    echo "📤 Pushing 'latest' tag..."
    docker push "$DOCKER_HUB_USERNAME/$IMAGE_NAME:latest"
fi

echo "🎉 Docker image built and pushed successfully!"
echo "📋 Image available as:"
echo "   - $DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG"
if [ "$TAG" != "latest" ]; then
    echo "   - $DOCKER_HUB_USERNAME/$IMAGE_NAME:latest"
fi

# Show image size
echo "📊 Image size:"
docker images "$DOCKER_HUB_USERNAME/$IMAGE_NAME:$TAG" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"