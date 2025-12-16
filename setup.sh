#!/bin/bash
echo "Setting up PixStory project..."

# Create directories
mkdir -p models results data

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo "✅ Setup complete!"
echo "Run experiments with: ./run_all_experiments.sh"
