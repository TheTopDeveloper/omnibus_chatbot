# Survey Data Chatbot - Deployment Guide

This guide provides detailed instructions for deploying, configuring, and using the Survey Data Chatbot application that allows natural language querying of the International Media Group Omnibus Survey data.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Options](#installation-options)
3. [Configuration](#configuration)
4. [Working with Survey Data](#working-with-survey-data)
5. [Model Selection and Performance](#model-selection-and-performance)
6. [Security Considerations](#security-considerations)
7. [Deployment Environments](#deployment-environments)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

For running with smaller models (recommended for most deployments):
- **CPU**: 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 5GB for application and models

For running with larger models (for enhanced capabilities):
- **CPU**: 8+ cores
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Disk Space**: 10GB+ for application and models

### Software Requirements

- **Operating System**: Linux (recommended), macOS, or Windows
- **Python**: Version 3.8 or higher
- **CUDA**: Version 11.2+ (if using GPU acceleration)

## Installation Options

### Option 1: Local Installation (Recommended for Development)

1. Clone the repository or download the application code:
   ```bash
   git clone https://github.com/your-org/survey-chatbot.git
   cd survey-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Option 2: Docker Deployment (Recommended for Production)

1. Create a Dockerfile:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t survey-chatbot .
   docker run -p 8501:8501 survey-chatbot
   ```

3. Access the application at http://localhost:8501

### Option 3: Cloud Deployment

#### Streamlit Cloud

1. Push your code to a GitHub repository
2. Sign up for Streamlit Cloud (https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the application with a few clicks

#### Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. Push your code to GitHub and deploy through the Heroku dashboard or CLI

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```
# Hugging Face settings
#HUGGINGFACE_API_TOKEN='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'  # Optional, for accessing gated models

# Application settings
DEBUG=False
ENABLE_TELEMETRY=False
MAX_CONTEXT_SIZE=5000
DEFAULT_MODEL=google/flan-t5-base
```

### Configuration File

Alternatively, create a `config.yaml` file in the application directory:

```yaml
# Model settings
models:
  default: "google/flan-t5-base"
  available:
    - "google/flan-t5-base"
    - "google/flan-t5-small"
    - "stabilityai/stablelm-tuned-alpha-3b"
    
# Vector store settings
vector_store:
  chunk_size: 1000
  chunk_overlap: 100
  
# Application settings
debug: false
max_token_limit: 4096
```

## Working with Survey Data

### Preparing CSV Files

For optimal performance, prepare your CSV files as follows:

1. **Headers**: Ensure column headers are clear and descriptive
2. **Encoding**: Use UTF-8 encoding
3. **Data Cleaning**: Remove duplicate entries and handle missing values
4. **Size Limitations**: Keep files under 100MB for best performance

### Data Structure Recommendations

The application works best with survey data that follows this structure:

1. **Demographics columns**: Placed first in the CSV
2. **Question responses**: One column per survey question
3. **Consistent values**: Use consistent formats for similar data types
4. **Descriptive labels**: Use descriptive labels instead of numeric codes

### Sample Data

The application includes sample data based on the Omnibus Survey structure. You can use this to test the application before uploading your own data.

## Model Selection and Performance

### Available Models

The application supports several open-source language models:

1. **google/flan-t5-small** (~300MB)
   - Fast responses, lower accuracy
   - Minimal resource requirements
   - Best for basic queries

2. **google/flan-t5-base** (~1GB)
   - Balanced performance
   - Moderate resource requirements
   - Good for most use cases

3. **stabilityai/stablelm-tuned-alpha-3b** (~3GB)
   - High-quality responses
   - Higher resource requirements
   - Best for complex analysis

### Model Performance Comparison

| Model | Initialization Time | Response Time | Memory Usage | Quality |
|-------|---------------------|---------------|--------------|---------|
| flan-t5-small | ~20 seconds | 1-3 seconds | ~1GB | Basic |
| flan-t5-base | ~40 seconds | 3-6 seconds | ~2GB | Good |
| stablelm-3b | ~90 seconds | 5-10 seconds | ~6GB | Excellent |

### Adding Custom Models

To add a custom model:

1. Add your model to the `model_option` selectbox in the code
2. Implement the loading logic in the `setup_llm_and_vectorstore` function
3. Adjust the resource requirements accordingly

## Security Considerations

### Data Privacy

- The application processes all data locally
- No data is sent to external servers (except when using Hugging Face API)
- Consider implementing authentication for multi-user deployments

### Model Security

- Open-source models are downloaded from Hugging Face
- Verify model sources and permissions before deployment
- Consider using model validation to prevent misuse

### Deployment Security

- For production deployments, use HTTPS
- Implement authentication for access control
- Consider deploying behind a reverse proxy

## Deployment Environments

### Development Environment

For local development and testing:

```bash
streamlit run app.py
```

### Testing Environment

For team testing and validation:

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Production Environment

For production deployment:

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=true
```

Using process manager (PM2):

```bash
pm2 start --name survey-chatbot -- streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## Troubleshooting

### Common Issues

#### Application won't start

Check:
- Python version (3.8+ required)
- Required dependencies installed
- Port 8501 available

#### Model initialization fails

Check:
- Sufficient RAM available
- Internet connection (for downloading models)
- Hugging Face token (if using gated models)

#### Slow performance

Solutions:
- Use a smaller model
- Reduce vector store chunk size
- Increase available memory
- Use GPU acceleration (if available)

#### Out of memory errors

Solutions:
- Use a smaller model
- Reduce batch size in model configuration
- Implement gradient checkpointing
- Clear memory using `torch.cuda.empty_cache()`

### Support Resources

- GitHub Repository: [github.com/your-org/survey-chatbot](https://github.com/your-org/survey-chatbot)
- Documentation: [your-org.github.io/survey-chatbot](https://your-org.github.io/survey-chatbot)
- Issue Tracker: [github.com/your-org/survey-chatbot/issues](https://github.com/your-org/survey-chatbot/issues)

For additional support, contact: support@your-org.com

---

Created for International Media Group - © 2025