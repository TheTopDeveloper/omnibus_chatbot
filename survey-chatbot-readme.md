# Survey Data Chatbot

A natural language interface for querying and analyzing the International Media Group Omnibus Survey data, powered by LangChain and open-source LLMs.

## Overview

This application provides a chatbot interface that allows users to:

- Ask natural language questions about survey data
- Get AI-generated insights and summaries
- Visualize distributions and patterns
- Compare demographics and analyze trends
- All without needing to write code or SQL queries

The chatbot uses Hugging Face's open-source language models combined with LangChain's framework to create a conversational interface for data analysis.

## Features

- **Natural Language Interface**: Query your survey data using everyday language
- **Data Visualization**: View distributions and statistics for any survey question
- **Contextual Memory**: The chatbot remembers previous queries for follow-up questions
- **Example Questions**: Predefined examples to help users get started
- **CSV Upload**: Analyze your own survey data by uploading a CSV file
- **LLM Selection**: Choose from different open-source models based on your needs
- **Responsive UI**: Clean, intuitive interface built with Streamlit

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager

### Setup

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

### Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.22.0
pandas>=1.5.3
numpy>=1.23.5
matplotlib>=3.7.1
seaborn>=0.12.2
langchain>=0.0.267
faiss-cpu>=1.7.4
transformers>=4.30.2
sentence-transformers>=2.2.2
huggingface-hub>=0.15.1
accelerate>=0.20.3
protobuf>=3.20.3
einops>=0.6.1
```

## Usage

### Initializing the Chatbot

1. Launch the application using `streamlit run app.py`
2. (Optional) Upload your own CSV file using the uploader in the sidebar
3. Select the desired language model
4. Click "Initialize Chatbot" to set up the language model and vector store

### Asking Questions

Once the chatbot is initialized, you can ask questions about the survey data in natural language. For example:

- "What percentage of respondents are female?"
- "Which age group uses social media the most?"
- "What are the top 3 favorite brands of bread?"
- "Compare social media usage between urban and rural respondents"
- "What's the correlation between education level and exercise habits?"

### Data Preview

The Data Preview tab allows you to:

- View a sample of the loaded survey data
- See summary statistics for numerical columns
- View column information (data types, unique values, missing data)
- Visualize the distribution of any column

## Implementation Details

The application uses the following components:

1. **LangChain**: For building the conversational retrieval chain
2. **Hugging Face Models**: Providing open-source language models
3. **FAISS Vector Store**: For efficient similarity search of survey information
4. **Streamlit**: For the web interface

The workflow is as follows:

1. Survey data is loaded from a CSV file or sample data is generated
2. A description of the survey data is created along with statistics and sample records
3. This information is chunked and embedded in a vector store
4. A language model is loaded to power the chatbot
5. User questions are processed through a conversational retrieval chain
6. The chain retrieves relevant information from the vector store and generates responses

## Customization

### Using Larger Models

For better performance, you can use larger language models if you have the computational resources. To do this:

1. Select a larger model from the dropdown (or modify the code to include more options)
2. Ensure you have sufficient RAM and CPU/GPU resources
3. Be aware that larger models will take longer to initialize and generate responses

### Adding Custom Models

To add a custom model:

1. Add your model to the model selection dropdown in the sidebar
2. Modify the `setup_llm_and_vectorstore` function to handle your custom model
3. Ensure you have the required dependencies installed

### Extending the Data Analysis

To add more analysis capabilities:

1. Update the survey description generation in `generate_survey_description`
2. Add custom prompts for specific types of analysis
3. Include additional visualizations in the Data Preview tab

## Troubleshooting

### Common Issues

- **Memory Usage**: Large models may cause memory issues. Try using a smaller model or reducing the chunk size.
- **Slow Initialization**: The initial setup may take several minutes, especially for larger models.
- **CSV Loading Issues**: If your CSV has encoding issues, try preprocessing it before uploading.

### Performance Tips

- Use smaller models for faster responses
- Reduce the chunk size for large datasets
- Clear chat history periodically to manage memory usage

## License

This project is open-source and available under the MIT License.

---

Created for International Media Group - © 2025
