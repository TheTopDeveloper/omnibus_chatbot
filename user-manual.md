# Survey Data Chatbot
## User Manual

Welcome to the Survey Data Chatbot application! This tool allows you to analyze the International Media Group Omnibus Survey data using natural language queries. Ask questions about the survey results in plain English, and the chatbot will provide insights, statistics, and visualizations.

## Getting Started

### Accessing the Application
After installation, the application will be available at: http://localhost:8501

### Initial Setup
When you first open the application, it will:
1. Automatically detect and load survey files from the project folder
2. Display notifications of successfully loaded files
3. Present a clean interface with three main tabs

### Initializing the Chatbot
Before you can start asking questions:
1. Look for the "Initialize Chatbot" button in the sidebar
2. Click the button to set up the language model and knowledge base
3. Wait for the initialization process to complete (you'll see a success message)

## Interface Overview

The application has three main tabs:

### 1. Chatbot Tab
This is where you interact with the AI assistant:
- Type your questions in the text input box
- View the conversation history above the input box
- User messages appear with a 👤 icon, and bot responses with a 🤖 icon

### 2. Data Preview Tab
Explore the survey data directly:
- View a sample of the survey data in a table format
- See basic statistics about the dataset
- Check column information including data types and unique values
- Create visualizations of individual columns by selecting them from the dropdown

### 3. Survey Questionnaire Tab
Read the original survey questions:
- View the full text of the survey questionnaire
- Understand the context and structure of the survey

## Sidebar Options

The sidebar provides additional functionality:

### Data Management
- **Upload CSV File**: Replace the current survey data with your own CSV file
- **Upload Questionnaire**: Add or replace the survey questionnaire document

### Model Settings
- **Select LLM Model**: Choose between different language models:
  - google/flan-t5-base (default): Good balance of accuracy and speed
  - google/flan-t5-small: Faster responses but less detailed
  - stabilityai/stablelm-tuned-alpha-3b: More advanced but requires more resources

### Chat Management
- **Initialize Chatbot**: Set up or reset the chatbot with the current data
- **Clear Chat History**: Remove all previous messages from the conversation

## Asking Questions

### Example Questions
The sidebar contains example questions to help you get started:
- What percentage of respondents are male?
- What's the most popular social media platform?
- How many hours do people sleep on average?
- Which age group exercises the most?
- Compare media usage between urban and rural respondents

### Effective Question Techniques

For the best results:

1. **Be specific about what you want to know:**
   - Good: "What percentage of respondents aged 18-24 use Instagram?"
   - Less effective: "Tell me about social media usage"

2. **Specify demographic segments when relevant:**
   - Good: "Compare TV watching habits between urban and rural respondents"
   - Less effective: "How much TV do people watch?"

3. **Ask for comparisons or trends:**
   - Good: "Show me how sleep patterns vary by age group"
   - Less effective: "Tell me about sleep"

4. **Request specific statistics when needed:**
   - Good: "What are the top 3 favorite bread brands by percentage?"
   - Less effective: "What bread do people like?"

5. **Ask follow-up questions to drill deeper:**
   - First question: "What percentage exercise more than 4 hours weekly?"
   - Follow-up: "How does that break down by gender?"

## Working with Results

The chatbot will provide:

- **Statistical summaries**: Percentages, counts, averages, and other statistics
- **Comparisons**: Differences between demographic groups
- **Insights**: Patterns and trends in the data
- **Direct answers**: Specific responses to your questions

If you want to explore the data further:
1. Check the Data Preview tab to see the raw data
2. Ask follow-up questions to get more details
3. Request specific comparisons between different segments

## Tips and Best Practices

### For Optimal Performance
- Initialize the chatbot only when needed (when changing data or models)
- Use the smallest model that meets your needs for faster responses
- Clear chat history occasionally for better performance

### For Better Results
- Ask one question at a time for clearer answers
- Be specific about what information you need
- Use the example questions as templates for your own queries

### For Data Analysis
- Start with broad questions and then narrow down
- Ask for comparisons between different demographic groups
- Request percentage-based results for easier interpretation

## Troubleshooting

### Common Issues

**The chatbot isn't initialized:**
- Look for a message saying "Please initialize the chatbot first"
- Click the "Initialize Chatbot" button in the sidebar

**Slow responses:**
- Try using a smaller language model from the dropdown
- Close other applications to free up system resources
- Check your internet connection

**Error in responses:**
- Clear the chat history and try asking your question again
- Reinitialize the chatbot if errors persist
- Make sure your question is clear and specific

**Files not loading:**
- Check that file names match exactly what the application expects
- Try uploading the files manually through the sidebar options

## Getting Help

If you encounter issues not covered in this manual:
- Check the Implementation Guide for technical details
- Review the README for general information
- Contact the system administrator for assistance

---

© 2025 International Media Group - All rights reserved
