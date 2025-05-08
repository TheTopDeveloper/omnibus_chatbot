import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import logging
import traceback
import gc
import torch
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader, DataFrameLoader, PyPDFLoader, Docx2txtLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import login

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clear_memory():
    """Clear memory and cache"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def log_error(error_msg, error=None):
    """Log error messages to both file and Streamlit"""
    logging.error(error_msg)
    if error:
        logging.error(traceback.format_exc())
    st.error(error_msg)

# Set page title and layout
st.set_page_config(
    page_title="Survey Data Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS for improved appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
    }
    .chat-message.user {
        background-color: #F0F2F6;
    }
    .chat-message.bot {
        background-color: #E1F5FE;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .footer {
        font-size: 0.8rem;
        color: #9e9e9e;
        text-align: center;
        margin-top: 2rem;
    }
    .status-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-message.success {
        background-color: #DCEDC8;
        border-left: 0.3rem solid #8BC34A;
    }
    .status-message.info {
        background-color: #E1F5FE;
        border-left: 0.3rem solid #03A9F4;
    }
    .status-message.warning {
        background-color: #FFF9C4;
        border-left: 0.3rem solid #FFEB3B;
    }
    .status-message.error {
        background-color: #FFEBEE;
        border-left: 0.3rem solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<div class="main-header">Survey Data Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query the Omnibus Survey using natural language</div>', unsafe_allow_html=True)

# Create tabs for chatbot and data view
tab1, tab2, tab3 = st.tabs(["Chatbot", "Data Preview", "Survey Questionnaire"])

# Initialize session state for chat history if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'df' not in st.session_state:
    st.session_state.df = None

if 'survey_desc' not in st.session_state:
    st.session_state.survey_desc = None

if 'questionnaire_text' not in st.session_state:
    st.session_state.questionnaire_text = None

if 'initialization_status' not in st.session_state:
    st.session_state.initialization_status = None

if 'data_load_complete' not in st.session_state:
    st.session_state.data_load_complete = False

if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False

# File paths for survey data and questionnaire
SURVEY_DATA_PATH = "Omnibus  survey results - January 2025.csv"
QUESTIONNAIRE_PATH = "Omnibus Survey Questionnaire - January 8 2025 - commented and updated  v 2.docx"

# Function to load the questionnaire document
def load_questionnaire():
    """Load the questionnaire document from the project folder"""
    try:
        # Check if docx file exists
        if os.path.exists(QUESTIONNAIRE_PATH):
            # Use Docx2txtLoader to extract text
            loader = Docx2txtLoader(QUESTIONNAIRE_PATH)
            documents = loader.load()
            questionnaire_text = documents[0].page_content
            return questionnaire_text, True
        else:
            return None, False
    except Exception as e:
        st.error(f"Error loading questionnaire: {str(e)}")
        return None, False

# Function to load the CSV data
def load_csv_data():
    """Load the CSV data from the project folder"""
    try:
        # Check if CSV file exists
        if os.path.exists(SURVEY_DATA_PATH):
            # Read CSV with explicit dtype handling
            df = pd.read_csv(
                SURVEY_DATA_PATH,
                low_memory=False,
                dtype_backend='numpy_nullable',
                # Convert all integer columns to float to avoid Int64Dtype issues
                dtype={col: 'float64' for col in pd.read_csv(SURVEY_DATA_PATH, nrows=0).columns if 'ID' in col}
            )
            
            # Convert any remaining Int64Dtype columns to float64
            for col in df.select_dtypes(include=['Int64']).columns:
                df[col] = df[col].astype('float64')
            
            # Convert any object columns that contain only numbers to float64
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    continue
            
            return df, True
        else:
            return None, False
    except Exception as e:
        st.error(f"Error loading CSV data: {str(e)}")
        return None, False

# Function to load sample data if no CSV is uploaded or found
def load_sample_data():
    """Create sample data based on the questionnaire"""
    # Creating sample data
    n_samples = 100
    
    # Basic demographics
    age_groups = ['18-24', '25-29', '30-34', '35-39', '40-49', '50-59', '60+']
    age_dist = [0.18, 0.17, 0.16, 0.15, 0.14, 0.12, 0.08]
    
    genders = ['Male', 'Female']
    gender_dist = [0.51, 0.49]
    
    locations = ['Urban', 'Peri-Urban', 'Rural']
    location_dist = [0.45, 0.25, 0.3]
    
    education_levels = ['No education', 'Pre-primary', 'Primary', 'Secondary/High School', 
                       'Tertiary College', "Bachelor's degree", "Master's Degree", 'PhD']
    edu_dist = [0.05, 0.03, 0.15, 0.35, 0.25, 0.12, 0.04, 0.01]
    
    # Sample data dictionary
    data = {
        'ID': np.arange(1, n_samples + 1),
        'Age': np.random.choice(age_groups, size=n_samples, p=age_dist),
        'Gender': np.random.choice(genders, size=n_samples, p=gender_dist),
        'Location': np.random.choice(locations, size=n_samples, p=location_dist),
        'Education': np.random.choice(education_levels, size=n_samples, p=edu_dist)
    }
    
    # Media habits
    media_types = ['Radio', 'TV', 'Social Media', 'Web', 'Podcasts', 'Music Streaming', 'Video Services']
    
    for media in media_types:
        data[f'Used_{media}_Yesterday'] = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
        data[f'Used_{media}_Week'] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
        
        time_spent = ['<30 Mins', '30 Mins -- 1 Hrs', '1-2 Hrs', '2-4 Hrs', '4-6 Hrs', '6+ Hrs']
        data[f'Time_{media}'] = np.random.choice(time_spent, size=n_samples)
    
    # Social media platforms
    platforms = ['Instagram', 'Snapchat', 'Facebook', 'Youtube', 'Twitter/X', 'TikTok', 'Whatsapp', 'Telegram']
    
    for platform in platforms:
        data[f'Use_{platform}'] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    data['Favorite_Platform'] = np.random.choice(platforms, size=n_samples)
    
    # Life aspirations
    aspirations = [
        'See and travel the world', 'Earn high salary/be wealthy', 'Buy/Build my own home',
        'Make a positive impact on community/society', 'Have children, start family',
        'Start my own business', 'Enter politics', 'Becoming Spiritually mature',
        'Finishing my Education', 'Having a career', 'Being Financially independent',
        'Caring for the poor and needy in the society', 'Losing Weight',
        'Becoming an influencer/content creator', 'Finding Love', 'Becoming Popular'
    ]
    
    for aspiration in aspirations:
        data[f'Aspiration_{aspiration}'] = np.random.choice(['Very Important', 'Important', 'Neutral', 'Not Important', 'Not Important at all'], size=n_samples)
    
    # Other metrics
    data['Sleep_Hours'] = np.random.choice(['1-2Hours', '3-5 Hours', '6-7 Hours', '8+'], size=n_samples, p=[0.05, 0.35, 0.45, 0.15])
    data['Eat_Fruits_Vegetables'] = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.6, 0.4])
    data['Exercise_Weekly'] = np.random.choice(['None', '1-2 Hours', '3-4 Hours', '5-10 Hours', 'Above 10 Hours'], size=n_samples)
    data['Weight_Concern'] = np.random.choice(['Not concerned at all', 'Not concerned', 'Neutral', 'Fairly Concerned', 'Very Concerned'], size=n_samples)
    
    # Brands
    cooking_oil_brands = ['Fresh Fri', 'Elianto', 'Golden Fry', 'Kimbo', 'Rina', 'Salit', 'Top Fri']
    data['Favorite_Cooking_Oil'] = np.random.choice(cooking_oil_brands, size=n_samples)
    
    bread_brands = ['Supa Loaf', 'Festive', 'Broadways', 'Kenblest', 'Loafa', 'Butter Toast']
    data['Favorite_Bread'] = np.random.choice(bread_brands, size=n_samples)
    
    telecom_companies = ['Safaricom', 'Airtel', 'Telkom Kenya']
    data['Favorite_Telecom'] = np.random.choice(telecom_companies, size=n_samples, p=[0.7, 0.25, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

# Function to generate survey description for the LLM context
def generate_survey_description(df, questionnaire_text=None):
    """Generate a description of the survey data for the LLM context"""
    
    # Get basic info about the dataframe
    num_respondents = len(df)
    num_columns = len(df.columns)
    
    # Start with basic description
    description = f"""
    This is an analysis of the International Media Group Omnibus Survey data from January 2025.
    The survey contains responses from {num_respondents} participants across {num_columns} questions/variables.
    
    Here's a summary of the key sections in the survey:
    
    1. Demographics: Age, Gender, Location, Education
    
    2. Media Habits: 
       - Media consumption yesterday and in the past week (Radio, TV, Social Media, Web, Podcasts, Music Streaming, Video Services)
       - Time spent on different media types
       - Social media platform usage
       - Favorite social media platforms
    
    3. Lifestyle and Consumption Habits:
       - Sleep patterns
       - Diet (fruit and vegetable consumption)
       - Exercise habits
       - Weight concerns
    
    4. Life Aspirations and Priorities:
       - Importance ratings for various life goals and aspirations
    
    5. Brand Preferences:
       - Favorite brands across different product categories
    
    The data columns include:
    """
    
    # Add column descriptions
    for col in df.columns:
        # Skip ID column
        if col == 'ID':
            continue
            
        # Get unique values and their counts
        value_counts = df[col].value_counts()
        top_values = value_counts.head(3).to_dict()
        
        # Format the top values
        top_values_str = ", ".join([f"'{k}': {v}" for k, v in top_values.items()])
        
        # Add column description to the overall description
        description += f"\n- {col}: {len(value_counts)} unique values. Top values: {top_values_str}"
    
    # Add questionnaire information if available
    if questionnaire_text:
        description += f"\n\nThe survey questionnaire contains the following information:\n{questionnaire_text[:2000]}...(truncated)"
    
    return description

# Function to setup the language model and vector store
def setup_llm_and_vectorstore(df, survey_desc):
    """Set up the language model and vector store for the chatbot"""
    try:
        with st.spinner("Setting up the language model and vector store... This might take a few minutes."):
            logging.info("Starting LLM and vector store setup")
            
            # Clear memory before starting
            clear_memory()
            
            # Create detailed data analysis with reduced memory usage
            data_analysis = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    value_counts = df[col].value_counts()
                    percentages = (value_counts / len(df) * 100).round(2)
                    analysis = f"\nAnalysis of {col}:\n"
                    for value, count in value_counts.items():
                        percentage = percentages[value]
                        analysis += f"- {value}: {count} respondents ({percentage}%)\n"
                    data_analysis.append(analysis)
                    # Clear memory after each column
                    del value_counts, percentages
                    clear_memory()
            
            logging.info("Data analysis completed")
            
            # Create a more concise data summary
            data_summary = f"""
            Survey Data Analysis:
            Total Respondents: {len(df)}
            Key Variables: {', '.join(df.columns[:10])}...
            """
            
            # Combine information with reduced memory usage
            combined_text = f"{survey_desc}\n\n{data_summary}"
            
            logging.info("Creating document from text")
            documents = [combined_text]
            
            # Split text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Reduced chunk size
                chunk_overlap=50  # Reduced overlap
            )
            texts = text_splitter.create_documents(documents)
            
            # Clear memory before creating embeddings
            clear_memory()
            
            logging.info("Creating embeddings")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            logging.info("Creating vector store")
            vector_store = FAISS.from_documents(texts, embeddings)
            
            # Clear memory before loading model
            clear_memory()
            
            logging.info("Setting up language model")
            model_id = "google/flan-t5-base"  # Using base model for stability
            
            # Load model with reduced memory usage
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,  # Reduced max length
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                device_map='auto'  # Let the pipeline decide the best device
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            
            logging.info("Creating memory and chain")
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=1000  # Limit memory size
            )
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Reduced k
                memory=memory,
                verbose=True
            )
            
            logging.info("Setup completed successfully")
            return vector_store, chain
            
    except Exception as e:
        error_msg = f"Error setting up the language model: {str(e)}"
        log_error(error_msg, e)
        return None, None
    finally:
        # Clear memory after setup
        clear_memory()

# Function to display status message
def show_status(message, status_type="info"):
    """Display a status message with appropriate styling"""
    st.markdown(f'<div class="status-message {status_type}">{message}</div>', unsafe_allow_html=True)

# Load data and questionnaire in the background if not already loaded
if not st.session_state.data_load_complete:
    try:
        with st.spinner("Loading survey data and questionnaire..."):
            logging.info("Starting data loading process")
            
            # Try to load CSV data
            df, csv_loaded = load_csv_data()
            if csv_loaded:
                st.session_state.df = df
                show_status("✅ Survey data loaded successfully", "success")
                logging.info("CSV data loaded successfully")
            else:
                # Use sample data if CSV not found
                df = load_sample_data()
                st.session_state.df = df
                show_status("⚠️ Survey data file not found. Using sample data instead.", "warning")
                logging.info("Using sample data")
            
            # Clear memory after data loading
            clear_memory()
            
            # Try to load questionnaire
            questionnaire_text, questionnaire_loaded = load_questionnaire()
            if questionnaire_loaded:
                st.session_state.questionnaire_text = questionnaire_text
                show_status("✅ Survey questionnaire loaded successfully", "success")
                logging.info("Questionnaire loaded successfully")
            else:
                st.session_state.questionnaire_text = None
                show_status("⚠️ Survey questionnaire file not found", "warning")
                logging.info("Questionnaire file not found")
            
            # Generate survey description
            survey_desc = generate_survey_description(st.session_state.df, st.session_state.questionnaire_text)
            st.session_state.survey_desc = survey_desc
            
            # Mark data loading as complete
            st.session_state.data_load_complete = True
            logging.info("Data loading process completed")
            
    except Exception as e:
        error_msg = f"Error during data loading: {str(e)}"
        log_error(error_msg, e)
        st.session_state.data_load_complete = False
    finally:
        # Clear memory after data loading
        clear_memory()

# Sidebar content
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.markdown("### Model Settings")
    model_option = st.selectbox(
        "Select LLM Model",
        ["google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small"],
        index=0,
        help="Larger models provide better responses but require more resources."
    )
    
    # Initialize/reset the chatbot
    if st.button("Initialize Chatbot"):
        # Setup the language model and vector store
        vector_store, chain = setup_llm_and_vectorstore(st.session_state.df, st.session_state.survey_desc)
        
        if vector_store is not None and chain is not None:
            st.session_state.vector_store = vector_store
            st.session_state.chain = chain
            st.session_state.model_initialized = True
            show_status("✅ Chatbot initialized successfully!", "success")
        else:
            show_status("❌ Failed to initialize chatbot. Please try again.", "error")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if 'memory' in st.session_state and st.session_state.chain is not None:
            st.session_state.chain.memory.clear()
        show_status("✅ Chat history cleared!", "success")
    
    st.markdown("### Example Questions")
    st.markdown("""
    Try asking questions like:
    - What percentage of respondents are male?
    - What's the most popular social media platform?
    - How many hours do people sleep on average?
    - Which age group exercises the most?
    - What are the top life aspirations for people under 30?
    - Compare media usage between urban and rural respondents.
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Chat interface in the first tab
with tab1:
    # Check if chain is initialized
    if not st.session_state.model_initialized:
        st.info("Please initialize the chatbot first by clicking the 'Initialize Chatbot' button in the sidebar.")
    else:
        # Initialize chat history if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f'<div class="chat-message user"><div class="avatar">👤</div><div class="message">{message}</div></div>', unsafe_allow_html=True)
            else:  # Bot message
                st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{message}</div></div>', unsafe_allow_html=True)
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question about the survey data:", key="question_input")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append(user_input)
                
                # Get response from chain
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chain({"question": user_input})
                        bot_response = response['answer']
                        
                        # Add bot response to chat history
                        st.session_state.chat_history.append(bot_response)
                        
                        # Force a rerun to update the display
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
                        # Remove the user message if we couldn't get a response
                        st.session_state.chat_history.pop()

# Data preview tab
with tab2:
    if st.session_state.df is not None:
        st.markdown("### Survey Data Preview")
        try:
            # Convert DataFrame to string representation for display
            st.text(st.session_state.df.head(10).to_string())
        except Exception as e:
            st.error(f"Error displaying data preview: {str(e)}")
        
        st.markdown("### Data Statistics")
        st.markdown("#### Basic Information")
        st.write(f"Number of respondents: {len(st.session_state.df)}")
        st.write(f"Number of questions/variables: {len(st.session_state.df.columns)}")
        
        st.markdown("#### Summary Statistics")
        try:
            # Only include numeric columns for describe
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.text(st.session_state.df[numeric_cols].describe().to_string())
        except Exception as e:
            st.error(f"Error displaying summary statistics: {str(e)}")
        
        # Display column information
        st.markdown("#### Column Information")
        try:
            column_info = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Data Type': st.session_state.df.dtypes,
                'Unique Values': [st.session_state.df[col].nunique() for col in st.session_state.df.columns],
                '% Missing': [(st.session_state.df[col].isna().sum() / len(st.session_state.df)) * 100 for col in st.session_state.df.columns]
            })
            st.text(column_info.to_string())
        except Exception as e:
            st.error(f"Error displaying column information: {str(e)}")
        
        # Allow user to select a column to view distribution
        st.markdown("#### Column Distribution")
        try:
            selected_column = st.selectbox("Select a column to view its distribution:", st.session_state.df.columns)
            
            if selected_column:
                # Create distribution plot based on column type
                if st.session_state.df[selected_column].dtype in ['int64', 'float64']:
                    # Numeric column - histogram
                    fig, ax = plt.subplots()
                    sns.histplot(st.session_state.df[selected_column].dropna(), kde=True, ax=ax)
                    plt.title(f"Distribution of {selected_column}")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    # Categorical column - bar chart
                    value_counts = st.session_state.df[selected_column].value_counts().reset_index()
                    value_counts.columns = [selected_column, 'Count']
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=selected_column, y='Count', data=value_counts, ax=ax)
                    plt.title(f"Distribution of {selected_column}")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error displaying column distribution: {str(e)}")
    else:
        st.info("No data available. Please upload a CSV file or initialize sample data.")

# Survey questionnaire tab
with tab3:
    if st.session_state.questionnaire_text is not None:
        st.markdown("### Survey Questionnaire")
        st.text_area("Questionnaire Content", st.session_state.questionnaire_text, height=600)
    else:
        st.info("No questionnaire available. Please place the questionnaire file in the project folder.")

# Footer
st.markdown("""
<div class="footer">
    Survey Data Chatbot - International Media Group<br>
    Powered by LangChain and Hugging Face - © 2025
</div>
""", unsafe_allow_html=True)