import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader, DataFrameLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from huggingface_hub import login

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
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<div class="main-header">Survey Data Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query the Omnibus Survey using natural language</div>', unsafe_allow_html=True)

# Create tabs for chatbot and data view
tab1, tab2 = st.tabs(["Chatbot", "Data Preview"])

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

# Function to load sample data if no CSV is uploaded
def load_sample_data():
    # Creating sample data based on the questionnaire
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
def generate_survey_description(df):
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
    
    return description

# Function to setup the language model and vector store
def setup_llm_and_vectorstore(df, survey_desc):
    """Set up the language model and vector store for the chatbot"""
    with st.spinner("Setting up the language model and vector store... This might take a few minutes."):
        try:
            # Create text data from the dataframe summary and description
            df_summary = df.describe(include='all').to_string()
            column_info = pd.DataFrame({
                'Column': df.columns,
                'DataType': df.dtypes,
                'Unique_Values': [df[col].nunique() for col in df.columns],
                'Sample_Values': [', '.join(map(str, df[col].dropna().unique()[:5])) for col in df.columns]
            }).to_string()
            
            # Combine information for the context
            combined_text = survey_desc + "\n\n" + df_summary + "\n\n" + column_info
            
            # Create document from text
            documents = [combined_text]
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.create_documents(documents)
            
            # Add dataframe specific information
            # Create a list of sample records as additional context
            sample_records = []
            for i in range(min(10, len(df))):
                record = df.iloc[i].to_dict()
                sample_records.append(f"Record {i+1}: {record}")
            
            sample_texts = text_splitter.create_documents(["\n".join(sample_records)])
            texts.extend(sample_texts)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create vector store
            vector_store = FAISS.from_documents(texts, embeddings)
            
            # Set up the local language model
            # Option 1: Use a smaller model that can run on CPU
            model_id = "google/flan-t5-base"  # A smaller model, ~1GB
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            # Create LLM
            llm = HuggingFacePipeline(pipeline=pipe)
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create the conversational chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                memory=memory,
                verbose=True
            )
            
            # Add custom instructions for data analysis to the chain
            # This is done by modifying the combine_docs_chain's prompt
            # Since we're using a base T5 model, we need to provide clear instructions
            custom_prompt = """
            You are an AI assistant specialized in analyzing survey data. 
            The user will ask questions about the International Media Group Omnibus Survey from January 2025.
            
            Use the following pieces of context to answer the user's question:
            {context}
            
            When analyzing the data:
            1. Always provide percentage-based insights
            2. Compare across demographics when relevant
            3. Identify key trends and patterns
            4. Be precise with numbers and statistics
            5. Provide direct answers to questions
            
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            User Question: {question}
            """
            
            # Set the custom prompt
            chain.combine_docs_chain.llm_chain.prompt.template = custom_prompt
            
            # Return the vector store and chain
            return vector_store, chain
            
        except Exception as e:
            st.error(f"Error setting up the language model: {str(e)}")
            return None, None

# Sidebar content
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.markdown("### Upload Survey Data")
    uploaded_file = st.file_uploader("Upload Omnibus Survey CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the uploaded CSV file
        file_content = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(file_content))
        st.session_state.df = df
        
        # Generate survey description
        survey_desc = generate_survey_description(df)
        st.session_state.survey_desc = survey_desc
        
        # Reset the vector store and chain
        st.session_state.vector_store = None
        st.session_state.chain = None
        
        st.success("CSV file uploaded successfully!")
    else:
        # Use sample data if no file is uploaded
        if st.session_state.df is None:
            st.info("Using sample data. Upload a CSV file to use your own data.")
            df = load_sample_data()
            st.session_state.df = df
            
            # Generate survey description
            survey_desc = generate_survey_description(df)
            st.session_state.survey_desc = survey_desc
    
    # Select the model
    st.markdown("### Model Settings")
    model_option = st.selectbox(
        "Select LLM Model",
        ["google/flan-t5-base", "google/flan-t5-small", "stabilityai/stablelm-tuned-alpha-3b"],
        index=0,
        help="Smaller models run faster but may be less accurate."
    )
    
    # Initialize/reset the chatbot
    if st.button("Initialize Chatbot"):
        # Setup the language model and vector store
        vector_store, chain = setup_llm_and_vectorstore(st.session_state.df, st.session_state.survey_desc)
        
        if vector_store is not None and chain is not None:
            st.session_state.vector_store = vector_store
            st.session_state.chain = chain
            st.success("Chatbot initialized successfully!")
        else:
            st.error("Failed to initialize chatbot. Please try again.")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if 'memory' in st.session_state and st.session_state.chain is not None:
            st.session_state.chain.memory.clear()
        st.success("Chat history cleared!")
    
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
    if st.session_state.chain is None:
        st.info("Please initialize the chatbot first by clicking the 'Initialize Chatbot' button in the sidebar.")
    else:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f'<div class="chat-message user"><div class="avatar">👤</div><div class="message">{message}</div></div>', unsafe_allow_html=True)
            else:  # Bot message
                st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{message}</div></div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ask a question about the survey data:", key="user_input")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append(user_input)
            
            # Get response from chain
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain({"question": user_input})
                    bot_response = response['answer']
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append(bot_response)
                    
                    # Rerun to update the UI with the new messages
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
                    # Remove the user message if we couldn't get a response
                    st.session_state.chat_history.pop()

# Data preview tab
with tab2:
    if st.session_state.df is not None:
        st.markdown("### Survey Data Preview")
        st.dataframe(st.session_state.df.head(10))
        
        st.markdown("### Data Statistics")
        st.markdown("#### Basic Information")
        st.write(f"Number of respondents: {len(st.session_state.df)}")
        st.write(f"Number of questions/variables: {len(st.session_state.df.columns)}")
        
        st.markdown("#### Summary Statistics")
        # Only include numeric columns for describe
        numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(st.session_state.df[numeric_cols].describe())
        
        # Display column information
        st.markdown("#### Column Information")
        column_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes,
            'Unique Values': [st.session_state.df[col].nunique() for col in st.session_state.df.columns],
            '% Missing': [(st.session_state.df[col].isna().sum() / len(st.session_state.df)) * 100 for col in st.session_state.df.columns]
        })
        st.dataframe(column_info)
        
        # Allow user to select a column to view distribution
        st.markdown("#### Column Distribution")
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
    else:
        st.info("No data available. Please upload a CSV file or initialize sample data.")

# Footer
st.markdown("""
<div class="footer">
    Survey Data Chatbot - International Media Group<br>
    Powered by LangChain and Hugging Face - © 2025
</div>
""", unsafe_allow_html=True)
