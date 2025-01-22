import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import torch
from transformers import T5Tokenizer
import numpy as np
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = TestCaseGenerator(model_name='t5-base', device=device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def validate_training_data(data: List[Dict]) -> tuple[bool, str]:
    """Validate the structure of training data"""
    required_fields = ['feature_description', 'feature_type', 'test_type']
    
    if not isinstance(data, list):
        return False, "Data must be a list of test cases"
    
    if len(data) == 0:
        return False, "Data list cannot be empty"
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item {idx} must be a dictionary"
        
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            return False, f"Item {idx} is missing required fields: {', '.join(missing_fields)}"
        
        if not isinstance(item.get('feature_description'), str):
            return False, f"Item {idx}: feature_description must be a string"
        if not isinstance(item.get('feature_type'), str):
            return False, f"Item {idx}: feature_type must be a string"
        if not isinstance(item.get('test_type'), str):
            return False, f"Item {idx}: test_type must be a string"
    
    return True, "Data structure is valid"

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file and return the file path"""
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def load_training_data(file_path: str) -> Optional[List[Dict]]:
    """Load training data from file"""
    try:
        file_path = Path(file_path)
        if file_path.suffix == '.json':
            with open(file_path) as f:
                return json.load(f)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            # Convert list-like strings to actual lists
            for col in ['steps', 'expected_results']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
            return df.to_dict('records')
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def split_dataset(data: List[Dict], val_split: float = 0.2) -> tuple[List[Dict], List[Dict]]:
    """Split data into training and validation sets"""
    np.random.shuffle(data)
    split_idx = int(len(data) * (1 - val_split))
    return data[:split_idx], data[split_idx:]

def load_sample_data():
    """Load sample test cases for demonstration"""
    return [
        {
            "id": "TC001",
            "title": "User Login Validation",
            "description": "Verify user login with valid credentials",
            "steps": [
                "Enter valid username",
                "Enter valid password",
                "Click login button"
            ],
            "expected_result": "User should be logged in successfully",
            "domain": "Authentication"
        },
        {
            "id": "TC002",
            "title": "Password Reset",
            "description": "Verify password reset functionality",
            "steps": [
                "Click forgot password",
                "Enter registered email",
                "Submit request"
            ],
            "expected_result": "Password reset email should be sent",
            "domain": "Authentication"
        }
    ]

def init_jira_client():
    """Initialize Jira client with credentials from environment variables."""
    if not JIRA_AVAILABLE:
        return None
    
    try:
        jira = JIRA(
            server=os.getenv('JIRA_SERVER'),
            basic_auth=(os.getenv('JIRA_EMAIL'), os.getenv('JIRA_API_TOKEN'))
        )
        return jira
    except Exception as e:
        st.error(f"Failed to connect to Jira: {str(e)}")
        return None

def fetch_jira_data(jira, jql_query):
    """Fetch Jira tickets based on JQL query."""
    try:
        issues = jira.search_issues(jql_query, maxResults=1000)
        data = []
        for issue in issues:
            data.append({
                'key': issue.key,
                'summary': issue.fields.summary,
                'description': issue.fields.description or '',
                'status': issue.fields.status.name,
                'type': issue.fields.issuetype.name,
                'created': issue.fields.created,
                'updated': issue.fields.updated
            })
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching Jira data: {str(e)}")
        return None

def main():
    st.title("🌳 GreenOak Personal QA")
    
    # Load model
    model = load_model_and_tokenizer()
    if model is None:
        st.error("Failed to load model. Please check the logs for details.")
        return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Test Case Generator", "Model Training", "Jira Integration (Optional)"])
    
    with tab1:
        st.header("Generate Test Cases")
        
        # Domain selection
        domain = st.selectbox(
            "Select Domain",
            ["Authentication", "Payment Processing", "User Management", "Data Validation"]
        )
        
        # Test case parameters
        col1, col2 = st.columns(2)
        with col1:
            complexity = st.slider("Test Case Complexity", 1, 5, 3)
            num_cases = st.number_input("Number of Test Cases", 1, 10, 3)
            test_type = st.selectbox(
                "Test Type",
                ["Functional", "Integration", "Security", "Performance", "Usability"]
            )
        
        with col2:
            include_edge_cases = st.checkbox("Include Edge Cases", True)
            include_negative_cases = st.checkbox("Include Negative Cases", True)
        
        if st.button("Generate Test Cases"):
            with st.spinner("Generating test cases..."):
                try:
                    # Generate test cases
                    test_cases = model.generate(
                        feature_description=f"{domain} system with {'edge cases and ' if include_edge_cases else ''}{'negative scenarios' if include_negative_cases else 'positive scenarios'}",
                        feature_type=domain,
                        test_type=test_type,
                        domain=domain,
                        num_return_sequences=3,
                        max_length=512
                    )
                    
                    if test_cases and len(test_cases) > 0:
                        st.success("✅ Test cases generated successfully!")
                        
                        # Display test cases in tabs
                        tabs = st.tabs([f"Test Case {i+1}" for i in range(len(test_cases))])
                        for i, (tab, test_case) in enumerate(zip(tabs, test_cases)):
                            with tab:
                                # Scenario
                                st.markdown("### 📋 Scenario")
                                st.info(test_case.get('scenario', 'No scenario provided'))
                                
                                # Steps
                                st.markdown("### 🔄 Test Steps")
                                steps = test_case.get('steps', [])
                                if steps:
                                    for j, step in enumerate(steps, 1):
                                        st.write(f"{j}. {step}")
                                else:
                                    st.warning("No steps provided")
                                
                                # Expected Results
                                st.markdown("### ✅ Expected Results")
                                results = test_case.get('expected_results', [])
                                if results:
                                    for result in results:
                                        st.write(f"- {result}")
                                else:
                                    st.warning("No expected results provided")
                                
                                # Test Data
                                if test_case.get('test_data'):
                                    st.markdown("### 📊 Test Data")
                                    st.json(test_case['test_data'])
                    else:
                        st.error("❌ No valid test cases were generated. Please try again with different inputs.")
                except Exception as e:
                    st.error(f"Error generating test cases: {str(e)}")
                    st.error("Please try again with different inputs or contact support if the issue persists.")
        
        # Analytics section
        st.header("Analytics")
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Test Cases", "150")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_metrics2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Success Rate", "95%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_metrics3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Complexity", "3.2")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Train Model")
        
        # Training data upload
        st.subheader("1. Upload Training Data")
        
        # Show sample data format
        with st.expander("View sample data format"):
            sample_data = {
                "feature_description": "User authentication system",
                "feature_type": "Authentication",
                "test_type": "Functional",
                "domain": "Security",
                "steps": ["Enter username", "Enter password", "Click login"],
                "expected_results": ["User should be logged in", "Redirect to dashboard"]
            }
            st.json(sample_data)
            st.info("Your training data should follow this format. You can provide it in either JSON or CSV format.")
        
        uploaded_file = st.file_uploader(
            "Upload your training data file",
            type=["json", "csv"],
            help="Upload a JSON or CSV file containing your training data"
        )
        
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                data = load_training_data(file_path)
                if data:
                    is_valid, message = validate_training_data(data)
                    if is_valid:
                        st.success(f"✅ Successfully loaded {len(data)} training examples")
                        
                        # Training parameters
                        st.subheader("2. Configure Training Parameters")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            epochs = st.number_input("Number of Epochs", 1, 50, 3)
                            batch_size = st.number_input("Batch Size", 1, 64, 8)
                            
                        with col2:
                            learning_rate = st.number_input(
                                "Learning Rate",
                                min_value=1e-6,
                                max_value=1e-2,
                                value=2e-5,
                                format="%e"
                            )
                            validation_split = st.slider(
                                "Validation Split",
                                min_value=0.1,
                                max_value=0.3,
                                value=0.2,
                                help="Portion of data to use for validation"
                            )
                        
                        if st.button("Start Training"):
                            try:
                                with st.spinner("Preparing datasets..."):
                                    # Split data
                                    train_data, val_data = split_dataset(data, validation_split)
                                    
                                    # Save splits temporarily
                                    temp_dir = Path("data/temp")
                                    temp_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    train_path = temp_dir / "train_temp.json"
                                    val_path = temp_dir / "val_temp.json"
                                    
                                    with open(train_path, 'w') as f:
                                        json.dump(train_data, f)
                                    with open(val_path, 'w') as f:
                                        json.dump(val_data, f)
                                    
                                    # Create datasets
                                    train_dataset = TestCaseDataset(str(train_path), model.tokenizer)
                                    val_dataset = TestCaseDataset(str(val_path), model.tokenizer)
                                    
                                    st.info(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")
                                    
                                    # Initialize trainer
                                    trainer = TestCaseTrainer(
                                        model=model,
                                        train_dataset=train_dataset,
                                        val_dataset=val_dataset,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size,
                                        num_epochs=epochs,
                                        output_dir="models"
                                    )
                                    
                                    # Training progress
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    metrics_container = st.empty()
                                    
                                    # Train the model
                                    for epoch in range(epochs):
                                        progress = (epoch + 1) / epochs
                                        progress_bar.progress(progress)
                                        status_text.text(f"Training Epoch {epoch+1}/{epochs}")
                                        
                                        metrics = trainer.train_epoch()
                                        metrics_container.write(f"""
                                        **Epoch {epoch+1} Metrics:**
                                        - Training Loss: {metrics['loss']:.4f}
                                        - Validation Loss: {metrics['val_loss']:.4f}
                                        """)
                                    
                                    st.success("✅ Training completed!")
                                    
                                    # Save the model
                                    save_path = Path("models") / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    model.save_pretrained(save_path)
                                    st.info(f"Model saved to: {save_path}")
                                    
                            except Exception as e:
                                st.error(f"Error during training: {str(e)}")
                                logger.exception("Training error")
                    else:
                        st.error(f"❌ Invalid data format: {message}")
    
    with tab3:
        st.header("Jira Integration (Optional)")
        
        # Optional Jira import
        try:
            from jira import JIRA
            JIRA_AVAILABLE = True
        except ImportError:
            JIRA_AVAILABLE = False
        
        if not JIRA_AVAILABLE:
            st.warning("Jira integration is not available. Install 'jira' package to enable this feature.")
            st.info("You can install it using: pip install jira")
        else:
            # Check for Jira credentials in .env
            jira_server = os.getenv("JIRA_SERVER")
            jira_email = os.getenv("JIRA_EMAIL")
            jira_token = os.getenv("JIRA_API_TOKEN")
            
            if not all([jira_server, jira_email, jira_token]):
                st.info("Jira credentials not found in .env file. Please configure them below:")
                
                with st.form("jira_config"):
                    jira_server = st.text_input("JIRA URL", placeholder="https://your-domain.atlassian.net")
                    jira_email = st.text_input("Email", placeholder="your-email@example.com")
                    jira_token = st.text_input("API Token", type="password", help="Generate an API token from your Atlassian account settings")
                    project_key = st.text_input("Project Key", placeholder="e.g., TEST, PROJ")
                    
                    if st.form_submit_button("Save Configuration"):
                        # Create or update .env file
                        env_path = Path(".env")
                        env_content = f"""JIRA_SERVER="{jira_server}"
JIRA_EMAIL="{jira_email}"
JIRA_API_TOKEN="{jira_token}"
JIRA_PROJECT_KEY="{project_key}"
"""
                        with open(env_path, "w") as f:
                            f.write(env_content)
                        st.success("✅ Configuration saved! Please restart the application.")
                
                with st.expander("How to get your API token"):
                    st.markdown("""
                    1. Log in to https://id.atlassian.com
                    2. Go to Security → API tokens
                    3. Click "Create API token"
                    4. Give it a name (e.g., "Test Case Generator")
                    5. Copy the token and paste it here
                    """)
            else:
                st.success("✅ Jira configuration found!")
                
                # Jira Query Section
                st.subheader("Import Test Cases from Jira")
                
                col1, col2 = st.columns(2)
                with col1:
                    jql = st.text_area(
                        "JQL Query",
                        value='project = "TEST" AND type = "Test Case"',
                        help="Enter your JQL query to filter issues"
                    )
                
                with col2:
                    max_results = st.number_input("Max Results", 1, 100, 10)
                
                if st.button("Fetch Test Cases"):
                    try:
                        jira = JIRA(
                            server=jira_server,
                            basic_auth=(jira_email, jira_token)
                        )
                        
                        with st.spinner("Fetching test cases from Jira..."):
                            issues = jira.search_issues(jql, maxResults=max_results)
                            
                            if issues:
                                st.success(f"Found {len(issues)} test cases")
                                
                                # Display issues
                                for issue in issues:
                                    with st.expander(f"{issue.key}: {issue.fields.summary}"):
                                        st.write(f"**Description:** {issue.fields.description or 'No description'}")
                                        st.write(f"**Status:** {issue.fields.status.name}")
                                        st.write(f"**Created:** {issue.fields.created}")
                                        
                                        # Add button to use this test case for training
                                        if st.button(f"Use for Training", key=issue.key):
                                            # Convert Jira issue to training format
                                            training_data = {
                                                "feature_description": issue.fields.summary,
                                                "feature_type": str(issue.fields.issuetype),
                                                "test_type": "Functional",
                                                "steps": [step.strip() for step in (issue.fields.description or "").split("\n") if step.strip()],
                                                "expected_results": ["Test case imported from Jira"]
                                            }
                                            
                                            # Save to training data
                                            data_dir = Path("data")
                                            data_dir.mkdir(exist_ok=True)
                                            
                                            training_file = data_dir / "jira_training_data.json"
                                            
                                            # Load existing data or create new
                                            if training_file.exists():
                                                with open(training_file) as f:
                                                    existing_data = json.load(f)
                                            else:
                                                existing_data = []
                                            
                                            existing_data.append(training_data)
                                            
                                            with open(training_file, "w") as f:
                                                json.dump(existing_data, f, indent=2)
                                            
                                            st.success(f"Added {issue.key} to training data!")
                            else:
                                st.warning("No test cases found matching the query")
                    
                    except Exception as e:
                        st.error(f"Error connecting to Jira: {str(e)}")

if __name__ == "__main__":
    try:
        from .model import TestCaseGenerator
        from .dataset import TestCaseDataset
        from .trainer import TestCaseTrainer
    except ImportError:
        from model import TestCaseGenerator
        from dataset import TestCaseDataset
        from trainer import TestCaseTrainer
    main()
