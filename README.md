# Time-Series-Forecasting-Visualization

## Project Overview
This project performs sentiment analysis using Google Gemini Pro and OpenAI's ChatGPT APIs in Python. It classifies Amazon Alexa customer reviews as positive or negative using LLMs, leveraging zero-shot and few-shot prompting techniques.

## Dataset
The dataset used is the **Amazon Alexa Customer Reviews** dataset, which contains:
- **Verified Reviews:** Customer review text.
- **Feedback:** Binary classification labels (1 = Positive, 0 = Negative).

## Workflow
### 1. Data Preparation
- Load the dataset (`Amazon_Alexa.tsv`).
- Extract relevant columns (`verified_reviews`, `feedback`).
- Perform data cleaning: remove special characters, HTML tags, and extra spaces.
- Balance the dataset using down-sampling.

### 2. Data Splitting
- Split the dataset into:
  - **5% training set** (for few-shot prompting).
  - **95% test set** (for sentiment prediction using APIs).

### 3. Sentiment Prediction using LLMs
#### **Google Gemini Pro API**
- Set up Google API key.
- Convert a sample of the test set to JSON format.
- Define a prompt specifying sentiment classification rules.
- Send an API request to Gemini Pro.
- Extract predictions and store results in a DataFrame.
- Compute accuracy using a confusion matrix.

#### **OpenAI ChatGPT API (Batch Processing)**
- Configure OpenAI API key.
- Implement batch processing for handling API token limits.
- Send multiple API calls for prediction.
- Extract results and compute accuracy.

### 4. Few-shot Prompting with ChatGPT
- Use a small training sample (5% of dataset) to provide example-based prompting.
- Compare accuracy with zero-shot predictions.

## Results
- **Google Gemini Pro:** Achieved **100% accuracy** on a small test sample.
- **ChatGPT (Zero-shot):** Achieved **91% accuracy** on batch-processed samples.
- **ChatGPT (Few-shot):** Accuracy remained similar, validating the efficiency of LLMs in sentiment classification.

## Installation & Setup
```bash
pip install openai google-generativeai pandas numpy
```

### API Key Configuration
Store API keys in environment variables or within the script:
```python
import os
os.environ['GOOGLE_API_KEY'] = 'your_google_api_key'
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
```

## Execution
Run the sentiment analysis script in a Jupyter Notebook or Google Colab:
```python
!python sentiment_analysis.py
```

## Conclusion
This project demonstrates the power of LLMs for sentiment analysis without training traditional ML models. It efficiently classifies reviews using Google Gemini Pro and OpenAI’s ChatGPT with minimal computational effort.

## Future Improvements
- Test with larger datasets for robust evaluation.
- Implement prompt engineering techniques for further accuracy improvements.
- Integrate a Streamlit app for real-time sentiment analysis.

## References
- [Google Gemini API Docs](https://ai.google.dev/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
