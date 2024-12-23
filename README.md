# 🚀 Dasher Support Automation with Large Language Models  

This repository contains a project that automates support for Dashers using **large language models (LLMs)**. The solution draws inspiration from DoorDash's innovative [Dasher support automation](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/) system and replicates many of its key ideas to demonstrate the power of RAG (Retrieval-Augmented Generation) systems for customer support.

---

## 🌟 **Features**  
1. **Web Scraping with Selenium**: Extracted articles from the [Dasher Help Center](https://help.doordash.com/dashers/s/?language=en_US) and saved them in CSV format.  
2. **RAG System**:  
   - Summarized and embedded articles for vector database indexing.  
   - Retrieved and re-ranked relevant articles to answer user queries.  
   - Integrated retrieved articles with LLMs to generate responses.  
3. **Response Guardrails**:  
   - **Layer 1**: Checked semantic similarity between generated response and retrieved content.  
   - **Layer 2**: Evaluated responses for groundedness, coherence, compliance, and query relevance using a larger LLM.  
4. **Streamlit Web App**: User-friendly interface for interacting with the project.  
5. **Command-Line Execution**: A single terminal command runs the entire pipeline.  

---

## 📂 **Project Structure**  
```plaintext
├── DoorDashCustomerCareRAG_WebScrape.ipynb  # Notebook for web scraping help articles
├── app.py                                   # Streamlit web app for user interaction
├── helper.py                                # Helper functions for embedding and evaluation
├── main.py                                  # Main script to execute the project pipeline
├── requirements.txt                         # List of dependencies
├── README.md                                # Project documentation
```

---

## 🔧 **Setup and Installation**  
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/dasher-support-automation.git
   cd dasher-support-automation
   ```

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:  
   Add your API keys for Tohether, Groq, and Cohere in an `.env` file:  
   ```env
   GROQ_API_KEY=your_groq_key
   TOGETHER_API_KEY=your_together_key
   COHERE_API_KEY=your_cohere_key
   ```

4. **Run the Web Scraping Notebook**:  
   Extract articles from the Dasher Help Center using the notebook `DoorDashCustomerCareRAG_WebScrape.ipynb`.  

---

## 🛠️ **How to Run**  
### Streamlit Web App  
Run the Streamlit app to interact with the solution through a web interface:  
```bash
streamlit run app.py
```

### Command-Line Execution  
Run the entire project pipeline directly from the terminal:  
```bash
python main.py
```

---

## 🚀 **How It Works**  

### Web Scraping  
- Used Selenium to scrape articles from the Dasher Help Center.  
- Addressed challenges like dynamic loading by handling the "Load More" button and waiting for content to load.

### RAG System  
- Summarized articles using an LLM and embedded summaries for vector search using Pinecone.  
- Retrieved top articles based on query relevance.  
- Re-ranked retrieved articles using Cohere to ensure the top 3 results were the most accurate.  

### Response Generation  
- Constructed a retrieval-augmented generation (RAG) pipeline.  
- Combined retrieved articles and user queries to generate precise and contextually relevant responses.

### Response Guardrails  
1. **Layer 1: Semantic Similarity Check**  
   Ensured a semantic similarity score of >70% between the retrieved articles and the generated response using cosine similarity.  

2. **Layer 2: LLM-Based Evaluation**  
   Used a larger LLM to verify that responses:  
   - Are **grounded** in retrieved content.  
   - Are **coherent** and logically structured.  
   - Are **compliant** with guidelines.  
   - **Answer the user’s query** effectively.  

   Failed responses escalated to a human agent.  

---

## 🌐 **Read More About It**  
Want to dive deeper into how this project was built? Check out my detailed Medium article:  
[**Building a Dasher Support Automation System: My Journey Replicating DoorDash’s Approach**](https://medium.com/@gargkeshav204/building-a-dasher-support-automation-system-my-journey-replicating-doordashs-approach-8837ead9bbd1)  

---


## 🤝 **Acknowledgments**  
This project is inspired by DoorDash's original Dasher support automation system. While this implementation is an independent learning exercise, full credit goes to DoorDash for their innovative approach, as outlined in their [official blog](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/).  

---

## 🤝 **Contributions**  
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.  


---


If you find this project helpful, please ⭐ the repository!  
