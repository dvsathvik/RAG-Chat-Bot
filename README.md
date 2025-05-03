# RAG Chat-Bot

![simple_rag_workflow_091648ef39](https://github.com/user-attachments/assets/8662b4d5-161b-4d43-a876-d84b57bc8192)

RAG Based chatbot are very efficient in giving the reply if they have a large input data.

Retrieval-Augmented Generation is a framework that:
1. Retrieves relevant information (documents, text chunks, FAQs, etc.) from a knowledge base using vector similarity search.
2. Augments a generative model (like ChatGPT or LLaMA) with that retrieved information.
3. Generates responses based on both the user’s query and the retrieved knowledge.


If we use the old approach to answering the query by checking whole data,can take a lot of time i fthe input data is big. 
In RAG chatbot , we use Vector Similarlity search,, where we find all the relavent part of the user query by using vector Calculations. And we use this filtered data to answer the question.
