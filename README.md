# Langchain MultiPDF Chat App with Python

## A fully functional PDF chat app written in python which uses langchain and a large language model (LLM) to answer queries about your PDF documents.

This is an example project which was built by following a tutorial on YouTube showing how to create a PDF chatbot using Langchain, OpenAIEmbeddings and Streamlit.
However I made an addition to it and also created a FastAPI implementation so that it can be deployed to a server and used as an API.

The example code show how to do the following:

* Read a PDF, extract the text from the document and create chunks from the text.
* Use OpenAIEmbeddings to create embeddings from the text chunks.
* Create a vector database using FAISS using the embeddings.
* Use an LLM which will answer your queries about the PDF documents you have uploaded.
* It will also keep a history or context about the conversation that has been going on.

The FastAPI implementation is in the `app.py` file and it shows how the code can be structured into an api format so you can call it from a client and display it in a chat interface.
I have created a companion iOS app for this project which shows how to fetch the document from storage, upload it using the API, ask a question about the document and display the
response in a chat interface.

It can be found here: [PDF Chat iOS](https://github.com/Shayanzahid/PDF-Chat-iOS)

## Dependencies and installation

To install the Langchain MultiPDF Chat App with Python, please follow these steps:

* Clone the repository in your machine.
* Install the required dependencies by running the following command in terminal or in VSCode terminal:
<pre>pip install -r requirements.txt</pre>
* Obtain an API key from OpenAI and place it in the .env.example file and remove the `.example` from the filename.
<pre>OPENAI_API_KEY=your_secret_api_key</pre>

## Usage

If you just want to use the streamlit interface version of this app, please follow these steps:

* Please make sure that you have installed all the dependencies and added the OpenAI api key in the .env file.
* Open the project in VSCode or navigate to the project from the terminal and execute the following command:
<pre>streamlit run main.py</pre>
* The app will be launched in your default browser and will show the UI where you can upload multiple files and chat with the PDFs.
* Use the chat interface to ask questions about your files.

## Contributions

Since this is an example project created by following a YouTube tutorial, feel free to clone the project and make changes to it according to your own needs.

The original YouTube video can be found here: [MultiPDF Chat App](https://youtu.be/dXxQ0LR-3Hg)

## License

The Langchain MultiPDF Chat App with Python is released under the [MIT License](https://opensource.org/licenses/MIT)

