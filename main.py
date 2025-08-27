import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import getpass
import os
from PyPDF2 import PdfReader
import random
import hashlib

# Create a directory for ChromaDB persistence
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

st.title("Welcome to Meghantra:bow_and_arrow: !")

# Initialize session state for quiz
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
if "current_chunk_index" not in st.session_state:
    st.session_state.current_chunk_index = 0
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

def get_next_chunk():
    if not st.session_state.document_chunks:
        return None

    chunk = st.session_state.document_chunks[st.session_state.current_chunk_index]
    st.session_state.current_chunk_index = (st.session_state.current_chunk_index + 1) % len(st.session_state.document_chunks)
    return chunk

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your Document", type=["txt", "pdf"])
    if uploaded_file:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
        else:  # txt file
            text_content = uploaded_file.read().decode('utf-8')

        st.success("File Successfully Uploaded!")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)
        st.session_state.document_chunks = chunks
        st.session_state.current_chunk_index = 0

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Initialize ChromaDB with persistence
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        vectorstore.persist()
        st.session_state.vectorstore = vectorstore

        if st.button("Start Quiz", key="start_quiz"):
            st.session_state.quiz_active = True
            st.session_state.quiz_score = 0
            st.session_state.total_questions = 0
            st.session_state.asked_questions = []
            st.session_state.current_chunk_index = 0
            st.session_state.current_question = None
            st.session_state.button_clicked = False
            st.rerun()

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.9,  # High temperature for more variety
    streaming=True
)

quiz_template = """
You are a quiz master. Based on the provided context, generate one multiple-choice question.
The question should test understanding of key concepts from the document.

Previously asked questions:
{previous_questions}

IMPORTANT: Generate a completely different question from any previous ones.
Focus on testing comprehension of this specific context:

{context}

You MUST format your response EXACTLY as follows (including the labels):
QUESTION: [your question here]
A) [first option]
B) [second option]
C) [third option]
D) [fourth option]
CORRECT_ANSWER: [A/B/C/D]
EXPLANATION: [brief explanation why this is correct]

Remember: Your question MUST be different from all previously asked questions.
"""

qa_template = """
You are an helpful assistant who analyzes and answers questions on the provided context.
Try to answer from the context and resolve user query.
Use the following pieces of context to answer the question at the end.

Context: {context}

Chat History: {chat_history}
Human: {question}
Assistant: """

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=qa_template,
)

def generate_question(context, previous_questions):
    """Generate a new quiz question."""
    prompt = quiz_template.format(
        context=context,
        previous_questions=previous_questions
    )

    response = llm.invoke(prompt)
    return response.content

# Quiz Mode
if st.session_state.quiz_active and "vectorstore" in st.session_state:
    st.subheader("📝 Quiz Mode")
    st.write(f"Current Score: {st.session_state.quiz_score}/{st.session_state.total_questions}")

    # Display previously asked questions in an expander
    with st.expander("View Previous Questions"):
        if st.session_state.asked_questions:
            st.write("Previously asked questions:")
            for i, q in enumerate(st.session_state.asked_questions):
                st.write(f"{i+1}. {q['question']}")
        else:
            st.write("No previous questions.")

    if st.session_state.current_question is None and not st.session_state.button_clicked:
        try:
            # Get next chunk of text
            context = get_next_chunk()
            if not context:
                st.error("No more content available for questions.")
                st.session_state.quiz_active = False
                st.rerun()

            # Format previous questions
            previous_questions = "\n".join([
                f"{i+1}. {q['question']}"
                for i, q in enumerate(st.session_state.asked_questions)
            ]) if st.session_state.asked_questions else "No previous questions."

            # Generate new question
            with st.spinner("Generating question..."):
                response = generate_question(context, previous_questions)

                # Parse the response
                lines = response.strip().split('\n')
                question_data = {
                    'explanation': 'No explanation provided.'  # Default explanation
                }

                for line in lines:
                    line = line.strip()
                    if line.startswith('QUESTION: '):
                        question_data['question'] = line[len('QUESTION: '):].strip()
                    elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                        if 'options' not in question_data:
                            question_data['options'] = []
                        question_data['options'].append(line)
                    elif line.startswith('CORRECT_ANSWER: '):
                        question_data['correct'] = line[len('CORRECT_ANSWER: '):].strip()
                    elif line.startswith('EXPLANATION: '):
                        question_data['explanation'] = line[len('EXPLANATION: '):].strip()

                if 'question' not in question_data or 'options' not in question_data or 'correct' not in question_data:
                    raise ValueError("Invalid question format")

                if len(question_data['options']) != 4:
                    raise ValueError(f"Expected 4 options, got {len(question_data['options'])}")

                if any(q['question'].lower() == question_data['question'].lower()
                       for q in st.session_state.asked_questions):
                    raise ValueError("Generated question is too similar to a previous question")

                st.session_state.current_question = question_data
                st.session_state.asked_questions.append(question_data)

        except Exception as e:
            st.error(f"An error occurred while generating the question: {str(e)}")
            st.error("Trying to generate a new question...")
            st.session_state.current_question = None
            st.rerun()

    if st.session_state.current_question:
        st.write("### " + st.session_state.current_question['question'])
        user_answer = st.radio(
            "Select your answer:",
            st.session_state.current_question['options'],
            key=f"quiz_answer_{st.session_state.total_questions}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer", key=f"submit_{st.session_state.total_questions}"):
                st.session_state.button_clicked = True
                selected_letter = user_answer[0]  # Get A, B, C, or D
                if selected_letter == st.session_state.current_question['correct']:
                    st.success("Correct! 🎉")
                    st.session_state.quiz_score += 1
                else:
                    st.error(f"Incorrect! The correct answer was {st.session_state.current_question['correct']}")

                explanation = st.session_state.current_question.get('explanation', 'No explanation provided.')
                st.info(f"Explanation: {explanation}")
                st.session_state.total_questions += 1

        with col2:
            if st.button("Next Question", key=f"next_{st.session_state.total_questions}"):
                st.session_state.current_question = None
                st.session_state.button_clicked = False
                st.rerun()

        if st.button("End Quiz", key="end_quiz"):
            st.session_state.quiz_active = False
            st.session_state.current_question = None
            st.session_state.asked_questions = []
            st.session_state.button_clicked = False
            st.rerun()

# Regular Q&A Mode
elif not st.session_state.quiz_active:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message[0].lower()):
            st.write(message[1])

    user_query = st.chat_input("Ask a question about your document")

    if user_query:
        with st.chat_message("human"):
            st.write(user_query)

        if "vectorstore" in st.session_state:
            try:
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    memory=ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key='answer'
                    ),
                    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
                )

                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({
                        "question": user_query
                    })

                    with st.chat_message("assistant"):
                        st.write(response['answer'])

                    st.session_state.chat_history.append(["Human", user_query])
                    st.session_state.chat_history.append(["AI", response['answer']])

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please try refreshing the page or uploading the document again.")

        else:
            with st.chat_message("assistant"):
                st.write("Please upload a document first!")
