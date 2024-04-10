import os
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import openai
import glob
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.schema.runnable.base import RunnableLambda

# 상수 정의
CHUNK_SIZE = 10  # 청크 크기 (분)
WHISPER_MODEL_NAME = "whisper-1"  # Whisper 모델 이름
CACHE_DIR = os.path.join("./.cache")  # 캐시 디렉토리 경로

# OpenAI의 ChatGPT 모델을 사용합니다.
# temperature는 출력 다양성을 제어하고, streaming=True로 설정하면 결과를 실시간으로 출력합니다.
llm = ChatOpenAI(temperature=0.1, streaming=True)

# Streamlit 앱 설정
st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

# 캐시된 텍스트 전사 파일이 있는지 확인
transcript_exists = os.path.exists(os.path.join(CACHE_DIR, "podcast.txt"))

# 텍스트를 청크(chunk)로 분할할 때 사용할 설정
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


# 검색 결과 문서를 문자열로 포맷팅하는 함수
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 파일을 임베딩하고 벡터 저장소를 생성하는 함수
@st.cache_data()
def embed_file(file_path):
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


# 오디오 청크를 텍스트로 전사하는 함수
@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if transcript_exists:
        return

    files = glob.glob(os.path.join(chunk_folder, "*.mp3"))
    files.sort()

    for file in files:
        try:
            with open(file, "rb") as audio_file, open(destination, "a") as text_file:
                transcript = openai.Audio.transcribe(
                    WHISPER_MODEL_NAME,
                    audio_file,
                )
                text_file.write(transcript["text"])
        except Exception as e:
            st.error(f"Error transcribing {file}: {str(e)}")


# 비디오에서 오디오를 추출하는 함수
@st.cache_data()
def extract_audio_from_video(video_path):
    if transcript_exists:
        return

    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    command = [
        "ffmpeg",
        "-y",  # 이전 파일을 덮어쓰기
        "-i",
        video_path,
        "-vn",  # 비디오 스트림 제외 (오디오만 추출)
        audio_path,
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting audio from video: {str(e)}")


# 오디오를 청크로 나누는 함수
@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if transcript_exists:
        return

    os.makedirs(chunks_folder, exist_ok=True)  # 디렉토리 생성

    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000  # 청크 크기를 밀리초로 변환
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]

        chunk_path = os.path.join(chunks_folder, f"chunk_{i}.mp3")
        try:
            chunk.export(chunk_path, format="mp3")
        except Exception as e:
            st.error(f"Error exporting chunk {i}: {str(e)}")


# 앱 소개 문구
st.markdown(
    """
# MeetingGPT

Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
Get started by uploading a video file in the sidebar.
"""
)

# 사이드바에서 비디오 파일 업로드
with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

# 비디오 파일이 업로드되었을 경우
if video:
    video_path = os.path.join(CACHE_DIR, video.name)
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    chunks_folder = os.path.join(CACHE_DIR, "chunks")
    transcript_path = os.path.splitext(video_path)[0] + ".txt"

    with st.status("Loading video...") as status:
        try:
            with open(video_path, "wb") as file:
                file.write(video.read())
            status.update(label="Extracting audio...")
            extract_audio_from_video(video_path)
            status.update(label="Cutting audio segments...")
            cut_audio_in_chunks(audio_path, CHUNK_SIZE, chunks_folder)
            status.update(label="Transcribing audio...")
            transcribe_chunks(chunks_folder, transcript_path)
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

    # 탭 생성
    transcript_tab, summary_tab, qa_tab = st.tabs(
        ["Transcript", "Summary", "Q&A"],
    )

    # 텍스트 전사 탭
    with transcript_tab:
        try:
            with open(transcript_path, "r") as file:
                st.write(file.read())
        except Exception as e:
            st.error(f"Error reading transcript: {str(e)}")

    # 요약 탭
    with summary_tab:
        start = st.button("Generate Summary")

        if start:
            try:
                loader = TextLoader(transcript_path)
                docs = loader.load_and_split(text_splitter=splitter)
                first_summary_prompt = ChatPromptTemplate.from_template(
                    """
                    Write a concise summary of the following:
                    "{text}"
                    CONCISE SUMMARY:
                """
                )

                first_summary_chain = first_summary_prompt | llm | StrOutputParser()

                summary = first_summary_chain.invoke(
                    {
                        "text": docs[0].page_content,
                    }
                )

                refine_prompt = ChatPromptTemplate.from_template(
                    """
                    Your job is to produce a final summary.
                    We have provided an existing summary up to a certain point: {existing_summary}
                    We have the opportunity to refine the existing summary (only if needed) with some more context below.
                    ------------
                    {context}
                    ------------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                    """
                )

                refine_chain = refine_prompt | llm | StrOutputParser()

                with st.status("Summarizing...") as status:
                    for i, doc in enumerate(docs[1:]):
                        status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                        summary = refine_chain.invoke(
                            {
                                "existing_summary": summary,
                                "context": doc.page_content,
                            }
                        )
                        st.write(summary)
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

    # Q&A 탭
    with qa_tab:
        try:
            # 파일 임베딩 및 벡터 저장소 생성
            retriever = embed_file(transcript_path)

            # Q&A 프롬프트 템플릿 설정
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        Answer the question using ONLY the following context.
                        If you don't know the answer just say you don't know.
                        DON'T make anything up.

                        Context: {context}
                        """,
                    ),
                    ("human", "{question}"),
                ]
            )

            # 텍스트 입력창에서 질문 받기
            message = st.text_input("Ask anything about your video...")

            # 질문이 입력되었을 경우
            if message:
                # 검색 결과와 질문을 프롬프트에 전달하는 체인 생성
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | qa_prompt
                    | llm
                )

                # 챗봇 출력 창에 결과 표시
                with st.chat_message("ai"):
                    chain.invoke(message).content
        except Exception as e:
            st.error(f"Error in Q&A: {str(e)}")
