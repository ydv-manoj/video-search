import os
import tempfile
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import subprocess
from groq import Groq
from sentence_transformers import SentenceTransformer
import uuid
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Search API", version="1.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize sentence transformer model (lightweight and fast)
logger.info("Loading sentence transformer model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Model loaded successfully!")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db_sentence_transformer")
collection = chroma_client.get_or_create_collection(
    name="video_transcripts",
    metadata={"hnsw:space": "cosine"}
)

# Store for tracking upload progress
upload_progress = {}

# Pydantic models
class SearchResponse(BaseModel):
    video_id: str
    video_name: str
    text: str
    start_time: float
    end_time: float
    confidence: float

class UploadResponse(BaseModel):
    video_id: str
    message: str
    segments_processed: int

class BatchUploadResponse(BaseModel):
    upload_id: str
    message: str
    total_videos: int
    processing: bool

class UploadStatus(BaseModel):
    upload_id: str
    total_videos: int
    completed_videos: int
    failed_videos: int
    current_video: Optional[str]
    status: str  # "processing", "completed", "failed"
    results: List[UploadResponse]

def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'libmp3lame', 
            '-ab', '192k', '-ar', '44100', 
            '-y', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return False

def transcribe_with_groq(audio_path: str) -> Optional[dict]:
    """Transcribe audio using Groq's Whisper API"""
    try:
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def get_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding using sentence-transformers model"""
    try:
        # sentence-transformers returns numpy array, convert to list
        embedding = embedding_model.encode(text).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def chunk_transcription(transcription_data: dict, chunk_duration: float = 10.0, overlap_duration: float = 2.0) -> List[dict]:
    """Split transcription into smaller, overlapping chunks for better search granularity"""
    chunks = []
    
    if hasattr(transcription_data, 'segments'):
        segments = transcription_data.segments
    else:
        # Fallback: create single chunk from full text
        return [{
            "text": transcription_data.text if hasattr(transcription_data, 'text') else str(transcription_data),
            "start": 0,
            "end": chunk_duration
        }]
    
    # Strategy 1: Sentence-based chunking (primary)
    sentence_chunks = create_sentence_based_chunks(segments, chunk_duration, overlap_duration)
    
    # Strategy 2: Sliding window chunking (backup for long sentences)
    if not sentence_chunks:
        sentence_chunks = create_sliding_window_chunks(segments, chunk_duration, overlap_duration)
    
    return sentence_chunks

def create_sentence_based_chunks(segments, max_duration: float = 10.0, overlap_duration: float = 2.0) -> List[dict]:
    """Create chunks based on sentence boundaries for better semantic coherence"""
    chunks = []
    current_chunk = {
        "text": "",
        "start": None,
        "end": None,
        "sentences": []
    }
    
    for segment in segments:
        segment_start = segment.get('start', 0)
        segment_end = segment.get('end', segment_start + 5)
        segment_text = segment.get('text', '').strip()
        
        if not segment_text:
            continue
            
        # Split segment into sentences
        sentences = split_into_sentences(segment_text)
        
        for sentence in sentences:
            sentence_duration = (segment_end - segment_start) / len(sentences)
            sentence_start = segment_start + (sentences.index(sentence) * sentence_duration)
            sentence_end = sentence_start + sentence_duration
            
            # Check if adding this sentence would exceed max duration
            if (current_chunk["start"] is not None and 
                sentence_end - current_chunk["start"] > max_duration and 
                current_chunk["text"]):
                
                # Finalize current chunk
                chunks.append({
                    "text": current_chunk["text"].strip(),
                    "start": current_chunk["start"],
                    "end": current_chunk["end"]
                })
                
                # Start new chunk with overlap
                overlap_start = max(0, current_chunk["end"] - overlap_duration)
                overlap_text = ""
                
                # Add overlapping sentences for context
                for prev_sentence in current_chunk["sentences"][-2:]:  # Last 2 sentences
                    if prev_sentence["start"] >= overlap_start:
                        overlap_text += prev_sentence["text"] + " "
                
                current_chunk = {
                    "text": overlap_text + sentence,
                    "start": overlap_start if overlap_text else sentence_start,
                    "end": sentence_end,
                    "sentences": [{"text": sentence, "start": sentence_start, "end": sentence_end}]
                }
            else:
                # Add to current chunk
                if current_chunk["start"] is None:
                    current_chunk["start"] = sentence_start
                current_chunk["text"] += " " + sentence if current_chunk["text"] else sentence
                current_chunk["end"] = sentence_end
                current_chunk["sentences"].append({
                    "text": sentence, 
                    "start": sentence_start, 
                    "end": sentence_end
                })
    
    # Add the last chunk
    if current_chunk["text"]:
        chunks.append({
            "text": current_chunk["text"].strip(),
            "start": current_chunk["start"],
            "end": current_chunk["end"]
        })
    
    return chunks

def create_sliding_window_chunks(segments, chunk_duration: float = 10.0, overlap_duration: float = 2.0) -> List[dict]:
    """Create sliding window chunks as fallback"""
    chunks = []
    all_words = []
    
    # Extract all words with timestamps
    for segment in segments:
        segment_start = segment.get('start', 0)
        segment_end = segment.get('end', segment_start + 5)
        segment_text = segment.get('text', '').strip()
        
        if segment_text:
            words = segment_text.split()
            word_duration = (segment_end - segment_start) / len(words)
            
            for i, word in enumerate(words):
                word_start = segment_start + (i * word_duration)
                word_end = word_start + word_duration
                all_words.append({
                    "text": word,
                    "start": word_start,
                    "end": word_end
                })
    
    if not all_words:
        return []
    
    # Create sliding window chunks
    start_idx = 0
    
    while start_idx < len(all_words):
        chunk_start_time = all_words[start_idx]["start"]
        chunk_text = ""
        chunk_end_time = chunk_start_time
        end_idx = start_idx
        
        # Collect words until we hit the time limit
        while (end_idx < len(all_words) and 
               all_words[end_idx]["start"] - chunk_start_time < chunk_duration):
            chunk_text += all_words[end_idx]["text"] + " "
            chunk_end_time = all_words[end_idx]["end"]
            end_idx += 1
        
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "start": chunk_start_time,
                "end": chunk_end_time
            })
        
        # Move start index forward, accounting for overlap
        next_start_time = chunk_start_time + chunk_duration - overlap_duration
        start_idx = end_idx
        
        # Find the word closest to next_start_time
        for i in range(start_idx, len(all_words)):
            if all_words[i]["start"] >= next_start_time:
                start_idx = i
                break
    
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics"""
    import re
    
    # Simple sentence splitting (can be enhanced with NLTK if needed)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Rejoin very short fragments with the previous sentence
    cleaned_sentences = []
    for sentence in sentences:
        if len(sentence) < 10 and cleaned_sentences:
            cleaned_sentences[-1] += ". " + sentence
        else:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def store_in_chromadb(video_id: str, video_name: str, chunks: List[dict]):
    """Store transcription chunks in ChromaDB with sentence-transformer embeddings"""
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        if chunk["text"].strip():  # Only store non-empty chunks
            chunk_id = f"{video_id}_{i}"
            text = chunk["text"].strip()
            
            # Generate embedding with sentence-transformers
            embedding = get_embedding(text)
            if embedding is None:
                logger.warning(f"Failed to generate embedding for chunk {chunk_id}")
                continue
                
            documents.append(text)
            metadatas.append({
                "video_id": video_id,
                "video_name": video_name,
                "start_time": chunk["start"],
                "end_time": chunk["end"],
                "chunk_index": i
            })
            ids.append(chunk_id)
            embeddings.append(embedding)
    
    if documents:
        # Store in ChromaDB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        
        logger.info(f"Stored {len(documents)} chunks for video {video_id}")
    
    return len(documents)

async def process_single_video(
    file_content: bytes, 
    filename: str, 
    chunk_size: float, 
    overlap: float
) -> UploadResponse:
    """Process a single video file"""
    video_id = str(uuid.uuid4())
    
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as video_temp:
        video_temp.write(file_content)
        video_path = video_temp.name
    
    # Extract audio
    audio_path = tempfile.mktemp(suffix='.mp3')
    if not extract_audio_from_video(video_path, audio_path):
        raise Exception("Failed to extract audio")
    
    # Transcribe with Groq
    transcription = transcribe_with_groq(audio_path)
    if not transcription:
        raise Exception("Failed to transcribe audio")
    
    # Chunk transcription with user-specified parameters
    chunks = chunk_transcription(transcription, chunk_duration=chunk_size, overlap_duration=overlap)
    
    # Store in ChromaDB
    segments_processed = store_in_chromadb(video_id, filename, chunks)
    
    # Cleanup temporary files
    os.unlink(video_path)
    os.unlink(audio_path)
    
    return UploadResponse(
        video_id=video_id,
        message=f"Video processed successfully with {chunk_size}s chunks",
        segments_processed=segments_processed
    )

async def process_multiple_videos_background(
    upload_id: str,
    files_data: List[tuple],  # [(file_content, filename), ...]
    chunk_size: float,
    overlap: float
):
    """Background task to process multiple videos"""
    total_videos = len(files_data)
    completed_videos = 0
    failed_videos = 0
    results = []
    
    upload_progress[upload_id] = UploadStatus(
        upload_id=upload_id,
        total_videos=total_videos,
        completed_videos=0,
        failed_videos=0,
        current_video=None,
        status="processing",
        results=[]
    )
    
    for file_content, filename in files_data:
        try:
            upload_progress[upload_id].current_video = filename
            
            result = await process_single_video(file_content, filename, chunk_size, overlap)
            results.append(result)
            completed_videos += 1
            
            upload_progress[upload_id].completed_videos = completed_videos
            upload_progress[upload_id].results = results
            
        except Exception as e:
            logger.error(f"Error processing video {filename}: {e}")
            failed_videos += 1
            upload_progress[upload_id].failed_videos = failed_videos
    
    # Update final status
    upload_progress[upload_id].status = "completed" if failed_videos == 0 else "partial"
    upload_progress[upload_id].current_video = None

@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    chunk_size: float = Query(8.0, description="Chunk duration in seconds (3-30)", ge=3.0, le=30.0),
    overlap: float = Query(1.5, description="Overlap duration in seconds (0-5)", ge=0.0, le=5.0)
):
    """Upload and process a single video file"""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    
    try:
        content = await file.read()
        return await process_single_video(content, file.filename, chunk_size, overlap)
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/upload-batch", response_model=BatchUploadResponse)
async def upload_multiple_videos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chunk_size: float = Query(8.0, description="Chunk duration in seconds (3-30)", ge=3.0, le=30.0),
    overlap: float = Query(1.5, description="Overlap duration in seconds (0-5)", ge=0.0, le=5.0)
):
    """Upload and process multiple video files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate all files
    for file in files:
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail=f"Unsupported video format: {file.filename}")
    
    # Generate upload ID
    upload_id = str(uuid.uuid4())
    
    # Read all files into memory
    files_data = []
    for file in files:
        content = await file.read()
        files_data.append((content, file.filename))
    
    # Start background processing
    background_tasks.add_task(
        process_multiple_videos_background,
        upload_id,
        files_data,
        chunk_size,
        overlap
    )
    
    return BatchUploadResponse(
        upload_id=upload_id,
        message=f"Processing {len(files)} videos in background",
        total_videos=len(files),
        processing=True
    )

@app.get("/upload-status/{upload_id}", response_model=UploadStatus)
async def get_upload_status(upload_id: str):
    """Get the status of a batch upload"""
    if upload_id not in upload_progress:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    return upload_progress[upload_id]

@app.get("/search", response_model=List[SearchResponse])
async def search_videos(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results")
):
    """Search for video segments matching the query"""
    try:
        # Generate query embedding with sentence-transformers
        query_embedding = get_embedding(query)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format response
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                confidence = 1.0 - results['distances'][0][i]  # Convert distance to confidence
                
                search_results.append(SearchResponse(
                    video_id=metadata['video_id'],
                    video_name=metadata['video_name'],
                    text=results['documents'][0][i],
                    start_time=metadata['start_time'],
                    end_time=metadata['end_time'],
                    confidence=max(0, confidence)  # Ensure non-negative
                ))
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error searching videos: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/videos")
async def list_videos():
    """List all processed videos"""
    try:
        # Get all unique videos from ChromaDB
        all_results = collection.get(include=['metadatas'])
        videos = {}
        
        for metadata in all_results['metadatas']:
            video_id = metadata['video_id']
            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id,
                    'video_name': metadata['video_name'],
                    'segments': 0
                }
            videos[video_id]['segments'] += 1
        
        return list(videos.values())
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its segments"""
    try:
        # Get all chunks for this video
        results = collection.get(
            where={"video_id": video_id},
            include=['ids']
        )
        
        if results['ids']:
            # Delete all chunks for this video
            collection.delete(ids=results['ids'])
            return {"message": f"Deleted video {video_id} and {len(results['ids'])} segments"}
        else:
            raise HTTPException(status_code=404, detail="Video not found")
            
    except Exception as e:
        logger.error(f"Error deleting video: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting video: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "collection_count": collection.count()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)