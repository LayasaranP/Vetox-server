from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import os
from typing import Optional
import tempfile
import httpx

voice_router = APIRouter(
    prefix="/voice",
    tags=["voice"]
)

# Groq API details for transcriptions
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_API_KEY = os.getenv("GROQ")


@voice_router.post("/transcribe")
async def transcribe_audio(
        file: UploadFile = File(...),
        language: Optional[str] = "en",
        response_format: str = "text"
):
    """
    Transcribe uploaded audio file using Groq Whisper (whisper-large-v3-turbo)

    - Supports common audio formats: mp3, wav, m4a, webm, ogg, etc.
    - Max file size: ~25MB (Groq limit)
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.webm', '.ogg', '.flac')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio format. Supported: mp3, wav, m4a, webm, ogg, flac"
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        async with httpx.AsyncClient() as client:
            with open(temp_file_path, "rb") as audio_file:
                files = {"file": (file.filename, audio_file)}
                data = {
                    "model": "whisper-large-v3-turbo",
                    "language": language,
                    "response_format": response_format
                }
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                
                response = await client.post(
                    GROQ_API_URL,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=60.0
                )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq API error: {response.text}"
            )

        result = response.json()
        
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass

        if response_format == "text":
            return JSONResponse(content={"transcription": result.get("text", "")})
        else:
            return JSONResponse(content={"result": result})

    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@voice_router.get("/health")
async def voice_health():
    return {"status": "voice service healthy", "model": "whisper-large-v3-turbo"}
