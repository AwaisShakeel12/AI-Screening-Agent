import os
import json
import logging
from typing import Optional, Any
from supabase import create_client, Client
import sentry_sdk

# Setup logging so Vercel and your terminal can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

url: str = os.getenv("SUPABASE_URL")
# IMPORTANT: Ensure this matches the exact name in your Vercel Environment Variables!
key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    logger.error("⚠️ Supabase credentials missing in environment!")
    supabase = None
else:
    supabase: Client = create_client(url, key)

def load_session(session_id: str, state_class: Any) -> Optional[Any]:
    if not supabase: return None
    try:
        response = supabase.table("screening_sessions").select("*").eq("session_id", session_id).execute()
        if response.data and len(response.data) > 0:
            row = response.data[0]
            
            messages_data = row.get("messages") or []
            if isinstance(messages_data, str):
                try:
                    messages_data = json.loads(messages_data)
                except Exception:
                    messages_data = []
                    
            state_dict = {
                "session_id": row["session_id"],
                "intent_to_apply": row["intent_to_apply"],
                "name": row["name"],
                "email": row["email"],
                "phone": row["phone"],
                "experience": row["experience"],
                "basic_complete": row["basic_complete"],
                "resume_path": row["resume_storage_path"],
                "resume_filename": row["resume_filename"],
                "resume_text": row["resume_text"],
                "resume_uploaded": row["resume_uploaded"],
                "interview_questions_asked": row["interview_questions_asked"],
                "max_interview_questions": row["max_interview_questions"],
                "interview_complete": row["interview_complete"],
                "hr_questions_asked": row["hr_questions_asked"],
                "max_hr_questions": row["max_hr_questions"],
                "hr_complete": row["hr_complete"],
                "report_path": row["report_storage_path"],
                "report_generated": row["report_generated"],
                "email_sent": row["email_sent"],
                "messages": messages_data,
                "user_input": "",
                "user_file": None,
                "user_filename": None
            }
            return state_class(**state_dict)
    except Exception as e:
        logger.error(f"❌ Error loading session: {e}")
        sentry_sdk.capture_exception(e)
    return None

def save_session(session_id: str, state: Any):
    if not supabase: return
    try:
        data = {
            "session_id": session_id,
            "intent_to_apply": state.intent_to_apply,
            "name": state.name,
            "email": state.email,
            "phone": state.phone,
            "experience": state.experience,
            "basic_complete": state.basic_complete,
            "resume_storage_path": state.resume_path,
            "resume_filename": state.resume_filename,
            "resume_text": state.resume_text,
            "resume_uploaded": state.resume_uploaded,
            "interview_questions_asked": state.interview_questions_asked,
            "max_interview_questions": state.max_interview_questions,
            "interview_complete": state.interview_complete,
            "hr_questions_asked": state.hr_questions_asked,
            "max_hr_questions": state.max_hr_questions,
            "hr_complete": state.hr_complete,
            "report_storage_path": state.report_path,
            "report_generated": state.report_generated,
            "email_sent": state.email_sent,
            "messages": state.messages,
            "updated_at": "now()"
        }
        supabase.table("screening_sessions").upsert(data, on_conflict="session_id").execute()
    except Exception as e:
        logger.error(f"❌ Error saving session: {e}")
        sentry_sdk.capture_exception(e)

def upload_file_to_storage(bucket: str, path: str, file_bytes: bytes, content_type: str = "application/octet-stream"):
    if not supabase: return path
    try:
        logger.info(f"☁️ Uploading to bucket '{bucket}' at path '{path}'")
        supabase.storage.from_(bucket).upload(
            path=path,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        logger.info("✅ Upload successful!")
        return path
    except Exception as e:
        logger.error(f"❌ Storage upload error: {str(e)}")
        sentry_sdk.capture_exception(e)
        raise e

def download_file_from_storage(bucket: str, path: str) -> bytes:
    if not supabase: return b""
    try:
        res = supabase.storage.from_(bucket).download(path)
        return res
    except Exception as e:
        logger.error(f"❌ Storage download error: {str(e)}")
        sentry_sdk.capture_exception(e)
        return b""