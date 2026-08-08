import os
import uuid
import traceback
from io import BytesIO
from typing import Optional, Dict, List
import base64
import re

import resend
import sentry_sdk
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# Import our new Supabase service
from supabase_service import load_session, save_session, upload_file_to_storage, download_file_from_storage

# ---------- Environment ----------
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    send_default_pii=False,
    enable_logs=True,
    environment="production",
)

# ---------- API Keys & LLM Setup (Groq) ----------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ No GROQ_API_KEY found in .env file!")

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=2048
)

# ---------- Email Config (Resend) ----------
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

# ---------- Company Context Data ----------
COMPANY_INFO = """
Company Name: NexusTech Innovations
Work Location: 100% Remote globally.
Departments: Engineering, Artificial Intelligence, Product Management, Human Resources, and Sales.
Open Positions & Salaries: 
- Senior Python Developer (5+ years exp) - Salary: $120,000 to $140,000 per year
- Frontend Engineer (React/Next.js, 3+ years exp) - Salary: $90,000 to $115,000 per year
- AI Prompt Engineer / Researcher (LLMs, LangChain) - Salary: $110,000 to $135,000 per year
- Product Manager (B2B SaaS) - Salary: $100,000 to $130,000 per year
Benefits: Comprehensive health insurance, unlimited PTO, remote work setup stipend, and yearly performance bonuses.
"""

# ---------- Pydantic Models ----------
class BasicInfo(BaseModel):
    intent_to_apply: Optional[bool] = Field(description="Set to true ONLY if the user explicitly says they want to apply, submit a resume, or start an application.", default=False)
    name: Optional[str] = Field(description="The user's full name", default=None)
    email: Optional[str] = Field(description="The user's email address", default=None)
    phone: Optional[str] = Field(description="The user's phone or WhatsApp number", default=None)
    experience: Optional[str] = Field(description="The user's years of experience or primary field", default=None)

# ---------- State Definition ----------
class ScreeningState(BaseModel):
    session_id: Optional[str] = None
    intent_to_apply: bool = False
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    experience: Optional[str] = None
    basic_complete: bool = False

    resume_path: Optional[str] = None
    resume_text: Optional[str] = None
    resume_uploaded: bool = False
    resume_filename: Optional[str] = None

    interview_questions_asked: int = 0
    max_interview_questions: int = 4
    interview_complete: bool = False

    hr_questions_asked: int = 0
    max_hr_questions: int = 3
    hr_complete: bool = False

    report_path: Optional[str] = None
    report_generated: bool = False
    email_sent: bool = False

    messages: List[Dict[str, str]] = []
    user_input: str = ""
    user_file: Optional[bytes] = None
    user_filename: Optional[str] = None

# ---------- Helper Functions ----------
def extract_basic_info(text: str) -> dict:
    parser = JsonOutputParser(pydantic_object=BasicInfo)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the text. Extract intent_to_apply (true if they want a job or uploaded a resume), name, email, phone, and experience. If a field is missing, output null. You must return valid JSON.\n{format_instructions}"),
        ("human", "{input}")
    ])
    chain = prompt | llm | parser
    try:
        return chain.invoke({"input": text[:3000], "format_instructions": parser.get_format_instructions()})
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {}

def process_resume_in_memory(file_bytes: bytes, filename: str) -> str:
    text = ""
    if filename.lower().endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            text = f"Error extracting text: {e}"
    else:
        text = "Non-PDF file uploaded."
    return text

def send_email(to_email: str, subject: str, body: str, attachment_bytes: Optional[bytes] = None, attachment_filename: Optional[str] = None) -> bool:
    if not RESEND_API_KEY:
        return False
    try:
        html_body = f"<pre style='font-family: sans-serif;'>{body}</pre>"
        params = {
            "from": "NexusTech HR <awais@toolsmaverick.cloud>", 
            "to": to_email,
            "subject": subject,
            "html": html_body,
        }
        
        if attachment_bytes and attachment_filename:
            file_content = base64.b64encode(attachment_bytes).decode("utf-8")
            params["attachments"] = [
                {"content": file_content, "filename": attachment_filename}
            ]
            
        r = resend.Emails.send(params)
        return True
    except Exception as e:
        sentry_sdk.capture_exception(e)
        return False

# ---------- AI Question Generators ----------
def generate_basic_question(state: ScreeningState) -> str:
    if state.intent_to_apply:
        target = ""
        if not state.name: target = "full name"
        elif not state.email: target = "email address"
        elif not state.phone: target = "phone number"
        elif not state.experience: target = "years of experience"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant at NexusTech Innovations.\nCompany Info:\n{company_info}\n\nRULES:\n1. Keep responses extremely short (max 2 sentences).\n2. DO NOT use markdown formatting like asterisks (*), bold, or italics. Use plain text.\n3. NEVER mention application portals, links, or say you cannot read files. You CAN read files.\n4. NEVER use the exact phrase 'upload your resume'.\n\nThe user is applying. We need their {target}.\nIf they ask a question, answer it concisely, then politely ask for their {target}."),
            ("human", "Conversation history:\n{context}\n\nGenerate your response:")
        ])
        chain = prompt | llm | StrOutputParser()
        context = "\n".join([f"{m['role']}: {m['content']}" for m in state.messages[-4:]]) if state.messages else ""
        return chain.invoke({"target": target, "company_info": COMPANY_INFO, "context": context})
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant at NexusTech Innovations.\nCompany Info:\n{company_info}\n\nRULES:\n1. Keep responses very short and conversational (max 2-3 sentences).\n2. DO NOT use markdown formatting like asterisks (*) or bold text. Use plain text.\n3. NEVER ask for their name or contact info unless they explicitly say they want to apply.\n4. NEVER use the phrase 'upload your resume'.\n\nAnswer the user's questions naturally based on the Company Info. If they are interested, ask if they want to apply."),
            ("human", "Conversation history:\n{context}\n\nGenerate your response:")
        ])
        chain = prompt | llm | StrOutputParser()
        context = "\n".join([f"{m['role']}: {m['content']}" for m in state.messages[-4:]]) if state.messages else ""
        return chain.invoke({"company_info": COMPANY_INFO, "context": context})

def generate_interview_question(state: ScreeningState) -> str:
    q_num = state.interview_questions_asked + 1
    max_q = state.max_interview_questions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior technical interviewer at NexusTech Innovations. This is question {q_num} of {max_q}.\nCompany Info:\n{company_info}\n\nRULES:\n1. Keep responses short and conversational. Ask ONLY ONE question.\n2. DO NOT use markdown formatting like asterisks (*) or bold text. Use plain text.\n3. NEVER say you cannot read files.\n\nProgression:\n- Early (Q1): Ask about a specific skill in their resume.\n- Middle (Q2-Q3): Ask them to describe a project not in their resume or how they solved a tough challenge.\n- Final (Q4+): Ask about system architecture or tools they prefer.\nIf they just answered, follow up naturally to dig deeper."),
        ("human", "Resume: {resume}\n\nConversation so far:\n{context}\n\nGenerate your next question:")
    ])
    chain = prompt | llm | StrOutputParser()
    context = "\n".join([f"{m['role']}: {m['content']}" for m in state.messages[-5:]])
    resume = state.resume_text[:1200] if state.resume_text else "No resume provided."
    return chain.invoke({"q_num": q_num, "max_q": max_q, "resume": resume, "context": context, "company_info": COMPANY_INFO})

def generate_hr_question(state: ScreeningState) -> str:
    q_num = state.hr_questions_asked + 1
    max_q = state.max_hr_questions
    topic_focus = "teamwork, conflict resolution, or workplace culture fit"
    if q_num == 2: topic_focus = "expected salary and compensation requirements"
    elif q_num == max_q: topic_focus = "availability to start, notice period, and location flexibility"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR Manager at NexusTech Innovations. This is question {q_num} of {max_q}.\nCompany Info:\n{company_info}\n\nRULES:\n1. Keep it warm, professional, and concise (max 2 sentences).\n2. DO NOT use markdown formatting like asterisks (*). Use plain text.\n3. Ask ONLY ONE question specifically about: {topic_focus}.\nIf the user asks a question, answer it quickly, then ask your HR question."),
        ("human", "Conversation so far:\n{context}\n\nNext HR response/question about {topic_focus}:")
    ])
    chain = prompt | llm | StrOutputParser()
    context = "\n".join([f"{m['role']}: {m['content']}" for m in state.messages[-4:]])
    return chain.invoke({"q_num": q_num, "max_q": max_q, "topic_focus": topic_focus, "context": context, "company_info": COMPANY_INFO})

def generate_report(state: ScreeningState) -> str:
    transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state.messages])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert recruiter. Generate a structured evaluation report. Include: Candidate Info, Resume Summary, Interview Highlights, Expected Salary & Availability, HR Assessment, and Overall Recommendation."),
        ("human", "Name: {name}\nEmail: {email}\nPhone: {phone}\nExperience: {experience}\n\nResume Excerpt: {resume}\n\nTranscript: {transcript}\n\nGenerate report:")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "name": state.name or "N/A", "email": state.email or "N/A", 
        "phone": state.phone or "N/A", "experience": state.experience or "N/A",
        "resume": state.resume_text[:1500] if state.resume_text else "None",
        "transcript": transcript[:3000]
    })

# ---------- Node Agents ----------
def basic_info_node(state: ScreeningState) -> ScreeningState:
    user_msg = state.user_input.strip()
    if user_msg: state.messages.append({"role": "user", "content": user_msg})

    if state.user_file:
        text = process_resume_in_memory(state.user_file, state.user_filename)
        state.resume_text = text
        state.resume_uploaded = True
        
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', state.user_filename or "resume.pdf")
        storage_path = f"{state.session_id}/{safe_filename}"
        BUCKET_NAME = "resume"
        
        upload_file_to_storage(BUCKET_NAME, storage_path, state.user_file, "application/pdf")
        state.resume_path = storage_path
        state.resume_filename = state.user_filename 
        
        state.user_file = None
        state.intent_to_apply = True 
        
        extracted_from_resume = extract_basic_info(text)
        if extracted_from_resume.get("name") and str(extracted_from_resume["name"]).lower() != "null": state.name = extracted_from_resume["name"]
        if extracted_from_resume.get("email") and str(extracted_from_resume["email"]).lower() != "null": state.email = extracted_from_resume["email"]
        if extracted_from_resume.get("phone") and str(extracted_from_resume["phone"]).lower() != "null": state.phone = extracted_from_resume["phone"]
        if extracted_from_resume.get("experience") and str(extracted_from_resume["experience"]).lower() != "null": state.experience = extracted_from_resume["experience"]

    if user_msg:
        extracted = extract_basic_info(user_msg)
        if extracted.get("intent_to_apply") in [True, "true", "True", "yes", "Yes"]: state.intent_to_apply = True
        if extracted.get("name") and str(extracted["name"]).lower() != "null": state.name = extracted["name"]
        if extracted.get("email") and str(extracted["email"]).lower() != "null": state.email = extracted["email"]
        if extracted.get("phone") and str(extracted["phone"]).lower() != "null": state.phone = extracted["phone"]
        if extracted.get("experience") and str(extracted["experience"]).lower() != "null": state.experience = extracted["experience"]

    if state.intent_to_apply and state.name and state.email and state.phone and state.experience:
        state.basic_complete = True
        if state.resume_uploaded:
            bot_msg = f"Thanks {state.name}! I scanned your details directly from your resume. Let's start the technical screening.\n\n"
            bot_msg += generate_interview_question(state)
        else:
            bot_msg = f"Thanks {state.name}! I have all your basic details. Please upload your resume (PDF or DOC) so we can proceed."
        state.messages.append({"role": "assistant", "content": bot_msg})
        return state

    if not state.messages:
        bot_msg = "Hello! I am the AI Assistant for NexusTech Innovations. I can answer questions about our company, open roles, or help you start a job application. How can I help you today?"
    else:
        bot_msg = generate_basic_question(state)

    state.messages.append({"role": "assistant", "content": bot_msg})
    return state

def resume_upload_node(state: ScreeningState) -> ScreeningState:
    user_msg = state.user_input.strip()
    if user_msg: state.messages.append({"role": "user", "content": user_msg})

    if not state.user_file:
        if user_msg:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI recruiter. The user needs to provide their file.\nRULES:\n1. Keep it strictly 1 sentence.\n2. DO NOT use markdown formatting (no asterisks).\n3. You MUST include the exact phrase 'upload your resume' so the interface shows the upload button.\nAnswer any question briefly, then ask them to upload your resume."),
                ("human", "Conversation history:\n{context}\n\nGenerate your response:")
            ])
            chain = prompt | llm | StrOutputParser()
            context = "\n".join([f"{m['role']}: {m['content']}" for m in state.messages[-3:]])
            bot_msg = chain.invoke({"context": context})
            state.messages.append({"role": "assistant", "content": bot_msg})
        else:
            state.messages.append({"role": "assistant", "content": "Please upload your resume (PDF/DOC) so we can continue."})
        return state

    text = process_resume_in_memory(state.user_file, state.user_filename)
    state.resume_text = text
    state.resume_uploaded = True
    
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', state.user_filename or "resume.pdf")
    storage_path = f"{state.session_id}/{safe_filename}"
    BUCKET_NAME = "resume"
    
    upload_file_to_storage(BUCKET_NAME, storage_path, state.user_file, "application/pdf")
    state.resume_path = storage_path
    state.resume_filename = state.user_filename 
    
    bot_msg = generate_interview_question(state)
    state.messages.append({"role": "assistant", "content": f"Resume received! Let's start the technical interview.\n\n{bot_msg}"})
    state.user_file = None 
    return state

def interview_node(state: ScreeningState) -> ScreeningState:
    user_msg = state.user_input.strip()
    if user_msg:
        state.messages.append({"role": "user", "content": user_msg})
        state.interview_questions_asked += 1

    if state.interview_questions_asked >= state.max_interview_questions:
        state.interview_complete = True
        bot_msg = generate_hr_question(state)
        state.messages.append({"role": "assistant", "content": f"Great! Let's wrap up with a few quick HR questions.\n\n{bot_msg}"})
        return state

    bot_msg = generate_interview_question(state)
    state.messages.append({"role": "assistant", "content": bot_msg})
    return state

def hr_node(state: ScreeningState) -> ScreeningState:
    user_msg = state.user_input.strip()
    if user_msg:
        state.messages.append({"role": "user", "content": user_msg})
        state.hr_questions_asked += 1

    if state.hr_questions_asked >= state.max_hr_questions:
        state.hr_complete = True
        return state

    bot_msg = generate_hr_question(state)
    state.messages.append({"role": "assistant", "content": bot_msg})
    return state

def report_node(state: ScreeningState) -> ScreeningState:
    try:
        report_content = generate_report(state)
        filename = f"candidate_{str(state.name).replace(' ', '_')}_{str(uuid.uuid4())[:8]}.txt"
        storage_path = f"{state.session_id}/{filename}"
        
        upload_file_to_storage("reports", storage_path, report_content.encode("utf-8"), "text/plain")
        state.report_path = storage_path
    except Exception as e:
        sentry_sdk.capture_exception(e)
    
    state.report_generated = True
    return state

def email_node(state: ScreeningState) -> ScreeningState:
    COMPANY_HR_EMAIL = "awais.ok612@gmail.com"
    
    if state.email:
        candidate_body = f"Dear {state.name},\n\nThank you for completing your application and interview with NexusTech Innovations. Your profile and evaluation are under review. We will contact you soon.\n\nBest,\nNexusTech HR Team"
        send_email(state.email, "Application Received - NexusTech Innovations", candidate_body)
        
        report_content = "Report generation failed or path is missing."
        if state.report_path:
            try:
                report_bytes = download_file_from_storage("reports", state.report_path)
                if report_bytes:
                    report_content = report_bytes.decode("utf-8")
            except Exception: pass
                
        company_body = f"""NEW CANDIDATE SCREENING COMPLETE
================================
Name: {state.name}
Email: {state.email}
Phone: {state.phone}
Experience: {state.experience}

--- AI EVALUATION REPORT ---
{report_content}

(Please find the candidate's original resume attached to this email.)
"""
        resume_bytes = download_file_from_storage("resume", state.resume_path) if state.resume_path else None
        
        final_filename = state.resume_filename
        if not final_filename and state.resume_path:
            final_filename = state.resume_path.split('/')[-1]

        send_email(
            COMPANY_HR_EMAIL, 
            f"New Candidate Screening: {state.name}", 
            company_body, 
            attachment_bytes=resume_bytes, 
            attachment_filename=final_filename
        )

    state.email_sent = True
    bot_msg = "Thank you! That concludes the interview. I have generated your evaluation report and sent a confirmation email to your inbox. We'll be in touch soon! Feel free to ask me any other questions you might have."
    state.messages.append({"role": "assistant", "content": bot_msg})
    return state

def general_chat_node(state: ScreeningState) -> ScreeningState:
    user_msg = state.user_input.strip()
    if user_msg:
        state.messages.append({"role": "user", "content": user_msg})
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant at NexusTech Innovations. The candidate has already completed their screening interview. Answer any further questions they have about the company, roles, or general topics.\nCompany Info:\n{company_info}\n\nRULES:\n1. Keep responses short and conversational.\n2. DO NOT use markdown formatting like asterisks (*).\n3. Be polite and helpful."),
            ("human", "Conversation history:\n{context}\n\nGenerate your response:")
        ])
        chain = prompt | llm | StrOutputParser()
        context = "\n".join([f"{m['role']}: {m['content']}" for m in state.messages[-4:]])
        bot_msg = chain.invoke({"company_info": COMPANY_INFO, "context": context})
        state.messages.append({"role": "assistant", "content": bot_msg})
    return state

# ---------- Graph Router & Build ----------
def route_step(state: ScreeningState):
    if state.email_sent and state.user_input.strip():
        return "general_chat"
        
    target_node = END
    if not state.basic_complete: target_node = "basic_info"
    elif not state.resume_uploaded: target_node = "resume_upload"
    elif not state.interview_complete: target_node = "interview"
    elif not state.hr_complete: target_node = "hr"
    return target_node

builder = StateGraph(ScreeningState)
builder.add_node("basic_info", basic_info_node)
builder.add_node("resume_upload", resume_upload_node)
builder.add_node("interview", interview_node)
builder.add_node("hr", hr_node)
builder.add_node("report", report_node)
builder.add_node("email", email_node)
builder.add_node("general_chat", general_chat_node)

builder.set_conditional_entry_point(
    route_step,
    {
        "basic_info": "basic_info", "resume_upload": "resume_upload",
        "interview": "interview", "hr": "hr", "general_chat": "general_chat", END: END
    }
)

builder.add_edge("general_chat", END)
builder.add_edge("basic_info", END)
builder.add_edge("resume_upload", END)
builder.add_edge("interview", END)

def hr_after(state: ScreeningState):
    return "report" if state.hr_complete else END

builder.add_conditional_edges("hr", hr_after, {"report": "report", END: END})
builder.add_edge("report", "email")
builder.add_edge("email", END)

graph = builder.compile()

# ---------- FastAPI ----------
app = FastAPI()

@app.get("/sentry-debug")
async def trigger_error():
    raise Exception("SENTRY TEST - FastAPI error")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(session_id: Optional[str] = Form(None), message: str = Form(""), file: Optional[UploadFile] = File(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
        
    state = load_session(session_id, ScreeningState)
    if not state:
        state = ScreeningState(session_id=session_id)
    
    state.user_input = message
    if file:
        state.user_file = await file.read()
        state.user_filename = file.filename

    try:
        updated_state_dict = graph.invoke(state.dict(), config={"recursion_limit": 10})
        updated_state = ScreeningState(**updated_state_dict)
        
        save_session(session_id, updated_state)
        
        last_msg = updated_state.messages[-1]["content"] if updated_state.messages else "No response."
        done = False 
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ [GRAPH ERROR] {error_details}")
        sentry_sdk.capture_exception(e)
        
        return {
            "session_id": session_id, 
            "response": f"🚨 SERVER ERROR: {str(e)}", 
            "done": False
        }

    return {"session_id": session_id, "response": last_msg, "done": done}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)