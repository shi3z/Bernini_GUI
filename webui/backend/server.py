"""
Bernini WebUI Backend - FastAPI server with job queue management.
"""

import asyncio
import os
import sys
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import threading
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Change to project root for relative paths

app = FastAPI(title="Bernini WebUI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, "Job"] = {}
JOB_QUEUE: asyncio.Queue = None
PIPELINE = None
DEVICE = None
OUTPUT_DIR = None
CURRENT_JOB_ID = None
WEBSOCKET_CONNECTIONS: dict[str, WebSocket] = {}


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.PENDING
    task_type: str
    prompt: str
    guidance_mode: Optional[str] = None
    video_path: Optional[str] = None
    image_paths: list[str] = []
    num_frames: int = 81
    num_inference_steps: int = 40
    seed: int = 42
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 40
    output_path: Optional[str] = None
    error: Optional[str] = None
    eta_seconds: Optional[float] = None


GUIDANCE_MODE_BY_TASK = {
    "t2i": "t2v_apg", "t2v": "t2v_apg", "i2i": "v2v",
    "v2v": "v2v_apg", "mv2v": "v2v_apg", "r2v": "r2v_apg",
    "rv2v": "rv2v", "ads2v": "v2v_apg",
}


@app.on_event("startup")
async def startup():
    global JOB_QUEUE, OUTPUT_DIR
    JOB_QUEUE = asyncio.Queue()
    OUTPUT_DIR = tempfile.mkdtemp(prefix="bernini_webui_")
    print(f"Output directory: {OUTPUT_DIR}")
    asyncio.create_task(worker_loop())


async def broadcast(job: Job):
    dead = []
    for cid, ws in WEBSOCKET_CONNECTIONS.items():
        try:
            await ws.send_json(job.model_dump())
        except:
            dead.append(cid)
    for cid in dead:
        WEBSOCKET_CONNECTIONS.pop(cid, None)


async def worker_loop():
    global PIPELINE, DEVICE, CURRENT_JOB_ID

    while True:
        job_id = await JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if not job or job.status == JobStatus.CANCELLED:
            continue

        CURRENT_JOB_ID = job_id
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        job.total_steps = job.num_inference_steps
        await broadcast(job)

        try:
            if PIPELINE is None:
                print("Loading pipeline...")
                await load_pipeline()
                print("Pipeline loaded!")

            output = await run_inference(job)
            job.status = JobStatus.COMPLETED
            job.output_path = output
            job.progress = 100.0
            job.current_step = job.total_steps

        except Exception as e:
            import traceback
            traceback.print_exc()
            job.status = JobStatus.FAILED
            job.error = str(e)

        job.completed_at = datetime.now().isoformat()
        CURRENT_JOB_ID = None

        # Clean up uploaded assets
        if job.video_path and os.path.exists(job.video_path):
            os.remove(job.video_path)
        for img_path in job.image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)

        await broadcast(job)


async def load_pipeline():
    global PIPELINE, DEVICE
    import torch
    from bernini.pipeline import BerniniRendererPipeline

    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)

    PIPELINE = BerniniRendererPipeline.from_pretrained(
        "configs/bernini_renderer_wan22",
        high_noise_ckpt="Bernini/bernini_renderer_high",
        low_noise_ckpt="Bernini/bernini_renderer_low",
        device=DEVICE,
        load_ckpt_weights=True,
        use_unipc=True,
        use_src_id_rotary_emb=True,
    )


async def run_inference(job: Job) -> str:
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.prompt_enhancer import get_system_prompt_for_task

    ext = "png" if job.task_type in ["t2i", "i2i"] else "mp4"
    output_path = os.path.join(OUTPUT_DIR, f"{job.id}.{ext}")
    guidance_mode = job.guidance_mode or GUIDANCE_MODE_BY_TASK.get(job.task_type, "t2v_apg")

    kwargs = {
        "prompt": job.prompt,
        "neg_prompt": DEFAULT_NEG_PROMPT,
        "video": [job.video_path] if job.video_path else None,
        "image": job.image_paths[0] if job.image_paths and job.task_type == "i2i" else None,
        "images": job.image_paths if job.image_paths and job.task_type in ["r2v", "rv2v", "ads2v"] else None,
        "num_inference_steps": job.num_inference_steps,
        "num_frames": 1 if job.task_type in ["t2i", "i2i"] else job.num_frames,
        "seed": job.seed,
        "guidance_mode": guidance_mode,
        "system_prompt": get_system_prompt_for_task(job.task_type),
        "output_path": output_path,
    }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: PIPELINE(write_output=True, **kwargs))
    return result or output_path


OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gpt-oss:20b"

T2V_ENHANCE_PROMPT = """You are a film director enhancing prompts for video generation.
Enhance the user's prompt by adding cinematic details:
- Lighting (soft, hard, natural, artificial)
- Color tone (warm, cool, mixed)
- Camera angle (medium shot, close-up, wide shot)
- Composition (center, symmetrical, rule of thirds)
- Motion description (smooth, dynamic, subtle)

Keep the original intent. Output must be in English, 60-200 words.
Do NOT explain what you did, just output the enhanced prompt directly."""


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_loaded": PIPELINE is not None, "queue_size": JOB_QUEUE.qsize() if JOB_QUEUE else 0}


class EnhanceRequest(BaseModel):
    prompt: str
    task_type: str = "t2v"


@app.post("/api/enhance-prompt")
async def enhance_prompt(req: EnhanceRequest):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/chat/completions",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": T2V_ENHANCE_PROMPT},
                        {"role": "user", "content": req.prompt}
                    ],
                    "max_tokens": 1024,
                }
            )
            data = response.json()
            enhanced = data["choices"][0]["message"]["content"].strip()
            return {"original": req.prompt, "enhanced": enhanced}
    except Exception as e:
        return {"original": req.prompt, "enhanced": req.prompt, "error": str(e)}


@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": [j.model_dump() for j in JOBS.values()]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.model_dump()


@app.post("/api/jobs")
async def create_job(
    task_type: str = Form(...),
    prompt: str = Form(...),
    guidance_mode: Optional[str] = Form(None),
    num_frames: int = Form(81),
    num_inference_steps: int = Form(40),
    seed: int = Form(42),
    video: Optional[UploadFile] = File(None),
    images: list[UploadFile] = File(default=[]),
):
    job_id = str(uuid.uuid4())[:8]

    video_path = None
    if video and video.filename:
        video_path = os.path.join(OUTPUT_DIR, f"{job_id}_video_{video.filename}")
        with open(video_path, "wb") as f:
            f.write(await video.read())

    image_paths = []
    for i, img in enumerate(images):
        if img and img.filename:
            p = os.path.join(OUTPUT_DIR, f"{job_id}_img{i}_{img.filename}")
            with open(p, "wb") as f:
                f.write(await img.read())
            image_paths.append(p)

    job = Job(
        id=job_id,
        task_type=task_type,
        prompt=prompt,
        guidance_mode=guidance_mode,
        video_path=video_path,
        image_paths=image_paths,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        created_at=datetime.now().isoformat(),
        total_steps=num_inference_steps,
    )
    JOBS[job_id] = job
    await JOB_QUEUE.put(job_id)
    await broadcast(job)

    return {"job_id": job_id, "queue_position": JOB_QUEUE.qsize()}


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404)
    if job.status == JobStatus.RUNNING:
        raise HTTPException(400, "Cannot cancel running job")
    job.status = JobStatus.CANCELLED
    await broadcast(job)
    return {"status": "cancelled"}


@app.get("/api/jobs/{job_id}/output")
async def get_output(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.output_path:
        raise HTTPException(404)
    return FileResponse(job.output_path)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cid = str(uuid.uuid4())
    WEBSOCKET_CONNECTIONS[cid] = ws
    try:
        for job in JOBS.values():
            await ws.send_json(job.model_dump())
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        WEBSOCKET_CONNECTIONS.pop(cid, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
