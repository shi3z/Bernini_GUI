import { useState, useEffect, useRef } from 'react'
import './App.css'

interface Job {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  task_type: string
  prompt: string
  guidance_mode?: string
  video_path?: string
  image_paths: string[]
  num_frames: number
  num_inference_steps: number
  progress: number
  current_step: number
  total_steps: number
  output_path?: string
  error?: string
  created_at: string
}

const TASK_TYPES = [
  { value: 't2i', label: 'Text to Image', needsVideo: false, needsImages: false, imageCount: 0 },
  { value: 't2v', label: 'Text to Video', needsVideo: false, needsImages: false, imageCount: 0 },
  { value: 'i2i', label: 'Image to Image', needsVideo: false, needsImages: true, imageCount: 1 },
  { value: 'v2v', label: 'Video to Video', needsVideo: true, needsImages: false, imageCount: 0 },
  { value: 'r2v', label: 'Reference to Video', needsVideo: false, needsImages: true, imageCount: 2 },
  { value: 'rv2v', label: 'Reference + Video', needsVideo: true, needsImages: true, imageCount: 2 },
]

const API_BASE = `http://${window.location.hostname}:8000`
const randomSeed = () => Math.floor(Math.random() * 1000000)

interface ImageSlot {
  file: File | null
  preview: string | null
}

function App() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [taskType, setTaskType] = useState('t2v')
  const [prompt, setPrompt] = useState('')
  const [numFrames, setNumFrames] = useState(81)
  const [numSteps, setNumSteps] = useState(40)
  const [seed, setSeed] = useState(randomSeed)
  const [video, setVideo] = useState<File | null>(null)
  const [videoPreview, setVideoPreview] = useState<string | null>(null)
  const [imageSlots, setImageSlots] = useState<ImageSlot[]>([
    { file: null, preview: null },
    { file: null, preview: null },
  ])
  const [connected, setConnected] = useState(false)
  const [enhancing, setEnhancing] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [dragVideo, setDragVideo] = useState(false)
  const [dragImage, setDragImage] = useState<number | null>(null)
  const [nowTick, setNowTick] = useState(Date.now())
  const wsRef = useRef<WebSocket | null>(null)
  // Per-job step timing for a frontend-only ETA: first/last observed step+time.
  const etaRef = useRef<Map<string, { firstStep: number; firstTime: number; lastStep: number; lastTime: number }>>(new Map())

  useEffect(() => {
    connectWebSocket()
    fetchJobs()
    // tick once a second so the ETA counts down live between step updates
    const t = setInterval(() => setNowTick(Date.now()), 1000)
    return () => { wsRef.current?.close(); clearInterval(t) }
  }, [])

  const connectWebSocket = () => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`)
    ws.onopen = () => setConnected(true)
    ws.onmessage = (e) => {
      const job = JSON.parse(e.data) as Job
      recordStepTiming(job)
      setJobs(prev => {
        const idx = prev.findIndex(j => j.id === job.id)
        if (idx >= 0) {
          const updated = [...prev]
          updated[idx] = job
          return updated
        }
        return [job, ...prev]
      })
    }
    ws.onclose = () => {
      setConnected(false)
      setTimeout(connectWebSocket, 3000)
    }
    wsRef.current = ws
  }

  const fetchJobs = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs`)
      const data = await res.json()
      setJobs(data.jobs.reverse())
    } catch (e) { console.error(e) }
  }

  // --- frontend-only ETA: estimate from observed per-step timing ---
  const recordStepTiming = (job: Job) => {
    if (job.status !== 'running' || job.current_step <= 0) {
      if (job.status !== 'running') etaRef.current.delete(job.id)
      return
    }
    const now = Date.now()
    const t = etaRef.current.get(job.id)
    if (!t || job.current_step < t.lastStep) {
      // first sample for this run (or step counter reset) — seed the baseline
      etaRef.current.set(job.id, { firstStep: job.current_step, firstTime: now, lastStep: job.current_step, lastTime: now })
    } else if (job.current_step > t.lastStep) {
      t.lastStep = job.current_step
      t.lastTime = now
    }
  }

  // seconds remaining, or null if not enough samples yet
  const computeEtaSeconds = (job: Job): number | null => {
    if (job.status !== 'running') return null
    const t = etaRef.current.get(job.id)
    if (!t || t.lastStep <= t.firstStep) return null
    const perStep = (t.lastTime - t.firstTime) / 1000 / (t.lastStep - t.firstStep)
    const remainingSteps = Math.max(0, job.total_steps - job.current_step)
    const elapsedSinceLast = (nowTick - t.lastTime) / 1000
    return Math.max(0, remainingSteps * perStep - elapsedSinceLast)
  }

  const formatDuration = (s: number): string => {
    s = Math.round(s)
    const m = Math.floor(s / 60)
    const sec = s % 60
    return m > 0 ? `${m}:${sec.toString().padStart(2, '0')}` : `${sec}s`
  }

  const setVideoFile = (file: File | null) => {
    setVideo(file)
    setVideoPreview(file ? URL.createObjectURL(file) : null)
  }

  const setImageFile = (index: number, file: File | null) => {
    setImageSlots(prev => {
      const next = [...prev]
      next[index] = file
        ? { file, preview: URL.createObjectURL(file) }
        : { file: null, preview: null }
      return next
    })
  }

  const removeImage = (index: number) => setImageFile(index, null)
  const removeVideo = () => setVideoFile(null)

  const submitJob = async () => {
    if (!prompt.trim()) return alert('Enter a prompt')
    const formData = new FormData()
    formData.append('task_type', taskType)
    formData.append('prompt', prompt)
    formData.append('num_frames', numFrames.toString())
    formData.append('num_inference_steps', numSteps.toString())
    formData.append('seed', seed.toString())
    if (video) formData.append('video', video)
    imageSlots.forEach(slot => { if (slot.file) formData.append('images', slot.file) })

    setSubmitting(true)
    try {
      await fetch(`${API_BASE}/api/jobs`, { method: 'POST', body: formData })
      // Keep prompt and reference images so the next request can reuse them;
      // only roll a fresh seed so repeated submits don't repeat the same output.
      setSeed(randomSeed())
    } catch (e) { alert('Failed to submit') }
    finally { setSubmitting(false) }
  }

  const cancelJob = async (id: string) => {
    await fetch(`${API_BASE}/api/jobs/${id}`, { method: 'DELETE' })
  }

  const enhancePrompt = async () => {
    if (!prompt.trim()) return
    setEnhancing(true)
    try {
      const res = await fetch(`${API_BASE}/api/enhance-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, task_type: taskType })
      })
      const data = await res.json()
      if (data.enhanced && !data.error) {
        setPrompt(data.enhanced)
      } else if (data.error) {
        alert(`Enhancement failed: ${data.error}`)
      }
    } catch (e) {
      alert('Failed to enhance prompt')
    } finally {
      setEnhancing(false)
    }
  }

  const currentTask = TASK_TYPES.find(t => t.value === taskType)
  const statusColor: Record<string, string> = {
    pending: '#f59e0b', running: '#3b82f6', completed: '#10b981',
    failed: '#ef4444', cancelled: '#6b7280'
  }

  // ---- drag & drop helpers ----
  const onVideoDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragVideo(false)
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('video/')) setVideoFile(f)
  }
  const onImageDrop = (index: number) => (e: React.DragEvent) => {
    e.preventDefault(); setDragImage(null)
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('image/')) setImageFile(index, f)
  }

  return (
    <div className="app">
      <header>
        <h1>Bernini WebUI</h1>
        <span className={connected ? 'connected' : 'disconnected'}>
          {connected ? '● Connected' : '○ Disconnected'}
        </span>
      </header>

      <div className="main">
        <div className="panel">
          <h2>New Job</h2>
          <div className="form-group">
            <label>Task Type</label>
            <select value={taskType} onChange={e => setTaskType(e.target.value)}>
              {TASK_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label>Prompt</label>
            <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={4} placeholder="Describe..." />
            <button
              className="enhance-btn"
              onClick={enhancePrompt}
              disabled={enhancing || !prompt.trim()}
            >
              {enhancing ? 'Enhancing...' : '✨ Enhance with AI (ollama)'}
            </button>
          </div>

          {currentTask?.needsVideo && (
            <div className="form-group">
              <label>Source Video (編集元の動画)</label>
              <div className="media-slot">
                {videoPreview ? (
                  <div className="media-preview">
                    <video src={videoPreview} />
                    <button className="remove-btn" onClick={removeVideo}>×</button>
                    <span className="media-label">Source Video</span>
                  </div>
                ) : (
                  <label
                    className={`upload-box${dragVideo ? ' dragover' : ''}`}
                    onDragOver={e => { e.preventDefault(); setDragVideo(true) }}
                    onDragLeave={() => setDragVideo(false)}
                    onDrop={onVideoDrop}
                  >
                    <input type="file" accept="video/*" onChange={e => setVideoFile(e.target.files?.[0] || null)} hidden />
                    <span>+ Upload Video</span>
                    <span className="drop-hint">またはドラッグ&ドロップ</span>
                  </label>
                )}
              </div>
            </div>
          )}

          {currentTask?.needsImages && (
            <div className="form-group">
              <label>Reference Images (参照画像)</label>
              <p className="hint">画像は左から順に images[0], images[1] として使用されます</p>
              <div className="image-slots">
                {imageSlots.slice(0, currentTask.imageCount || 2).map((slot, i) => (
                  <div key={i} className="media-slot">
                    {slot.preview ? (
                      <div className="media-preview">
                        <img src={slot.preview} alt={`Image ${i + 1}`} />
                        <button className="remove-btn" onClick={() => removeImage(i)}>×</button>
                        <span className="media-label">Image {i + 1}</span>
                      </div>
                    ) : (
                      <label
                        className={`upload-box${dragImage === i ? ' dragover' : ''}`}
                        onDragOver={e => { e.preventDefault(); setDragImage(i) }}
                        onDragLeave={() => setDragImage(null)}
                        onDrop={onImageDrop(i)}
                      >
                        <input type="file" accept="image/*" onChange={(e) => setImageFile(i, e.target.files?.[0] || null)} hidden />
                        <span className="slot-number">Image {i + 1}</span>
                        <span>+ Upload</span>
                        <span className="drop-hint">D&D可</span>
                      </label>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="form-row">
            <div className="form-group"><label>Frames</label><input type="number" value={numFrames} onChange={e => setNumFrames(+e.target.value)} /></div>
            <div className="form-group"><label>Steps</label><input type="number" value={numSteps} onChange={e => setNumSteps(+e.target.value)} /></div>
            <div className="form-group">
              <label>Seed</label>
              <div className="seed-row">
                <input type="number" value={seed} onChange={e => setSeed(+e.target.value)} />
                <button className="dice-btn" title="ランダムシード" onClick={() => setSeed(randomSeed())}>🎲</button>
              </div>
            </div>
          </div>

          <button className="submit-btn" onClick={submitJob} disabled={submitting}>
            {submitting ? 'Submitting...' : 'Submit Job'}
          </button>
        </div>

        <div className="panel jobs">
          <h2>Jobs ({jobs.filter(j => j.status === 'pending').length} pending, {jobs.filter(j => j.status === 'running').length} running)</h2>
          <div className="jobs-list">
            {jobs.map(job => {
              const isImg = job.task_type.includes('2i')
              return (
              <div key={job.id} className={`job-card ${job.status}`}>
                <div className="job-header">
                  <span className="job-id">#{job.id}</span>
                  <span className="job-type">{job.task_type}</span>
                  <span className="job-status" style={{ background: statusColor[job.status] }}>{job.status}</span>
                </div>
                <p className="job-prompt">{job.prompt.slice(0, 120)}{job.prompt.length > 120 ? '...' : ''}</p>

                {/* input thumbnails (queued reference images / source video) */}
                {(job.image_paths?.length > 0 || job.video_path) && (
                  <div className="job-inputs">
                    {job.video_path && (
                      <video className="thumb" src={`${API_BASE}/api/jobs/${job.id}/video`} muted />
                    )}
                    {job.image_paths?.map((_, i) => (
                      <img key={i} className="thumb" src={`${API_BASE}/api/jobs/${job.id}/input/${i}`} alt={`in${i}`} />
                    ))}
                  </div>
                )}

                {job.status === 'running' && (() => {
                  const eta = computeEtaSeconds(job)
                  return (
                    <div className="progress">
                      <div className="progress-bar" style={{ width: `${job.progress || 0}%` }} />
                      <span>
                        {job.current_step}/{job.total_steps} ({(job.progress || 0).toFixed(1)}%)
                        {' · '}
                        {eta === null ? 'ETA 推定中…' : `残り ${formatDuration(eta)}`}
                      </span>
                    </div>
                  )
                })()}

                {job.status === 'completed' && job.output_path && (
                  <div className="output">
                    {isImg ? (
                      <img src={`${API_BASE}/api/jobs/${job.id}/output`} alt="output" />
                    ) : (
                      <video src={`${API_BASE}/api/jobs/${job.id}/output`} controls loop />
                    )}
                  </div>
                )}

                {job.status === 'failed' && <p className="error">{job.error}</p>}
                {job.status === 'pending' && <button className="cancel-btn" onClick={() => cancelJob(job.id)}>Cancel</button>}
              </div>
            )})}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
