# Quick Test Guide - Video Optimization

## 🧪 Testing the Optimizations

### Step 1: Start Backend Server

```bash
cd api_service
python main.py
```

Expected output:
```
================================================================================
Loading Dog Emotion Recognition Models...
================================================================================

✅ Models loaded successfully!
🖥️  Running on: CPU (or GPU)
================================================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start Frontend Development Server

```bash
cd web_intf
npm run dev
```

Expected output:
```
  VITE v8.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

### Step 3: Test Video Upload with Progress

1. Open browser: `http://localhost:5173`
2. Click "🎬 Upload Video" button
3. Select a video file (MP4, < 20 seconds, < 50MB)
4. **Observe**:
   - ✅ Progress bar should animate smoothly from 0% to 100%
   - ✅ Console logs show: "Progress: 10.0% (10/100 frames)"
   - ✅ Processing completes faster than before
   - ✅ Video plays with smooth bounding box animations

### Step 4: Verify Performance Improvements

**Check Browser Console:**
```javascript
// You should see progress updates like:
Progress: 5.0% (5/100 frames)
Progress: 10.0% (10/100 frames)
Progress: 15.0% (15/100 frames)
...
Video analysis complete: 100 frames analyzed
```

**Check Backend Logs:**
```python
Video properties: 30.0fps, 600 frames, 20.00s duration
Sampling at 5.0fps (every 6 frames)
Processed 10 frames...
Processed 20 frames...
...
Analysis complete: Analyzed 100 frame(s) at 5.0fps over 20.0 seconds
```

---

## 🔍 Troubleshooting

### Issue 1: Progress bar still stuck at 0%

**Possible causes:**
- Frontend not using new SSE endpoint
- CORS issues blocking streaming response

**Solution:**
```bash
# Check if frontend is calling correct endpoint
# In browser DevTools > Network tab, look for:
POST /api/analyze-video-stream

# Verify response type is "text/event-stream"
```

### Issue 2: No improvement in processing speed

**Possible causes:**
- Still using old `/api/analyze-video` endpoint instead of stream version
- Model loading failed

**Solution:**
```bash
# Verify backend is running the new code
# Look for this in terminal when uploading video:
"Video properties: ..."  # Should appear immediately

# Check that no temporary files are created:
ls /tmp/*.jpg  # Should be empty during processing
```

### Issue 3: SSE connection fails

**Check CORS configuration:**
```python
# In api_service/main.py, verify:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    ...
)
```

---

## 📊 Performance Metrics

### Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| 10-second video (50 frames) | ~15-18s | ~10-12s | **~33% faster** |
| Progress feedback | None | Real-time | **100% better UX** |
| Temp files created | 50+ | 0 | **Eliminated** |
| Memory usage | Higher | Lower | **More efficient** |

### Monitoring Tips

**Backend (Python):**
```python
import time
start = time.time()
# ... process frames ...
elapsed = time.time() - start
print(f"Total processing time: {elapsed:.2f}s")
print(f"Average per frame: {elapsed/num_frames*1000:.0f}ms")
```

**Frontend (JavaScript):**
```javascript
const startTime = Date.now();
const results = await analyzeVideo(videoFile, (progress) => {
  const elapsed = (Date.now() - startTime) / 1000;
  console.log(`Progress: ${progress}% after ${elapsed.toFixed(1)}s`);
});
```

---

## ✅ Success Criteria

You know the optimization is working when:

1. ✅ Progress bar animates smoothly (not jumping from 0% to 100%)
2. ✅ Console shows incremental progress updates every frame
3. ✅ Total processing time is reduced by ~30-50%
4. ✅ No temporary `.jpg` files appear in `/tmp/` during processing
5. ✅ Video playback remains smooth with interpolated annotations

---

**Last Updated**: 2026-04-25
