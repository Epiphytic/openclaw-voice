# Speaker Enrollment Guide

To enable personalized responses (e.g., "Welcome back, Alice"), you must enroll speakers by providing voice samples.

## Prerequisites

- `openclaw-voice` running with the Speaker ID service enabled.
- A collection of clean audio samples (WAV/MP3) for each person, ideally 5-10 seconds long.
- `curl` or a similar HTTP client.

## Enrolling a Speaker

Use the `/enroll` endpoint on the Speaker ID server (default port 8003).

### 1. Collect Audio
Record 3-5 different samples of the person speaking naturally. Vary the tone and content.
```bash
# Example recording with ffmpeg (or use Voice Memos)
ffmpeg -f alsa -i default -t 5 -ac 1 -ar 16000 alice_sample1.wav
```

### 2. Send Enrollment Request
Send each sample to the server. The server averages new embeddings with existing ones for that name.

**Parameters:**
- `name`: The unique ID/name for the person (e.g., "Alice").
- `access_level`: Authorization level (`full`, `standard`, `basic`).
- `file`: The audio file.

```bash
curl -X POST "http://localhost:8003/enroll" \
     -F "name=Alice" \
     -F "access_level=full" \
     -F "file=@alice_sample1.wav"
```

Repeat for all samples:
```bash
curl -X POST "http://localhost:8003/enroll" -F "name=Alice" -F "file=@alice_sample2.wav"
curl -X POST "http://localhost:8003/enroll" -F "name=Alice" -F "file=@alice_sample3.wav"
```

### 3. Verify Enrollment
Check the list of enrolled speakers:
```bash
curl "http://localhost:8003/speakers"
```
Response:
```json
{
  "speakers": [
    {
      "name": "Alice",
      "access_level": "full",
      "samples": 3,
      "enrolled_at": 1708812345.0
    }
  ]
}
```

## Testing Identification

Test the system with a new audio clip:
```bash
curl -X POST "http://localhost:8003/identify" \
     -F "file=@test_clip.wav" \
     -F "threshold=0.75"
```

Response:
```json
{
  "speaker": "Alice",
  "confidence": 0.92,
  "access_level": "full",
  "elapsed": 0.15
}
```

## Managing Profiles

Profiles are stored as JSON files in the configured `profiles_dir` (default: `./speaker-profiles/`).

**To delete a speaker:**
```bash
curl -X DELETE "http://localhost:8003/speakers/Alice"
```
Or simply delete the corresponding JSON file from the disk.
