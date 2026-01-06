# Real-time 3D Motion Generation Demo

## Quick Start

```bash
cd web_demo
./server.sh start
```

Visit: **http://localhost:5000**  
Refresh: `Ctrl+Shift+R` (clear cache)

## Features

-   🎨 **Text-driven**: Input motion descriptions to generate 3D animations in real-time
-   🎭 **Streaming Rendering**: Smooth 20 FPS playback
-   🎪 **Colorful Stick Figure**: Different colors for different body parts
-   📹 **Smart Camera Follow**: Auto-follow after 3 seconds of inactivity
-   🌐 **Infinite Scene**: 1000x1000 unit ground with grid

## Usage

1. Enter motion description (e.g., "walk forward", "jump", "dance")
2. Click "Start"
3. Watch real-time generated 3D animation
4. Use "Update Text" to change motion anytime
5. Mouse drag to rotate view, scroll to zoom
6. Camera auto-follows character after 3 seconds of inactivity

## Architecture

```
Text Input → Model(1 token) → VAE(4 frames) →
StreamJointRecovery(22 joints) → Buffer → 20fps Rendering
```

## File Structure

```
web_demo/
├── app.py                # Flask server
├── model_manager.py      # Model and buffer management
├── server.sh            # Server management script
├── start.sh             # Legacy startup script
├── static/
│   ├── css/style.css    # Clean white UI
│   └── js/
│       ├── main.js      # Main logic (20fps + camera follow)
│       └── skeleton.js  # Stick figure rendering (colorful)
└── templates/
    └── index.html       # Web interface
```

## Performance

-   **Model Generation**: 52 FPS on 5090 (76ms/step)
-   **Rendering**: 20 FPS (matches model output)
-   **Buffer**: 4 frames minimum

## Technical Details

### StreamJointRecovery263

-   Process 263-dim motion data frame by frame
-   Output 22 joint 3D coordinates
-   Test verified: Identical to batch processing

### HumanML3D Skeleton

```python
[0,2,5,8,11]      # Spine (orange)
[0,1,4,7,10]      # Left leg (cyan)
[0,3,6,9,12,15]   # Right leg (blue)
[9,14,17,19,21]   # Left arm (amber)
[9,13,16,18,20]   # Right arm (aquamarine)
```

## Server Management

Use `server.sh` for reliable process management:

```bash
./server.sh start    # Start server
./server.sh stop     # Stop server
./server.sh restart  # Restart server
./server.sh status   # Check status
```

## Tiny Model

You can use different model configurations by specifying a config file:

```bash
./server.sh start configs/stream_tiny.yaml
./server.sh restart configs/stream_tiny.yaml
```
