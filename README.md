# AHGI — Assistant Hand Gesture Interface

A real-time computer vision application that transforms webcam-based hand gestures into interactive visual effects. Built with Python, OpenCV, and MediaPipe, AHGI features gesture-controlled drawing, customizable visual filters, and a custom camera interface.

## Features

### Real-Time Hand Tracking
- Detects and tracks up to two hands using MediaPipe Hand Landmarker.
- Visualizes hand landmarks and finger joints with a custom skeleton overlay.
- Uses gesture recognition to identify specific hand poses.
- Applies point smoothing to reduce unwanted jitter during movement.

### Gesture-Controlled Drawing
- Activate drawing mode by holding a pointing gesture for approximately two seconds.
- Draw using index-finger movements.
- Generate fading neon trails with glow effects.
- Exit drawing mode by making a fist or releasing the pointing gesture.

### Interactive Visual Filters
- Create triangular or quadrilateral regions using two-hand gestures.
- Activate a region by holding the required gesture.
- Apply one of four visual filters within the selected region:
  - **MONO:** Enhanced grayscale with a film-grain effect.
  - **NEON:** Edge detection with colored neon outlines and glow.
  - **HEAT:** Inferno-style color mapping.
  - **GRAINY:** Grayscale with film grain and a subtle warm tint.
- Cycle between filters by lowering a raised pinky finger after it has been raised.

### Custom Camera Interface
- Mirrored webcam feed for intuitive interaction.
- Custom hand skeletons, gesture progress rings, and region outlines.
- On-screen FPS, date, time, and application status.
- Fullscreen mode and keyboard shortcuts.
- Automatic region closure when hands are lost for a specified period.

## Technologies

- **Python** — Application logic and gesture processing.
- **OpenCV** — Webcam capture, image processing, visual effects, drawing, and interface rendering.
- **MediaPipe Tasks Vision** — Hand landmark detection and video-based tracking.
- **NumPy** — Image arrays, polygon geometry, smoothing, and pixel-level operations.

## Requirements

- Python installed on your computer.
- A working webcam.
- The Python packages listed in `requirements.txt`.
- An internet connection for the initial download of the MediaPipe hand landmark model.

## Installation

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. (Recommended) Create and activate a Python virtual environment.
4. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   python main.py
   ```

On the first launch, the application downloads the MediaPipe hand landmark model automatically. Subsequent launches use the downloaded model file.

## Controls

| Input | Action |
|---|---|
| Hold a pointing gesture for approximately 2 seconds | Activate drawing mode |
| Move your index finger while drawing | Create a neon trail |
| Make a fist while drawing | Exit drawing mode |
| Form the supported two-hand frame gesture and hold | Activate a visual filter region |
| Lower a previously raised pinky | Cycle to the next visual filter |
| Bring both palms together briefly | Close the active filter region |
| Lose hand tracking for approximately 1.5 seconds | Automatically close the active filter region |
| `F` | Toggle fullscreen mode |
| `C` | Reset selected application state and counters |
| `Q` or `Esc` | Exit the application |

## How It Works

1. **Capture:** OpenCV reads frames from the webcam and mirrors the image.
2. **Detection:** MediaPipe identifies hand landmarks in each video frame.
3. **Gesture recognition:** Landmark positions are converted into pixel coordinates and evaluated against custom finger-pose rules.
4. **Interaction:** Recognized gestures activate drawing mode or define a filter region.
5. **Rendering:** OpenCV applies visual effects, renders hand landmarks and interface elements, and displays the resulting frame.

## Configuration

Several constants near the beginning of `main.py` control the application's behavior, including:

- Gesture activation duration.
- Drawing trail lifetime.
- Webcam resolution.
- Gesture detection tolerance.
- Point smoothing.
- Palm proximity and hand-loss timeouts.

These settings can be adjusted to experiment with responsiveness and interaction behavior.

## Future Improvements

- Add configurable gestures and user-defined commands.
- Improve gesture recognition across different hand orientations and lighting conditions.
- Add an interface for adjusting filter and tracking settings.
- Improve installation and dependency version management.
- Record a demonstration showing the supported gestures and visual effects.

## Project Status

A personal Python computer vision project exploring real-time hand tracking, gesture-based interaction, and interactive image processing.
