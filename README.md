GatiLochana
The eyes on traffic

Team members: Aditya Suri, Abhishek Yadav, Srihan Uppala, Deekshith

We didn't have time to learn github or to create a good readme or even a file structure

so here is some freshly baked AI slop for you

NOTE: Most of the features are still under progress


GatiLochana gives traffic lights real-time vision through computer vision and adaptive logic. Instead of fixed timers, intersections adjust to actual traffic conditions, with instant priority for emergency vehicles.

What It Does
- Detects vehicles in real time using YOLOv8
- Calculates optimal green-light durations based on traffic density
- Gives immediate priority to ambulances
- Runs a full SUMO traffic simulation
- Displays live traffic state with a Pygame dashboard

## Tech Stack
- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- SUMO
- Pygame

## Quick Start

Clone the repository:
```bash
git clone https://github.com/Udbhav2025/HT100-CV-009.git
cd HT100-CV-009
```

Setup:
**Windows**
```cmd
pip install ultralytics opencv-python numpy
```



Run the system:
```bash
python main.py                  

```



## How It Works
1. YOLOv8 detects vehicles in each frame  
2. Counts are weighted to estimate traffic load  
3. The logic engine calculates the ideal green-light duration  
4. Ambulances trigger an immediate override  
5. SUMO simulates updated timings, and Pygame visualizes the state  

## Project Structure
```

main.py       all the code
```

## Why It Matters
- Reduces congestion by adapting to real demand
- Helps emergency vehicles move faster and more safely
- Combines computer vision, simulation, and automation
- Easily extendable to real-world intersections

## Contributions
Feel free to fork the project and submit improvements.

# Built for Hackathon Innovation
Smarter cities begin with smarter traffic.
