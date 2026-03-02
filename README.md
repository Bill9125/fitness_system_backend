# Fitness System Backend

This is the backend for the Fitness System, built with **Django** and **Django REST Framework (DRF)**. The system is designed to provide comprehensive analysis of fitness training recordings by evaluating repetitions, computing joint angles, and generating AI-powered training suggestions and workout plans.

## Features

- **Action Detection & Evaluation**: Evaluates repetitions in user-uploaded fitness videos, extracting key metrics such as scores, bar positions, hip angles, and knee-to-hip relations.
- **Annotated Video Streaming**: Capable of serving processed videos with body tracking and annotations.
- **AI-Powered Training Suggestions**: Leverages OpenAI to generate and stream personalized training suggestions based on the user's performance and errors.
- **AI-Powered Workout Plans**: Generates and streams structured workout plans tailored to the user.
- **Recommended Videos**: Automatically suggests instructional fitness videos (e.g., proper form for squats, deadlifts) based on the user's specific mistakes.
- **Interactive Documentation**: Integrated with `drf-spectacular` providing a Swagger UI for API exploration.

## Tech Stack

- **Framework**: Django, Django REST Framework
- **Database**: Django ORM (configured in `settings.py`)
- **API Documentation**: drf-spectacular (OpenAPI 3)
- **AI Integration**: OpenAI API (for Suggestions and Workout Plans)

## Project Structure

```text
fitness_system_backend/
├── fitness_system_backend/   # Core Django project configuration
├── fitness_analysis/         # App handling video analysis and AI features
│   ├── models.py             # Database models (Recording, Repetition, etc.)
│   ├── views.py              # API view endpoints
│   ├── urls.py               # API route definitions
│   └── utils.py              # Utilities (OpenAIClient, ProcessorFactory, etc.)
├── users/                    # App for user management (if implemented)
└── manage.py                 # Django command-line utility
```

## API Endpoints Overview

The API is mounted under `/api/analysis/`. Key endpoints include:

- `GET /api/analysis/recordings/<user_id>/` - Retrieve all recording IDs for a specific user.
- `GET /api/analysis/result/<recording_id>/` - Retrieve detailed detection results (scores, angles, repetition splits) for a recording.
- `GET /api/analysis/videos/<recording_id>/<vision_index>/` - Fetch the annotated/processed MP4 video.
- `GET /api/analysis/suggestion/<recording_id>/` - Stream AI-generated training suggestions (Server-Sent Events).
- `GET /api/analysis/workout_plan/<recording_id>/` - Stream AI-generated workout plans (Server-Sent Events).

## API Documentation

Swagger UI is available to explore and test the APIs natively. Once the server is running, visit:
- **Swagger UI**: `http://127.0.0.1:8000/api/docs/`

## Running the Project

1. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate fitness_system_backend
   ```
2. Apply database migrations:
   ```bash
   python manage.py migrate
   ```
3. Run the development server:
   ```bash
   python manage.py runserver
   ```
