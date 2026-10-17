# Contributing to BiomechAI

Thank you for your interest in contributing! We welcome enhancements in computer vision accuracy, new biomechanical indicators, and user interface designs.

## Development Workflow

1. **Fork the Repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/biomech-ai.git
   cd biomech-ai
   ```
3. **Set up the virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
5. **Verify code compiles**:
   ```bash
   python3 -m py_compile *.py
   ```
6. **Commit your changes**:
   ```bash
   git commit -m "feat: add amazing new posture metric"
   ```
7. **Push to your fork and submit a Pull Request**.

## Guidelines
- Ensure personal snapshots and report outputs remain excluded in `.gitignore`.
- Maintain clean, descriptive variable names and document new biometric mathematical formulas.
