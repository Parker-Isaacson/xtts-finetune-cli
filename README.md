# XTTS CLI Pipeline

A command-line pipeline for fine-tuning [XTTS](https://github.com/coqui-ai/TTS) models, running inference, and packaging the optimized model. Heavily written from [xtts-finetune-webui](https://github.com/daswer123/xtts-finetune-webui)

This project provides:
- **Step 1:** Create a dataset from input audio
- **Step 2:** Train the XTTS model
    - Edit epochs, batch size, etc. here.
- **Step 2.5:** Optimize the trained model
- **Step 3:** Load the optimized model
- **Step 4:** Run inference (generate speech from text)
- **Step 5:** Package the model into a `.zip`

---

## Usage

### 1. Install dependencies

Excluding below make 

```bash
conda create -n xtts-env python=3.11
conda activate xtts-env
pip install -r requirements.txt
```