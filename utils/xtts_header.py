import os
import shutil
import tempfile
from pathlib import Path

import torch
import torchaudio
from faster_whisper import WhisperModel

from .formatter import format_audio_list, list_audios
from .gpt_train import train_gpt
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Step 1 - Create Dataset
def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path):
    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)

    if audio_folder_path:
        audio_files = list(list_audios(audio_folder_path))
    else:
        audio_files = audio_path

    if not audio_files:
        raise ValueError("No audio files found!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "float32"

    asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    train_meta, eval_meta, audio_total_size = format_audio_list(
        audio_files,
        asr_model=asr_model,
        target_language=language,
        out_path=out_path,
    )

    if audio_total_size < 120:
        raise ValueError("Dataset too small. Provide at least 2 minutes of audio.")

    # Ensure ready/ folder exists
    ready_dir = Path(out_path).parent / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)

    # Copy first audio file as reference.wav
    import shutil
    ref_src = audio_files[0]
    ref_dst = ready_dir / "reference.wav"
    shutil.copy(ref_src, ref_dst)
    print(f"Reference audio staged at {ref_dst}")

    return train_meta, eval_meta


# Step 2 - Train
def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs,
                batch_size, grad_acumm, output_path, max_audio_length):
    max_audio_length = int(max_audio_length * 22050)

    # Run training
    train_gpt(
        custom_model,
        version,
        language,
        num_epochs,
        batch_size,
        grad_acumm,
        train_csv,
        eval_csv,
        output_path=output_path,
        max_audio_length=max_audio_length,
    )

    # Find last checkpoint inside run/
    run_dir = Path(output_path) / "run"
    ready_dir = Path(output_path) / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(run_dir.glob("**/*.pth"), key=os.path.getmtime)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in run/ after training")

    last_ckpt = checkpoints[-1]
    staged_ckpt = ready_dir / "unoptimize_model.pth"

    shutil.copy(last_ckpt, staged_ckpt)
    print(f"Staged checkpoint {last_ckpt} → {staged_ckpt}")



# Step 2.5 - Optimize Model
def optimize_model(out_path, clear_train_data="none"):
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"

    if clear_train_data in {"run", "all"} and run_dir.exists():
        shutil.rmtree(run_dir)
    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    model_path = ready_dir / "unoptimize_model.pth"
    if not model_path.is_file():
        raise FileNotFoundError("Unoptimized model not found in ready folder")

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    os.remove(model_path)
    optimized_model = ready_dir / "model.pth"
    torch.save(checkpoint, optimized_model)
    return str(optimized_model)


# Step 3 - Load Model
XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    XTTS_MODEL.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        vocab_path=xtts_vocab,
        speaker_file_path=xtts_speaker,
        use_deepspeed=False,
    )
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    return XTTS_MODEL


# Step 4 - Inference
def run_tts(lang, tts_text, speaker_audio_file,
            temperature=0.75, length_penalty=1.0, repetition_penalty=5.0,
            top_k=50, top_p=0.85, sentence_split=True, use_config=False):
    if XTTS_MODEL is None:
        raise RuntimeError("Model not loaded. Run load_model() first.")

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split,
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return out_path


# Step 5 - Get Optimized Model ZIP
def get_model_zip(out_path):
    ready_folder = os.path.join(out_path, "ready")
    if os.path.exists(ready_folder):
        zip_path = shutil.make_archive(
            os.path.join(tempfile.gettempdir(), "optimized_model"), 'zip', ready_folder
        )
        return zip_path
    return None
