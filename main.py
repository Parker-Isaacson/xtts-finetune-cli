import os
import shutil
import torch
import tempfile
import torchaudio
from pathlib import Path
import argparse

# Import functions from utils
from utils.xtts_header import (
    preprocess_dataset,
    train_model,
    optimize_model,
    load_model,
    run_tts,
    get_model_zip,
)


def main():
    parser = argparse.ArgumentParser(
        description="XTTS Training + Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_audio", help="Path to input audio file (WAV)")
    parser.add_argument(
        "--model_dir",
        default="finetune_models",
        help="Directory to store model checkpoints",
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Directory to store final outputs",
    )
    parser.add_argument(
        "--text",
        default="Tom sat back down, this time not under the tree but closer to the edge of the lake. "
                "He looked across the water, then up at the sky. One star blinked brighter than the rest. "
                "It blinked red. Tom took out the map again. Now it was blank. The star was gone. "
                "The path was gone. Just white space.",
        help="Text to synthesize with the trained model",
    )
    parser.add_argument(
        "--help_only",
        action="store_true",
        help="Print help message and exit",
    )
    args = parser.parse_args()

    if args.help_only:
        parser.print_help()
        return

    # Convert to absolute paths
    input_audio = str(Path(args.input_audio).resolve())
    model_dir = str(Path(args.model_dir).resolve())
    out_dir = str(Path(args.out_dir).resolve())

    # Reset model dir each run
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    print("Step 1 - Creating dataset...")
    train_csv, eval_csv = preprocess_dataset([input_audio], "", "en", "small", model_dir)

    print("Step 2 - Training...")
    train_model(
        "",
        "v2.0.2",
        "en",
        train_csv,
        eval_csv,
        num_epochs=6,
        batch_size=2,
        grad_acumm=1,
        output_path=model_dir,
        max_audio_length=11,
    )

    print("Step 2.5 - Optimizing model...")
    optimized_path = optimize_model(model_dir)
    print("Optimized model at:", optimized_path)

    print("Step 3 - Loading model...")
    model = load_model(
        f"{model_dir}/ready/model.pth",
        f"{model_dir}/ready/config.json",
        f"{model_dir}/ready/vocab.json",
        f"{model_dir}/ready/speakers_xtts.pth",
    )

    print("Step 4 - Running inference...")
    output_audio = run_tts(
        "en",
        args.text,
        f"{model_dir}/ready/reference.wav",
    )
    print("Generated speech at:", output_audio)

    print("Step 5 - Zipping optimized model...")
    zip_file = get_model_zip(model_dir)
    print("Optimized model zip:", zip_file)

    # Store results
    results_dir = Path(out_dir).resolve()
    results_dir.mkdir(exist_ok=True)

    shutil.move(output_audio, results_dir / "audio.wav")
    shutil.move(zip_file, results_dir / "model.zip")
    print(f"Results stored in {results_dir}")


if __name__ == "__main__":
    main()
