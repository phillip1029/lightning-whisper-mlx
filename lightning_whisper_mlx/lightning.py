from .transcribe import transcribe_audio
from huggingface_hub import hf_hub_download

models = {
    "tiny": {
        "base": "mlx-community/whisper-tiny",
        "4bit": "mlx-community/whisper-tiny-mlx-4bit",
        "8bit": "mlx-community/whisper-tiny-mlx-8bit"
    },
    "small": {
        "base": "mlx-community/whisper-small-mlx",
        "4bit": "mlx-community/whisper-small-mlx-4bit",
        "8bit": "mlx-community/whisper-small-mlx-8bit"
    },
    "distil-small.en": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "base": {
        "base": "mlx-community/whisper-base-mlx",
        "4bit": "mlx-community/whisper-base-mlx-4bit",
        "8bit": "mlx-community/whisper-base-mlx-8bit"
    },
    "medium": {
        "base": "mlx-community/whisper-medium-mlx",
        "4bit": "mlx-community/whisper-medium-mlx-4bit",
        "8bit": "mlx-community/whisper-medium-mlx-8bit"
    },
    "distil-medium.en": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "large": {
        "base": "mlx-community/whisper-large-mlx",
        "4bit": "mlx-community/whisper-large-mlx-4bit",
        "8bit": "mlx-community/whisper-large-mlx-8bit",
    },
    "large-v2": {
        "base": "mlx-community/whisper-large-v2-mlx",
        "4bit": "mlx-community/whisper-large-v2-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v2-mlx-8bit",
    },
    "distil-large-v2": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "large-v3": {
        "base": "mlx-community/whisper-large-v3-mlx",
        "4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v3-mlx-8bit",
    },
    "Belle-whisper-large-v3-zh-mlx": {
        "base": "ppeng08/Belle-whisper-large-v3-zh-mlx",
    },
}

class LightningWhisperMLX():
    def __init__(self, model, batch_size=12, quant=None):
        if quant and (quant != "4bit" and quant != "8bit"):
            raise ValueError("Quantization must be `4bit` or `8bit`")

        if model not in models:
            raise ValueError("Please select a valid model")

        self.name = model
        self.batch_size = batch_size

        # resolve repo_id
        if quant and "distil" not in model:
            repo_id = models[model][quant]
        else:
            repo_id = models[model]["base"]

        # preserve your distil naming behavior
        if quant and "distil" in model:
            if quant == "4bit":
                self.name += "-4-bit"
            else:
                self.name += "-8-bit"

        if "distil" in model:
            # keep original distil behavior
            filename1 = f"./mlx_models/{self.name}/weights.npz"
            filename2 = f"./mlx_models/{self.name}/config.json"
            local_dir = "./"

            hf_hub_download(repo_id=repo_id, filename=filename1, local_dir=local_dir)
            hf_hub_download(repo_id=repo_id, filename=filename2, local_dir=local_dir)

        else:
            # NEW: support safetensors first, then npz
            local_dir = f"./mlx_models/{self.name}"
            filename2 = "config.json"

            weight_candidates = ["weights.safetensors", "model.safetensors", "weights.npz"]
            last_err = None
            for filename1 in weight_candidates:
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename1, local_dir=local_dir)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e

            if last_err is not None:
                raise FileNotFoundError(
                    f"Could not download any weights file from {repo_id}. "
                    f"Tried: {weight_candidates}. Last error: {last_err}"
                )

            # Download config (must exist)
            hf_hub_download(repo_id=repo_id, filename=filename2, local_dir=local_dir)

    def transcribe(self, audio_path, language=None, **kwargs):
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size
        if "path_or_hf_repo" not in kwargs:
            kwargs["path_or_hf_repo"] = f"./mlx_models/{self.name}"
        result = transcribe_audio(audio_path, language=language, **kwargs)
        return result
 