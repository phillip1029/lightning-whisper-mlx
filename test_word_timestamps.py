from lightning_whisper_mlx import LightningWhisperMLX

APPEND_PUNCTUATIONS = "\"'.。,，!！?？:：”)]}、"
PREPEND_PUNCTUATIONS = "\"'“¿([{-"


def main() -> None:
    whisper = LightningWhisperMLX(model="tiny", batch_size=12, quant=None)
    result = whisper.transcribe(
        audio_path="sample/Interview with Shanshan 20251216.mp3",
        word_timestamps=True,
        prepend_punctuations=PREPEND_PUNCTUATIONS,
        append_punctuations=APPEND_PUNCTUATIONS,
        verbose=False,
    )

    segments = result.get("segments", [])
    if not segments:
        raise AssertionError("No segments returned.")

    words = [w for s in segments for w in s.get("words", [])]
    if not words:
        raise AssertionError("No word timestamps produced.")

    punct_chars = set(PREPEND_PUNCTUATIONS + APPEND_PUNCTUATIONS)
    has_punctuation = any(
        any(ch in punct_chars for ch in w.get("word", "")) for w in words
    )
    if not has_punctuation:
        raise AssertionError("No punctuation merged into words.")

    for w in words:
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None or start > end:
            raise AssertionError(f"Bad word timing: {w!r}")

    print(f"OK: {len(words)} word timestamps with punctuation.")


if __name__ == "__main__":
    main()
