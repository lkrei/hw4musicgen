import argparse
import json
import logging
from pathlib import Path

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


logger = logging.getLogger("infer_prompts")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


PROMPTS = [
    {
        "description": "An epic and triumphant orchestral soundtrack featuring powerful brass and a sweeping string ensemble, driven by a fast march-like rhythm and an epic background choir, recorded with massive stadium reverb.",
        "general_mood": "Epic, heroic, triumphant, building tension",
        "genre_tags": ["Cinematic", "Orchestral", "Soundtrack"],
        "lead_instrument": "Powerful brass section (horns, trombones)",
        "accompaniment": "Sweeping string ensemble, heavy cinematic percussion, timpani",
        "tempo_and_rhythm": "Fast, driving, march-like rhythm",
        "vocal_presence": "Epic choir in the background (wordless chanting)",
        "production_quality": "High fidelity, wide stereo image, massive stadium reverb",
    },
    {
        "description": "A relaxing lo-fi hip-hop instrumental with a muffled electric piano playing jazz chords over a dusty vinyl crackle, deep sub-bass, and a slow boom-bap drum loop.",
        "general_mood": "Relaxing, nostalgic, chill, melancholic",
        "genre_tags": ["Lo-Fi Hip Hop", "Chillhop", "Instrumental"],
        "lead_instrument": "Muffled electric piano (Rhodes) playing jazz chords",
        "accompaniment": "Dusty vinyl crackle, deep sub-bass, soft boom-bap drum loop",
        "tempo_and_rhythm": "Slow, laid-back, swinging groove",
        "vocal_presence": "None",
        "production_quality": "Lo-Fi, vintage, warm tape saturation, slightly muffled high frequencies",
    },
    {
        "description": "An energetic progressive house dance track with a bright detuned synthesizer lead, pumping sidechain bass, and chopped vocal samples over a fast four-on-the-floor beat.",
        "general_mood": "Energetic, uplifting, party vibe, euphoric",
        "genre_tags": ["EDM", "Progressive House", "Dance"],
        "lead_instrument": "Bright, detuned synthesizer lead",
        "accompaniment": "Pumping sidechain bass, risers, crash cymbals",
        "tempo_and_rhythm": "Fast, driving, strict four-on-the-floor beat",
        "vocal_presence": "Chopped vocal samples used as a rhythmic instrument",
        "production_quality": "Modern, extremely loud, punchy, club-ready mix",
    },
    {
        "description": "An intimate acoustic folk instrumental featuring a fingerpicked acoustic guitar, light tambourine, and subtle upright bass, played in a gentle waltz-like rhythm.",
        "general_mood": "Intimate, warm, acoustic, peaceful",
        "genre_tags": ["Folk", "Acoustic", "Indie"],
        "lead_instrument": "Fingerpicked acoustic guitar",
        "accompaniment": "Light tambourine, subtle upright bass, distant ambient room sound",
        "tempo_and_rhythm": "Mid-tempo, gentle, waltz-like triple meter",
        "vocal_presence": "None",
        "production_quality": "Raw, organic, close-mic recording, natural room acoustics",
    },
    {
        "description": "A dark cyberpunk synthwave instrumental driven by an aggressive distorted analog bass synthesizer, arpeggiated synth plucks, and a retro 80s drum machine.",
        "general_mood": "Dark, futuristic, gritty, mysterious",
        "genre_tags": ["Synthwave", "Cyberpunk", "Darkwave"],
        "lead_instrument": "Aggressive, distorted analog bass synthesizer",
        "accompaniment": "Arpeggiated synth plucks, retro 80s drum machine (gated snare)",
        "tempo_and_rhythm": "Driving, mid-tempo, robotic precision",
        "vocal_presence": "None",
        "production_quality": "Retro-futuristic, heavy compression, synthetic, 80s aesthetic",
    },
]


def build_prompt_text(meta: dict) -> str:
    genre_str = ", ".join(meta.get("genre_tags", []))
    parts = [
        meta.get("description", ""),
        f"Mood: {meta.get('general_mood', '')}",
        f"Genres: {genre_str}",
        f"Lead instrument: {meta.get('lead_instrument', '')}",
        f"Accompaniment: {meta.get('accompaniment', '')}",
        f"Tempo and rhythm: {meta.get('tempo_and_rhythm', '')}",
        f"Vocal presence: {meta.get('vocal_presence', '')}",
        f"Production quality: {meta.get('production_quality', '')}",
    ]
    return ". ".join(p.rstrip(".") for p in parts if p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True
    )
    parser.add_argument(
        "--output_dir", type=str, required=True
    )
    parser.add_argument(
        "--duration", type=float, default=12.0
    )
    parser.add_argument(
        "--top_k", type=int, default=250
    )
    parser.add_argument(
        "--cfg_coef", type=float, default=3.0
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading model: %s (%s)", args.model_path, args.device)
    model = MusicGen.get_pretrained(args.model_path, device=args.device)
    model.set_generation_params(
        duration=args.duration,
        use_sampling=True,
        top_k=args.top_k,
        cfg_coef=args.cfg_coef,
    )

    texts = [build_prompt_text(p) for p in PROMPTS]

    for i, text in enumerate(texts, start=1):
        logger.info("prompt %d: %s", i, text[:120] + "..." if len(text) > 120 else text)

    logger.info("generating: %d", len(texts))
    wavs = model.generate(texts, progress=True)

    for idx, wav in enumerate(wavs, start=1):
        stem = str(output_dir / f"prompt_{idx}")
        saved_path = audio_write(
            stem,
            wav.cpu(),
            model.sample_rate,
            format="wav",
            strategy="loudness",
            loudness_compressor=True,
        )
        logger.info("saved: %s", saved_path)

    prompts_dump = output_dir / "prompts_hw4.json"
    with prompts_dump.open("w") as f:
        json.dump(PROMPTS, f, ensure_ascii=False, indent=2)
    logger.info("saved: %s", prompts_dump)


if __name__ == "__main__":
    main()
