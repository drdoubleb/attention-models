"""Entry point: train the tiny model and render the animation videos.

    python main.py all                 # train + render everything
    python main.py train               # just train (saves out/run_data.pkl)
    python main.py animate             # render all videos from saved run
    python main.py animate attention   # render a single video

Tweak texts, prompt, model size, and video settings in config.py.
"""

import sys

import config as cfg
from attention_model.training import train, load_run
from animations import (anim_tokenization, anim_embeddings, anim_attention,
                        anim_training, anim_generation)

RENDERERS = {
    "tokenization": anim_tokenization.render,
    "embeddings": anim_embeddings.render,
    "attention": anim_attention.render,
    "training": anim_training.render,
    "generation": anim_generation.render,
}


def animate(run, only=None):
    names = [only] if only else list(RENDERERS)
    for name in names:
        if name not in RENDERERS:
            sys.exit(f"unknown video '{name}' — choose from: {', '.join(RENDERERS)}")
        print(f"rendering {name} ...")
        RENDERERS[name](run, cfg)


def main(argv):
    command = argv[1] if len(argv) > 1 else "all"
    if command == "train":
        train(cfg)
    elif command == "animate":
        animate(load_run(cfg), only=argv[2] if len(argv) > 2 else None)
    elif command == "all":
        run = train(cfg)
        animate(run)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
