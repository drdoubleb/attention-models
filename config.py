"""Central configuration for the attention-model animation project.

Everything a user is expected to tweak lives here: which text to train on,
what prompt to generate from, how big the model is, and how the videos are
rendered.  All sizes are deliberately tiny so that every matrix fits on
screen in the animations.
"""

# ---------------------------------------------------------------- data ----
# Plain-text file the tokenizer and model are trained on.  Replace the
# contents of this file (or point this path elsewhere) to use your own text.
TRAINING_TEXT_PATH = "data/training_text.txt"

# Starter text handed to the trained model for the generation video.
PROMPT = "the cat"

# Number of tokens to generate after the prompt.
GENERATE_TOKENS = 24

# ----------------------------------------------------------- tokenizer ----
# How many byte-pair merges the mini-BPE tokenizer learns.  The final
# vocabulary is (distinct characters in the text) + NUM_MERGES.
NUM_MERGES = 40

# --------------------------------------------------------------- model ----
D_MODEL = 16       # width of every token vector
CONTEXT = 16       # maximum sequence length the model can see
MLP_HIDDEN = 32    # hidden width of the feed-forward layer

# ------------------------------------------------------------ training ----
TRAIN_STEPS = 3000
BATCH_SIZE = 8
LEARNING_RATE = 3e-3
SEED = 42

# How often (in steps) to record a snapshot for the training animation.
SNAPSHOT_EVERY = 30

# --------------------------------------------------------- generation ----
# Sampling temperature: 0 -> greedy argmax, 1 -> sample from the model's
# distribution unchanged.  Low values keep the tiny model coherent.
TEMPERATURE = 0.6

# ---------------------------------------------------------- animation ----
OUTPUT_DIR = "out"           # run data + rendered videos land here
VIDEO_FPS = 2                # global frames-per-second for the videos
TRAINING_VIDEO_FPS = 8       # the training video has ~100 quick frames
VIDEO_DPI = 110              # resolution of the rendered frames

# Sentence used to illustrate tokenization / embeddings / attention.
# Keep it short (its token count must fit in CONTEXT).  If None, the first
# sentence of the training text is used.
PROBE_TEXT = None
