EMBED_DIM=512
NUMHEAD=8
VOCAB=10000
NUM_ENCODE=1
NUM_DECODE=1
D_FF=1024
MAXLEN=2048

EMBEDDING_DROPOUT=0.0
ENCODE_DROPOUT = [0.1 + i * 0.001 for i in range(NUM_ENCODE)]
DECODE_DROPOUT = [0.1 + i * 0.001 for i in range(NUM_DECODE)]

OUTPUT_PROJ_BIAS = False
ENCODE_BIAS = [False, False, False, False, False, False, False, False]
DECODE_BIAS = [False, False, False, False, False, False]