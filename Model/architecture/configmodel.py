EMBED_DIM=768
NUMHEAD=8
VOCAB=10000
NUM_ENCODE=8
NUM_DECODE=8
D_FF=3072
MAXLEN=2048

EMBEDDING_DROPOUT=0.0
ENCODE_DROPOUT = [0.1 + i * 0.001 for i in range(NUM_ENCODE)]
DECODE_DROPOUT = [0.1 + i * 0.001 for i in range(NUM_DECODE)]

OUTPUT_PROJ_BIAS = True
ENCODE_BIAS = [False, False, False, False, True, True, True, True]
DECODE_BIAS = [False, False, False, False, True, True, True, True]