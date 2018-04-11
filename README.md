# session4rec
GRu4Rec in Tensorflow

# Get data

- http://2015.recsyschallenge.com/challenge.html
- https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z

# Preprocess data

- change input and output directories in /scripts/preprocess.py
- python ./scripts/preprocess.py

# Results

## loss = top1, rnn-size = 100, epochs = 3
  - Recall@20: 0.4243443414791227
  - MRR@20: 0.10790868224055285

# References
  - Main source: https://github.com/Songweiping/GRU4Rec_TensorFlow
  - The original implementation: https://github.com/hidasib/GRU4Rec
