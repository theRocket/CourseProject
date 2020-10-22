# BERT and ERNIE (ryanjr3)

The aim of my project is to become better acquainted with a state-of-the-art NLP model called BERT or [Bidirectional Encoder Representations from Transformers](https://en.wikipedia.org/wiki/BERT_(language_model)) which has two pre-trained models for the English language (work sponsored by Google) coupled with state-of-the-art domain specific hardware for neural networks called TPUs (Tensor Processing Units) - including pods organized in 2-D toroidal mesh networks - available for general public use in Google Cloud infrastructure, optimized to run TensorFlow.

## Options:
1. [BERT](https://github.com/google-research/bert/) NLP model benchmarking on Google Cloud TPU.

I could try to replicate results from options "available in cloud" (only Google) on [MLPerf](https://mlperf.org/training-results-0-6/) results [example](https://github.com/mlperf/training_results_v0.6/blob/master/Google/systems/tpu-v3-2048.json).

2. If Option 1 is too expensive/too difficult/too easy, I could pivot or combine/compare with AWS offerings. I could also attempt to automate some infrastructure setup for making other CS410 students' code run on AWS Lambda or [AWS Deep Learning Container](https://aws.amazon.com/machine-learning/containers/) infrastructure.

Could I also run a NLP model like BERT on [PyTorch](https://github.com/pytorch/pytorch) using an AWS Deep Learning container?

Could I run [Fairseq](https://github.com/pytorch/fairseq)?
