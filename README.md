# Project BERT and ERNIE (ryanjr3)
## CS 410 - Text Information Systems
#### Masters in Computer Science at UIUC

The aim of my project is to become better acquainted with a state-of-the-art NLP model called BERT or [Bidirectional Encoder Representations from Transformers](https://en.wikipedia.org/wiki/BERT_(language_model)) which has two pre-trained models for the English language (work sponsored by Google).

I want to learn this model by test driving it on state-of-the-art domain specific hardware for neural networks called TPUs (Tensor Processing Units) - including pods organized in 2-D toroidal mesh networks - also by Google and available for general public use in their Cloud infrastructure.

This infrastructure is optimized to run [TensorFlow](https://github.com/tensorflow/tensorflow) - "An Open Source Machine Learning Framework for Everyone") - developed by the [Google Brain Team](https://en.wikipedia.org/wiki/Google_Brain#Google_Translate).

## Options:
1. [BERT](https://github.com/google-research/bert/) NLP model benchmarking on [Google Cloud TPU](https://cloud.google.com/tpu).

I could try to replicate results from options "available in cloud" (only Google) on [MLPerf](https://mlperf.org/training-results-0-6/) results [example](https://github.com/mlperf/training_results_v0.6/blob/master/Google/systems/tpu-v3-2048.json).

2. If Option 1 is too expensive/too difficult/too easy, I could pivot or combine/compare with AWS offerings. I could also attempt to automate some infrastructure setup for making other CS410 students' code run on [AWS Lambda using Python layers](https://towardsdatascience.com/introduction-to-amazon-lambda-layers-and-boto3-using-python3-39bd390add17) or [AWS Deep Learning Container](https://aws.amazon.com/machine-learning/containers/) infrastructure.


Could I also run a NLP model like BERT on [PyTorch](https://github.com/pytorch/pytorch) using an AWS Deep Learning container?

Could I run [Fairseq](https://github.com/pytorch/fairseq) on either architecture? "Facebook is releasing the data set, model, training and evaluation setups as open source to the research community to help spur on further advancements." (source: [Engadget](https://www.engadget.com/facebooks-ai-can-translate-languages-directly-into-one-another-150029679.html))
