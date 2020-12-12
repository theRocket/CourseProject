# Training FastText "Bag of Tricks" using Gluon & MXNet on AWS GPUs (ryanjr3)
## CS 410 - Text Information Systems (MS-CS at UIUC)
#### Gluon for NLP and MXNet

The aim of my project is to become better acquainted with the [Gluon API for Natural Language Processing](nlp.gluon.ai) (NLP).

> GluonNLP provides implementations of the state-of-the-art (SOTA) deep learning models in NLP, and build blocks for text data pipelines and models. It is designed for engineers, researchers, and students to fast prototype research ideas and products based on these models.

For example, it can easily provide the cosine similarity of two word vectors with the following simple python function, `cos_similarity()`:

```
import mxnet as mx
import gluonnlp as nlp

glove = nlp.embedding.create('glove', source='glove.6B.50d')

def cos_similarity(embedding, word1, word2):
    vec1, vec2 = embedding[word1], embedding[word2]
    return mx.nd.dot(vec1, vec2) / (vec1.norm() * vec2.norm())

print('Similarity between "baby" and "infant": ', cos_similarity(glove, 'baby', 'infant').asnumpy()[0])
```

The Gluon API provides user-friendly access to the Apache MXNet library for Deep Learning, which advertises itself as being a "truly open source deep learning framework suited for flexible research prototyping and production." A MXNet enabled runtime on Python 3.6 on Amazon Web Services (AWS) Sagemaker instances is also well-supported with `conda_mxnet_p36` selected for the Jupyter Notebook. Gluon NLP dependencies are easily added to the notebook by running a cell with:
```
!pip install gluonnlp
```

#### AWS Sagemaker DL on NVIDIA GPUs
I plan to implement the training job using high-end [P3 AWS Sagemaker instances](https://aws.amazon.com/ec2/instance-types/p3/) to benchmark rapid training of models using python v3.6. According to the table of instance sizes listed at the bottom of the above linked page, the cheapest instance offered - `p3.2xlarge` - provides 1 Tesla V100 GPU with 16GB of GPU memory for $3.07/hr on demand.

We have [published benchmarks of NVIDIA GPUs](https://www.microway.com/knowledge-center-articles/comparison-of-nvidia-geforce-gpus-and-nvidia-tesla-gpus/) provided in TensorFLOPs, which are units of floating-point arithmetic performance aimed at NVIDIA GPU hardware called Tensor Cores:

>A new, specialized Tensor Core unit was introduced with “Volta” generation GPUs. It combines a multiply of two FP16 units (into a full precision product) with a FP32 accumulate operation—the exact operations used in Deep Learning Training computation. NVIDIA is now measuring GPUs with Tensor Cores by a new deep learning performance metric: a new unit called TensorTFLOPS.

According to that metric, the Tesla V100 GPU rates around 112-125 TensorTFLOPS (exact figure depending on the use of PCI-Express or SXM2 SKU interfaces). For comparison, the maximum known deep learning performance at *any* precision of the Tesla K80 is 5.6 TFLOPS for FP32. This GPU is provided on the [P2 Sagemaker instances](https://aws.amazon.com/ec2/instance-types/p2/), and for 1 GPU on the `p2.xlarge` instance size, the cost is $0.90/hr. If we can attain a 20x performance increase on our training job for approx. 4x compute resource cost, that seems like a great win!

I also compare running this training job on my Macbook Pro CPU - a 2.6 GHz 6-Core Intel Core i7. Since the included AMD Radeon Pro 5300M graphics card does not implement CUDA architecture, we must deleted the following option when running the training job: `--gpu=0`. Otherwise, this signals the index number of the GPU to use. Since we selected AWS instances with only one GPU, this flag will always be zero.

#### FastText "Bag of Tricks" and Yelp Sentiment Classification

The primary influence for this project was an entry hosted at nlp.gluon.ai for [Text Classification](https://nlp.gluon.ai/model_zoo/text_classification/index.html) called **Fast-text Word N-gram**. It leverages the [fastText python library](https://fasttext.cc/) used for "efficient text classification and representation learning" and [developed at Facebook research](https://github.com/facebookresearch/fastText). The paper was also published by the Facebook AI Research team in 2016 and is called [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759) (the full PDF is [included in this repo](BagofTricks_1607.01759.pdf)). They claim in the abstract:

>Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation. 

In this project, I begin by training the model on AWS using the Yelp Sentiment (binary classification) data to establish a workflow. Once the architecture is in place and proven to achieve timely results, we can expand into the other datasets. Each are manually uploaded to S3 buckets to make them accessible to our Sagemaker instance, rather than using the script provided by fastText (although we do use their [text normalization function](./text_classification/data_fetch.sh)).
#### Output from AWS Sagemaker:
See the [Jupyter notebook](main.md)

#### Other Resources:

[Deploying custom models built with Gluon and Apache MXNet on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/deploying-custom-models-built-with-gluon-and-apache-mxnet-on-amazon-sagemaker/)

[Use MXNet with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/mxnet/using_mxnet.html)

[SageMaker MXNet Inference Toolkit](https://github.com/aws/sagemaker-mxnet-inference-toolkit)

Another possible compute resource are the [AWS Deep Learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md). Amazon claims that:

> AWS DL Containers include AWSoptimizations and improvements to the latest versions of popular frameworks, like TensorFlow, PyTorch, and Apache MXNet, and libraries to deliver the highest performance for training and inference in the cloud. For example, AWS TensorFlow optimizations allow models to train up to twice as fast through significantly improved GPU scaling.



