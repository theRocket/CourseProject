```python
import sagemaker
from sagemaker.s3 import S3Downloader
from sagemaker.s3 import S3Uploader

print('Sagemaker version: '+sagemaker.__version__)

from sagemaker import get_execution_role
from sagemaker.session import Session

# S3 bucket for saving code and model artifacts.
# Feel free to specify a different bucket here if you wish.
bucket = 'sagemaker-cs410-finalproj'
filename = 'yelp_review_polarity'
# bucket = Session().default_bucket()

print('Bucket location: '+bucket)

# Bucket location where your custom code will be saved in the tar.gz format.
# custom_code_upload_location = 's3://{}/code'.format(bucket)

# Bucket location where results of model training are saved.
# model_artifacts_location = 's3://{}/artifacts'.format(bucket)

# IAM execution role that gives SageMaker access to resources in your AWS account.
# We can use the SageMaker Python SDK to get the role from our notebook environment. 
role = get_execution_role()

print('Execution role: '+role)

train_data_location = 's3://{}/{}.train'.format(bucket,filename)
output_data_location = 's3://{}/{}.gluon'.format(bucket,filename)
test_data_location = 's3://{}/{}.test'.format(bucket,filename)

print('Training data location: '+train_data_location)
print('Test data location: '+test_data_location)
print('Output data location: '+output_data_location)

test_data_smaller = 's3://{}/{}/yelp_review_polarity.test'.format(bucket,'test')
print('Testing reader on smaller data file: '+test_data_smaller)

input_file = S3Downloader.read_file(test_data_smaller)
input_text = input_file.split('\n')
data = []
labels = []
for line in input_text:
    tokens = line.split(',', 1)
    labels.append(tokens[0].strip())
    data.append(tokens[1].strip())
print('Parsed '+str(len(labels))+' test data labels')
```

    Sagemaker version: 2.16.4.dev0
    Bucket location: sagemaker-cs410-finalproj
    Execution role: arn:aws:iam::876612415673:role/service-role/AmazonSageMaker-ExecutionRole-20201210T175209
    Training data location: s3://sagemaker-cs410-finalproj/yelp_review_polarity.train
    Test data location: s3://sagemaker-cs410-finalproj/yelp_review_polarity.test
    Output data location: s3://sagemaker-cs410-finalproj/yelp_review_polarity.gluon
    Testing reader on smaller data file: s3://sagemaker-cs410-finalproj/test/yelp_review_polarity.test
    Parsed 33 test data labels



```python
!pip install gluonnlp
```

    Collecting gluonnlp
      Downloading gluonnlp-0.10.0.tar.gz (344 kB)
    [K     |████████████████████████████████| 344 kB 6.7 MB/s eta 0:00:01
    [?25hRequirement already satisfied: numpy>=1.16.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gluonnlp) (1.18.1)
    Requirement already satisfied: cython in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gluonnlp) (0.29.15)
    Requirement already satisfied: packaging in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gluonnlp) (20.1)
    Requirement already satisfied: six in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from packaging->gluonnlp) (1.14.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from packaging->gluonnlp) (2.4.6)
    Building wheels for collected packages: gluonnlp
      Building wheel for gluonnlp (setup.py) ... [?25ldone
    [?25h  Created wheel for gluonnlp: filename=gluonnlp-0.10.0-cp36-cp36m-linux_x86_64.whl size=468341 sha256=127373874f936ebab2fc1865bc6d6bd8672aa16607719e5b35cb0b344370c0f3
      Stored in directory: /home/ec2-user/.cache/pip/wheels/62/62/9a/53be069ac8c9dde533dacce0e716193a4a43e87b5d37f5008c
    Successfully built gluonnlp
    Installing collected packages: gluonnlp
    Successfully installed gluonnlp-0.10.0
    [33mWARNING: You are using pip version 20.0.2; however, version 20.3.1 is available.
    You should consider upgrading via the '/home/ec2-user/anaconda3/envs/mxnet_p36/bin/python -m pip install --upgrade pip' command.[0m



```python
!pip install smart_open
```

    Collecting smart_open
      Downloading smart_open-4.0.1.tar.gz (117 kB)
    [K     |████████████████████████████████| 117 kB 6.4 MB/s eta 0:00:01
    [?25hBuilding wheels for collected packages: smart-open
      Building wheel for smart-open (setup.py) ... [?25ldone
    [?25h  Created wheel for smart-open: filename=smart_open-4.0.1-py3-none-any.whl size=108249 sha256=4ba1ef2cc0894193823763f651a17f232b4148e9afad49c090fcc5052b3962bd
      Stored in directory: /home/ec2-user/.cache/pip/wheels/8b/5e/70/42adcaea93c80417ec6accf7db1d6d02367ed02f2254cd5eef
    Successfully built smart-open
    Installing collected packages: smart-open
    Successfully installed smart-open-4.0.1
    [33mWARNING: You are using pip version 20.0.2; however, version 20.3.1 is available.
    You should consider upgrading via the '/home/ec2-user/anaconda3/envs/mxnet_p36/bin/python -m pip install --upgrade pip' command.[0m



```python
%run fasttext_word_ngram.py --input $train_data_location \
                            --output $output_data_location \
                            --validation $test_data_location \
                            --ngrams 1 --epochs 10 --lr 0.1 --emsize 100 --gpu=0
```

    INFO:root:Ngrams range for the training run : 1
    INFO:root:Loading Training data
    INFO:root:Opening file s3://sagemaker-cs410-finalproj/yelp_review_polarity.train for reading input
    INFO:root:Loading Test data
    INFO:root:Opening file s3://sagemaker-cs410-finalproj/yelp_review_polarity.test for reading input
    INFO:root:Vocabulary size: 464402
    INFO:root:Training data converting to sequences...
    INFO:root:Done! Sequence conversion Time=34.45s, #Sentences=560000
    INFO:root:Done! Sequence conversion Time=12.03s, #Sentences=38000
    INFO:root:Encoding labels
    INFO:root:Label mapping:{'__label__1': 0, '__label__2': 1}
    INFO:root:Done! Preprocessing Time=11.63s, #Sentences=560000
    INFO:root:Done! Preprocessing Time=3.14s, #Sentences=38000
    INFO:root:Number of labels: 2
    INFO:root:Initializing network
    INFO:root:Running Training on ctx:gpu(0)
    INFO:root:Embedding Matrix Length:464402
    INFO:root:Number of output units in the last layer :1
    INFO:root:Network initialized
    INFO:root:Changing the loss function to sigmoid since its Binary Classification
    INFO:root:Loss function for training:SigmoidBinaryCrossEntropyLoss(batch_axis=0, w=None)
    INFO:root:Starting Training!
    INFO:root:Training on 560000 samples and testing on 38000 samples
    INFO:root:Number of batches for each epoch : 35000.0, Display cadence: 3500
    INFO:root:Epoch : 0, Batches complete :0
    INFO:root:Epoch : 0, Batches complete :3500
    INFO:root:Epoch : 0, Batches complete :7000
    INFO:root:Epoch : 0, Batches complete :10500
    INFO:root:Epoch : 0, Batches complete :14000
    INFO:root:Epoch : 0, Batches complete :17500
    INFO:root:Epoch : 0, Batches complete :21000
    INFO:root:Epoch complete :0, Computing Accuracy
    INFO:root:Epochs completed : 0 Test Accuracy: 0.9224210526315789, Test Loss: 0.37670645791753526
    INFO:root:Epoch : 1, Batches complete :0
    INFO:root:Epoch : 1, Batches complete :3500
    INFO:root:Epoch : 1, Batches complete :7000
    INFO:root:Epoch : 1, Batches complete :10500
    INFO:root:Epoch : 1, Batches complete :14000
    INFO:root:Epoch : 1, Batches complete :17500
    INFO:root:Epoch : 1, Batches complete :21000
    INFO:root:Epoch complete :1, Computing Accuracy
    INFO:root:Epochs completed : 1 Test Accuracy: 0.9303684210526316, Test Loss: 0.24866203612015617
    INFO:root:Epoch : 2, Batches complete :0
    INFO:root:Epoch : 2, Batches complete :3500
    INFO:root:Epoch : 2, Batches complete :7000
    INFO:root:Epoch : 2, Batches complete :10500
    INFO:root:Epoch : 2, Batches complete :14000
    INFO:root:Epoch : 2, Batches complete :17500
    INFO:root:Epoch : 2, Batches complete :21000
    INFO:root:Epoch complete :2, Computing Accuracy
    INFO:root:Epochs completed : 2 Test Accuracy: 0.935921052631579, Test Loss: 0.2255847477009397
    INFO:root:Epoch : 3, Batches complete :0
    INFO:root:Epoch : 3, Batches complete :3500
    INFO:root:Epoch : 3, Batches complete :7000
    INFO:root:Epoch : 3, Batches complete :10500
    INFO:root:Epoch : 3, Batches complete :14000
    INFO:root:Epoch : 3, Batches complete :17500
    INFO:root:Epoch : 3, Batches complete :21000
    INFO:root:Epoch complete :3, Computing Accuracy
    INFO:root:Epochs completed : 3 Test Accuracy: 0.9370526315789474, Test Loss: 0.19228459512859025
    INFO:root:Epoch : 4, Batches complete :0
    INFO:root:Epoch : 4, Batches complete :3500
    INFO:root:Epoch : 4, Batches complete :7000
    INFO:root:Epoch : 4, Batches complete :10500
    INFO:root:Epoch : 4, Batches complete :14000
    INFO:root:Epoch : 4, Batches complete :17500
    INFO:root:Epoch : 4, Batches complete :21000
    INFO:root:Epoch complete :4, Computing Accuracy
    INFO:root:Epochs completed : 4 Test Accuracy: 0.9399736842105263, Test Loss: 0.18041272283296456
    INFO:root:Epoch : 5, Batches complete :0
    INFO:root:Epoch : 5, Batches complete :3500
    INFO:root:Epoch : 5, Batches complete :7000
    INFO:root:Epoch : 5, Batches complete :10500
    INFO:root:Epoch : 5, Batches complete :14000
    INFO:root:Epoch : 5, Batches complete :17500
    INFO:root:Epoch : 5, Batches complete :21000
    INFO:root:Epoch complete :5, Computing Accuracy
    INFO:root:Epochs completed : 5 Test Accuracy: 0.9398947368421052, Test Loss: 0.17765162684229294
    INFO:root:Epoch : 6, Batches complete :0
    INFO:root:Epoch : 6, Batches complete :3500
    INFO:root:Epoch : 6, Batches complete :7000
    INFO:root:Epoch : 6, Batches complete :10500
    INFO:root:Epoch : 6, Batches complete :14000
    INFO:root:Epoch : 6, Batches complete :17500
    INFO:root:Epoch : 6, Batches complete :21000
    INFO:root:Epoch complete :6, Computing Accuracy
    INFO:root:Epochs completed : 6 Test Accuracy: 0.9399473684210526, Test Loss: 0.17690470782618312
    INFO:root:Epoch : 7, Batches complete :0
    INFO:root:Epoch : 7, Batches complete :3500
    INFO:root:Epoch : 7, Batches complete :7000
    INFO:root:Epoch : 7, Batches complete :10500
    INFO:root:Epoch : 7, Batches complete :14000
    INFO:root:Epoch : 7, Batches complete :17500
    INFO:root:Epoch : 7, Batches complete :21000
    INFO:root:Epoch complete :7, Computing Accuracy
    INFO:root:Epochs completed : 7 Test Accuracy: 0.9401578947368421, Test Loss: 0.17890373866995307
    INFO:root:Epoch : 8, Batches complete :0
    INFO:root:Epoch : 8, Batches complete :3500
    INFO:root:Epoch : 8, Batches complete :7000
    INFO:root:Epoch : 8, Batches complete :10500
    INFO:root:Epoch : 8, Batches complete :14000
    INFO:root:Epoch : 8, Batches complete :17500
    INFO:root:Epoch : 8, Batches complete :21000
    INFO:root:Epoch complete :8, Computing Accuracy
    INFO:root:Epochs completed : 8 Test Accuracy: 0.9397105263157894, Test Loss: 0.17806511536407177
    INFO:root:Epoch : 9, Batches complete :0
    INFO:root:Epoch : 9, Batches complete :3500
    INFO:root:Epoch : 9, Batches complete :7000
    INFO:root:Epoch : 9, Batches complete :10500
    INFO:root:Epoch : 9, Batches complete :14000
    INFO:root:Epoch : 9, Batches complete :17500
    INFO:root:Epoch : 9, Batches complete :21000
    INFO:root:Epoch complete :9, Computing Accuracy
    INFO:root:Epochs completed : 9 Test Accuracy: 0.939921052631579, Test Loss: 0.17803387705344548



```python

```
