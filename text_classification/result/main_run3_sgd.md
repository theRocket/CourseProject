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
    [K     |████████████████████████████████| 344 kB 6.2 MB/s eta 0:00:01
    [?25hRequirement already satisfied: numpy>=1.16.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gluonnlp) (1.18.1)
    Requirement already satisfied: cython in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gluonnlp) (0.29.15)
    Requirement already satisfied: packaging in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gluonnlp) (20.1)
    Requirement already satisfied: six in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from packaging->gluonnlp) (1.14.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from packaging->gluonnlp) (2.4.6)
    Building wheels for collected packages: gluonnlp
      Building wheel for gluonnlp (setup.py) ... [?25ldone
    [?25h  Created wheel for gluonnlp: filename=gluonnlp-0.10.0-cp36-cp36m-linux_x86_64.whl size=468341 sha256=fb8af7feecb090b9134d0b99142aeb4fbec426ddccdf2db36fda9b3d8781e98f
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
    [K     |████████████████████████████████| 117 kB 6.7 MB/s eta 0:00:01
    [?25hBuilding wheels for collected packages: smart-open
      Building wheel for smart-open (setup.py) ... [?25ldone
    [?25h  Created wheel for smart-open: filename=smart_open-4.0.1-py3-none-any.whl size=108249 sha256=4e2b18b97a29b09329198d17a24ada7aeb45a365e8491388e2e0e6bc735eba44
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
                            --ngrams 1 --epochs 25 --lr 0.1 --emsize 100 --optimizer 'sgd' --gpu=0
```

    INFO:root:Ngrams range for the training run : 1
    INFO:root:Loading Training data
    INFO:root:Opening file s3://sagemaker-cs410-finalproj/yelp_review_polarity.train for reading input
    INFO:root:Loading Test data
    INFO:root:Opening file s3://sagemaker-cs410-finalproj/yelp_review_polarity.test for reading input
    INFO:root:Vocabulary size: 464402
    INFO:root:Training data converting to sequences...
    INFO:root:Done! Sequence conversion Time=26.87s, #Sentences=560000
    INFO:root:Done! Sequence conversion Time=11.85s, #Sentences=38000
    INFO:root:Encoding labels
    INFO:root:Label mapping:{'__label__1': 0, '__label__2': 1}
    INFO:root:Done! Preprocessing Time=14.27s, #Sentences=560000
    INFO:root:Done! Preprocessing Time=1.64s, #Sentences=38000
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
    INFO:root:Epochs completed : 0 Test Accuracy: 0.9048684210526315, Test Loss: 0.40878741784467926
    INFO:root:Epoch : 1, Batches complete :0
    INFO:root:Epoch : 1, Batches complete :3500
    INFO:root:Epoch : 1, Batches complete :7000
    INFO:root:Epoch : 1, Batches complete :10500
    INFO:root:Epoch : 1, Batches complete :14000
    INFO:root:Epoch : 1, Batches complete :17500
    INFO:root:Epoch : 1, Batches complete :21000
    INFO:root:Epoch complete :1, Computing Accuracy
    INFO:root:Epochs completed : 1 Test Accuracy: 0.9265, Test Loss: 0.24937551502025637
    INFO:root:Epoch : 2, Batches complete :0
    INFO:root:Epoch : 2, Batches complete :3500
    INFO:root:Epoch : 2, Batches complete :7000
    INFO:root:Epoch : 2, Batches complete :10500
    INFO:root:Epoch : 2, Batches complete :14000
    INFO:root:Epoch : 2, Batches complete :17500
    INFO:root:Epoch : 2, Batches complete :21000
    INFO:root:Epoch complete :2, Computing Accuracy
    INFO:root:Epochs completed : 2 Test Accuracy: 0.9335, Test Loss: 0.20890068642353887
    INFO:root:Epoch : 3, Batches complete :0
    INFO:root:Epoch : 3, Batches complete :3500
    INFO:root:Epoch : 3, Batches complete :7000
    INFO:root:Epoch : 3, Batches complete :10500
    INFO:root:Epoch : 3, Batches complete :14000
    INFO:root:Epoch : 3, Batches complete :17500
    INFO:root:Epoch : 3, Batches complete :21000
    INFO:root:Epoch complete :3, Computing Accuracy
    INFO:root:Epochs completed : 3 Test Accuracy: 0.9370526315789474, Test Loss: 0.18883446813855115
    INFO:root:Epoch : 4, Batches complete :0
    INFO:root:Epoch : 4, Batches complete :3500
    INFO:root:Epoch : 4, Batches complete :7000
    INFO:root:Epoch : 4, Batches complete :10500
    INFO:root:Epoch : 4, Batches complete :14000
    INFO:root:Epoch : 4, Batches complete :17500
    INFO:root:Epoch : 4, Batches complete :21000
    INFO:root:Epoch complete :4, Computing Accuracy
    INFO:root:Epochs completed : 4 Test Accuracy: 0.9402894736842106, Test Loss: 0.1758434079364924
    INFO:root:Epoch : 5, Batches complete :0
    INFO:root:Epoch : 5, Batches complete :3500
    INFO:root:Epoch : 5, Batches complete :7000
    INFO:root:Epoch : 5, Batches complete :10500
    INFO:root:Epoch : 5, Batches complete :14000
    INFO:root:Epoch : 5, Batches complete :17500
    INFO:root:Epoch : 5, Batches complete :21000
    INFO:root:Epoch complete :5, Computing Accuracy
    INFO:root:Epochs completed : 5 Test Accuracy: 0.9397105263157894, Test Loss: 0.17382905319323275
    INFO:root:Epoch : 6, Batches complete :0
    INFO:root:Epoch : 6, Batches complete :3500
    INFO:root:Epoch : 6, Batches complete :7000
    INFO:root:Epoch : 6, Batches complete :10500
    INFO:root:Epoch : 6, Batches complete :14000
    INFO:root:Epoch : 6, Batches complete :17500
    INFO:root:Epoch : 6, Batches complete :21000
    INFO:root:Epoch complete :6, Computing Accuracy
    INFO:root:Epochs completed : 6 Test Accuracy: 0.9395526315789474, Test Loss: 0.18225679093707228
    INFO:root:Epoch : 7, Batches complete :0
    INFO:root:Epoch : 7, Batches complete :3500
    INFO:root:Epoch : 7, Batches complete :7000
    INFO:root:Epoch : 7, Batches complete :10500
    INFO:root:Epoch : 7, Batches complete :14000
    INFO:root:Epoch : 7, Batches complete :17500
    INFO:root:Epoch : 7, Batches complete :21000
    INFO:root:Epoch complete :7, Computing Accuracy
    INFO:root:Epochs completed : 7 Test Accuracy: 0.9403157894736842, Test Loss: 0.17671697772343298
    INFO:root:Epoch : 8, Batches complete :0
    INFO:root:Epoch : 8, Batches complete :3500
    INFO:root:Epoch : 8, Batches complete :7000
    INFO:root:Epoch : 8, Batches complete :10500
    INFO:root:Epoch : 8, Batches complete :14000
    INFO:root:Epoch : 8, Batches complete :17500
    INFO:root:Epoch : 8, Batches complete :21000
    INFO:root:Epoch complete :8, Computing Accuracy
    INFO:root:Epochs completed : 8 Test Accuracy: 0.9398421052631579, Test Loss: 0.17823867248302738
    INFO:root:Epoch : 9, Batches complete :0
    INFO:root:Epoch : 9, Batches complete :3500
    INFO:root:Epoch : 9, Batches complete :7000
    INFO:root:Epoch : 9, Batches complete :10500
    INFO:root:Epoch : 9, Batches complete :14000
    INFO:root:Epoch : 9, Batches complete :17500
    INFO:root:Epoch : 9, Batches complete :21000
    INFO:root:Epoch complete :9, Computing Accuracy
    INFO:root:Epochs completed : 9 Test Accuracy: 0.94, Test Loss: 0.17777415844768635
    INFO:root:Epoch : 10, Batches complete :0
    INFO:root:Epoch : 10, Batches complete :3500
    INFO:root:Epoch : 10, Batches complete :7000
    INFO:root:Epoch : 10, Batches complete :10500
    INFO:root:Epoch : 10, Batches complete :14000
    INFO:root:Epoch : 10, Batches complete :17500
    INFO:root:Epoch : 10, Batches complete :21000
    INFO:root:Epoch complete :10, Computing Accuracy
    INFO:root:Epochs completed : 10 Test Accuracy: 0.939921052631579, Test Loss: 0.17711406388310205
    INFO:root:Epoch : 11, Batches complete :0
    INFO:root:Epoch : 11, Batches complete :3500
    INFO:root:Epoch : 11, Batches complete :7000
    INFO:root:Epoch : 11, Batches complete :10500
    INFO:root:Epoch : 11, Batches complete :14000
    INFO:root:Epoch : 11, Batches complete :17500
    INFO:root:Epoch : 11, Batches complete :21000
    INFO:root:Epoch complete :11, Computing Accuracy
    INFO:root:Epochs completed : 11 Test Accuracy: 0.9397894736842105, Test Loss: 0.17753077432728945
    INFO:root:Epoch : 12, Batches complete :0
    INFO:root:Epoch : 12, Batches complete :3500
    INFO:root:Epoch : 12, Batches complete :7000
    INFO:root:Epoch : 12, Batches complete :10500
    INFO:root:Epoch : 12, Batches complete :14000
    INFO:root:Epoch : 12, Batches complete :17500
    INFO:root:Epoch : 12, Batches complete :21000
    INFO:root:Epoch complete :12, Computing Accuracy
    INFO:root:Epochs completed : 12 Test Accuracy: 0.9397105263157894, Test Loss: 0.17750860167579022
    INFO:root:Epoch : 13, Batches complete :0
    INFO:root:Epoch : 13, Batches complete :3500
    INFO:root:Epoch : 13, Batches complete :7000
    INFO:root:Epoch : 13, Batches complete :10500
    INFO:root:Epoch : 13, Batches complete :14000
    INFO:root:Epoch : 13, Batches complete :17500
    INFO:root:Epoch : 13, Batches complete :21000
    INFO:root:Epoch complete :13, Computing Accuracy
    INFO:root:Epochs completed : 13 Test Accuracy: 0.9396842105263158, Test Loss: 0.17746215330466308
    INFO:root:Epoch : 14, Batches complete :0
    INFO:root:Epoch : 14, Batches complete :3500
    INFO:root:Epoch : 14, Batches complete :7000
    INFO:root:Epoch : 14, Batches complete :10500
    INFO:root:Epoch : 14, Batches complete :14000
    INFO:root:Epoch : 14, Batches complete :17500
    INFO:root:Epoch : 14, Batches complete :21000
    INFO:root:Epoch complete :14, Computing Accuracy
    INFO:root:Epochs completed : 14 Test Accuracy: 0.9397631578947369, Test Loss: 0.17755343740698168
    INFO:root:Epoch : 15, Batches complete :0
    INFO:root:Epoch : 15, Batches complete :3500
    INFO:root:Epoch : 15, Batches complete :7000
    INFO:root:Epoch : 15, Batches complete :10500
    INFO:root:Epoch : 15, Batches complete :14000
    INFO:root:Epoch : 15, Batches complete :17500
    INFO:root:Epoch : 15, Batches complete :21000
    INFO:root:Epoch complete :15, Computing Accuracy
    INFO:root:Epochs completed : 15 Test Accuracy: 0.9397368421052632, Test Loss: 0.17758055445022536
    INFO:root:Epoch : 16, Batches complete :0
    INFO:root:Epoch : 16, Batches complete :3500
    INFO:root:Epoch : 16, Batches complete :7000
    INFO:root:Epoch : 16, Batches complete :10500
    INFO:root:Epoch : 16, Batches complete :14000
    INFO:root:Epoch : 16, Batches complete :17500
    INFO:root:Epoch : 16, Batches complete :21000
    INFO:root:Epoch complete :16, Computing Accuracy
    INFO:root:Epochs completed : 16 Test Accuracy: 0.9397368421052632, Test Loss: 0.1775786137607341
    INFO:root:Epoch : 17, Batches complete :0
    INFO:root:Epoch : 17, Batches complete :3500
    INFO:root:Epoch : 17, Batches complete :7000
    INFO:root:Epoch : 17, Batches complete :10500
    INFO:root:Epoch : 17, Batches complete :14000
    INFO:root:Epoch : 17, Batches complete :17500
    INFO:root:Epoch : 17, Batches complete :21000
    INFO:root:Epoch complete :17, Computing Accuracy
    INFO:root:Epochs completed : 17 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758427821354636
    INFO:root:Epoch : 18, Batches complete :0
    INFO:root:Epoch : 18, Batches complete :3500
    INFO:root:Epoch : 18, Batches complete :7000
    INFO:root:Epoch : 18, Batches complete :10500
    INFO:root:Epoch : 18, Batches complete :14000
    INFO:root:Epoch : 18, Batches complete :17500
    INFO:root:Epoch : 18, Batches complete :21000
    INFO:root:Epoch complete :18, Computing Accuracy
    INFO:root:Epochs completed : 18 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758827968265273
    INFO:root:Epoch : 19, Batches complete :0
    INFO:root:Epoch : 19, Batches complete :3500
    INFO:root:Epoch : 19, Batches complete :7000
    INFO:root:Epoch : 19, Batches complete :10500
    INFO:root:Epoch : 19, Batches complete :14000
    INFO:root:Epoch : 19, Batches complete :17500
    INFO:root:Epoch : 19, Batches complete :21000
    INFO:root:Epoch complete :19, Computing Accuracy
    INFO:root:Epochs completed : 19 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758838810507016
    INFO:root:Epoch : 20, Batches complete :0
    INFO:root:Epoch : 20, Batches complete :3500
    INFO:root:Epoch : 20, Batches complete :7000
    INFO:root:Epoch : 20, Batches complete :10500
    INFO:root:Epoch : 20, Batches complete :14000
    INFO:root:Epoch : 20, Batches complete :17500
    INFO:root:Epoch : 20, Batches complete :21000
    INFO:root:Epoch complete :20, Computing Accuracy
    INFO:root:Epochs completed : 20 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758833738161553
    INFO:root:Epoch : 21, Batches complete :0
    INFO:root:Epoch : 21, Batches complete :3500
    INFO:root:Epoch : 21, Batches complete :7000
    INFO:root:Epoch : 21, Batches complete :10500
    INFO:root:Epoch : 21, Batches complete :14000
    INFO:root:Epoch : 21, Batches complete :17500
    INFO:root:Epoch : 21, Batches complete :21000
    INFO:root:Epoch complete :21, Computing Accuracy
    INFO:root:Epochs completed : 21 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758887278141533
    INFO:root:Epoch : 22, Batches complete :0
    INFO:root:Epoch : 22, Batches complete :3500
    INFO:root:Epoch : 22, Batches complete :7000
    INFO:root:Epoch : 22, Batches complete :10500
    INFO:root:Epoch : 22, Batches complete :14000
    INFO:root:Epoch : 22, Batches complete :17500
    INFO:root:Epoch : 22, Batches complete :21000
    INFO:root:Epoch complete :22, Computing Accuracy
    INFO:root:Epochs completed : 22 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758879167733602
    INFO:root:Epoch : 23, Batches complete :0
    INFO:root:Epoch : 23, Batches complete :3500
    INFO:root:Epoch : 23, Batches complete :7000
    INFO:root:Epoch : 23, Batches complete :10500
    INFO:root:Epoch : 23, Batches complete :14000
    INFO:root:Epoch : 23, Batches complete :17500
    INFO:root:Epoch : 23, Batches complete :21000
    INFO:root:Epoch complete :23, Computing Accuracy
    INFO:root:Epochs completed : 23 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758852977060574
    INFO:root:Epoch : 24, Batches complete :0
    INFO:root:Epoch : 24, Batches complete :3500
    INFO:root:Epoch : 24, Batches complete :7000
    INFO:root:Epoch : 24, Batches complete :10500
    INFO:root:Epoch : 24, Batches complete :14000
    INFO:root:Epoch : 24, Batches complete :17500
    INFO:root:Epoch : 24, Batches complete :21000
    INFO:root:Epoch complete :24, Computing Accuracy
    INFO:root:Epochs completed : 24 Test Accuracy: 0.9397105263157894, Test Loss: 0.17758843273002864



```python

```
