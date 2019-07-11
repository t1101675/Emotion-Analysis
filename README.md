# Emational Analysis

## 1 Preparation

``` bash
cd emotion/
```

### 1.1 Install Requirements

```bash
pip3 install -r requirements.txt
```

### 1.2 Prepare train & test data

```bash
mv your_train_data data/train.in
mv your_test_data data/test.in
```

### 1.3 Get pre-trained word embeddings

Sogou news word embeddings are used:

+ [sgns.sogou.word](https://pan.baidu.com/s/1FO39ZYy1mStERf_b53Y_yQ)

+ [sgns.sogou.bigram](https://pan.baidu.com/s/13yVrXeGYkxdGW3P6juiQmA)

+ [sgns.sogounews.bigram-char](https://pan.baidu.com/s/1svFOwFBKnnlsqrF1t99Lnw)

```bash
# After unzipping word embeddings file
mv path_to_word_embeddings data/
```

### 1.4 Pre-process Data

```bash
python3 main.py --data_dir data/ --process_data --pre_vec your_word_embeddings_name
```

Now `data/` should be like:

![1559382178515](C:\Users\GuYuxian\AppData\Roaming\Typora\typora-user-images\1559382178515.png)

## 2 Train

### 2.1 Best Models Train

```bash
#cnn
python3 main.py --data_dir data/ --model cnn --save_dir models/cnn --batch_size 1 --epoch 500 --pre_vec sgns.sogou.bigram

#rnn
python3 main.py --data_dir data/ --model rnn --save_dir models/rnn --batch_size 1 --epoch 200 --pre_vec sgns.sogou.bigram --initial orthogonal

#mlp(baseline)
python3 main.py --data_dir data/ --model baseline --save_dir models/cnn --batch_size 1 --epoch 500 --pre_vec sgns.sogou.bigram
```

### 2.2 Other Train Examples

```bash
# get help
python3 main.py -h
```

```
  -h, --help            show this help message and exit
  --model MODEL         select model type
  --test_only           select test only mode
  --data_dir DATA_DIR   data directory
  --process_data        process data mode
  --class_dim CLASS_DIM
                        the number of classes
  --batch_size BATCH_SIZE
                        batch size
  --test_batch_size TEST_BATCH_SIZE
                        test batch size
  --hidden_dim HIDDEN_DIM
                        hidden dimension for rnn
  --embedding_size EMBEDDING_SIZE
                        embedding size for word vector
  --num_layers NUM_LAYERS
                        rnn layer number
  --dropout DROPOUT     drop out rate the probability to drop an element
  --epoch EPOCH         epoch to train
  --lr LR               learning rate
  --gpu_device GPU_DEVICE
                        choose cuda device
  --optim OPTIM         optimizer SGD or Adam
  --loss LOSS           select loss, CEL or MSE, default is CEL
  --pre_vector PRE_VECTOR
                        pretrained vector name
  --finetune_pv         whether to finetune the pre-trained vectors
  --fix_length FIX_LENGTH
                        fix length passages
  --save_dir SAVE_DIR   select dir to save models
  --load_dir LOAD_DIR   selet models to load models
  --initial INITIAL     choose a parameter initialization, should be
                        orthogonal or normal

```

```bash
# normal initialization & no dropout
python3 main.py --data_dir data/ --model cnn --save_dir models/test_cnn --pre_vec sgns.sogou.bigram --initial normal --dropout 1

# othogonal initialization & set learning rate
python3 main.py --data_dir data/ --model cnn --save_dir models/test_cnn --pre_vec sgns.sogou.bigram --initial normal --lr 0.005

#use other word embeddings & set batch size
python3 main.py --data_dir data/ --model cnn --save_dir models/test_cnn --pre_vec sgns.sogounews.bigram-char --batch_size 64

#use other MSE loss and Adam for optimize
python3 main.py --data_dir data/ --model cnn --save_dir models/test_cnn --pre_vec sgns.sogounews.bigram-char --loss MSE
```



## 3 Test

```bash
#best cnn
python3 main.py --data_dir data/ --model cnn --load_dir models/best_cnn --pre_vec sgns.sogou.bigram

#best rnn
python3 main.py --data_dir data/ --model rnn --load_dir models/best_rnn --pre_vec sgns.sogou.bigram

#best mlp(baseline)
python3 main.py --data_dir data/ --model baseline --load_dir models/best_baseline --pre_vec sgns.sogou.bigram
```

