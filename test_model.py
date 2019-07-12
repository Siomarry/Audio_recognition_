from builtins import print
from glob import glob
import os
import numpy as np
import pandas as pd
import keras.backend as K

import constants as c
from pre_process import data_catalog, preprocess_and_save
from eval_metrics import evaluate
from models import convolutional_model, recurrent_model
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content
import tensorflow as tf
from pre_process import read_audio,extract_features,find_files
num_neg = c.TEST_NEGATIVE_No

def normalize_scores(m,epsilon=1e-12):
    return (m - np.mean(m)) / max(np.std(m),epsilon)

def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames + 20:
        bias = np.random.randint(20, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    elif x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x

def create_test_data(test_dir,check_partial):
    global num_neg
    libri = data_catalog(test_dir)
    unique_speakers = list(libri['speaker_id'].unique())
    np.random.shuffle(unique_speakers)
    num_triplets = len(unique_speakers)
    if check_partial:
        num_neg = 49; num_triplets = min(num_triplets, 30)
    test_batch = None
    for ii in range(num_triplets):
        anchor_positive_file = libri[libri['speaker_id'] == unique_speakers[ii]]
        if len(anchor_positive_file) <2:
            continue
        anchor_positive_file = anchor_positive_file.sample(n=2, replace=False)
        anchor_df = pd.DataFrame(anchor_positive_file[0:1])
        anchor_df['training_type'] = 'ancfrom thor'                      # 1 anchor，1 positive，num_neg negative
        if test_batch is None:
            test_batch = anchor_df.copy()
        else:
            test_batch = pd.concat([test_batch, anchor_df], axis=0)

        positive_df = pd.DataFrame(anchor_positive_file[1:2])
        positive_df['training_type'] = 'positive'
        test_batch = pd.concat([test_batch, positive_df], axis=0)

        negative_files = libri[libri['speaker_id'] != unique_speakers[ii]].sample(n=num_neg, replace=False)
        for index in range(len(negative_files)):
            negative_df = pd.DataFrame(negative_files[index:index+1])
            negative_df['training_type'] = 'negative'
            test_batch = pd.concat([test_batch, negative_df], axis=0)

    new_x = []
    for i in range(len(test_batch)):
        filename = test_batch[i:i + 1]['filename'].values[0]
        x = np.load(filename)
        new_x.append(clipped_audio(x))
    x = np.array(new_x)  # (batchsize, num_frames, 64, 1)
    new_y = np.hstack(([1], np.zeros(num_neg)))  # 1 positive, num_neg negative
    y = np.tile(new_y, num_triplets)
    return x, y

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul,axis=1)

    #l1 = np.sum(np.multiply(x1, x1),axis=1)
    #l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s

def call_similar(x):
    no_batch = int(x.shape[0] / (num_neg+2))  # each batch was consisted of 1 anchor ,1 positive , num_neg negative, so the number of batch
    similar = []
    for ep in range(no_batch):
        index = ep*(num_neg + 2)
        anchor = np.tile(x[index],(num_neg + 1, 1))
        pos_neg = x[index+1: index + num_neg + 2]
        sim = batch_cosine_similarity(anchor, pos_neg)
        similar.extend(sim)
    return np.array(similar)

def eval_model(model,train_batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH, test_dir= c.TEST_DIR, check_partial=False, gru_model=None):
    #???from the test_Dir there is just one speaker and it's ture?
    x, y_true = create_test_data(test_dir,check_partial)
    # print(x)
    # print(x.shape)
    # print(y_true)
    # print(y_true.shape)
    batch_size = x.shape[0]#153
    b = x[0]
    num_frames = b.shape[0]#160
    input_shape = (num_frames, b.shape[1], b.shape[2])#160X64X1

    '''
    print('test_data:')
    print('num_frames = {}'.format(num_frames))
    print('batch size: {}'.format(batch_size))
    print('input shape: {}'.format(input_shape))
    print('x.shape before reshape: {}'.format(x.shape))
    print('x.shape after  reshape: {}'.format(x.shape))
    print('y.shape: {}'.format(y_true.shape))
    '''
    #embedding = model.predict_on_batch(x)
    test_epoch = int(len(y_true)/train_batch_size)
    embedding = None
    for ep in range(test_epoch):
        x_ = x[ep*train_batch_size: (ep + 1)*train_batch_size]
        embed = model.predict_on_batch(x_)
        if embedding is None:
            embedding = embed.copy()
        else:
            embedding = np.concatenate([embedding, embed], axis=0)
    y_pred = call_similar(embedding)
    if gru_model is not None:
        embedding_gru = None
        for ep in range(test_epoch):
            x_ = x[ep * train_batch_size: (ep + 1) * train_batch_size]
            embed = model.predict_on_batch(x_)
            if embedding_gru is None:
                embedding_gru = embed.copy()
            else:
                embedding_gru = np.concatenate([embedding_gru, embed], axis=0)
        y_pred_gru = call_similar(embedding_gru)

        y_pred = (normalize_scores(y_pred) + normalize_scores(y_pred_gru))/2  # or   y_pred = (y_pred + y_pred_gru)/2

    nrof_pairs = min(len(y_pred), len(y_true))
    y_pred = y_pred[:nrof_pairs]
    #print(y_pred)
    y_true = y_true[:nrof_pairs]
    #print(y_true)
    fm, tpr, acc, eer = evaluate(y_pred, y_true)
    return fm, tpr, acc, eer


def enroll(model,enroll_dir = c.AISHELL_train_dir):
    libri = pd.DataFrame()  # a DataStrcture of 2x2 like Excel Table
    libri['filename'] = find_files(enroll_dir,pattern="*.npy")
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split(".")[0].split("-")[0])
    unique_speakers = list(libri['speaker_id'].unique())
    family_number = len(unique_speakers)
    member = {}
    for i in range(family_number):
        member[i]=unique_speakers[i]

    family_anchor = None
    for ii in range(family_number):
        anchor_positive_file = libri[libri['speaker_id'] == unique_speakers[ii]]
        anchor_positive_file = anchor_positive_file.sample(n=1, replace=False)
        family_anchor = pd.concat([family_anchor,anchor_positive_file],axis=0)

    new_x = []
    for i in range(family_number):
        filename = family_anchor[i:i + 1]['filename'].values[0]
        x = np.load(filename)
        new_x.append(clipped_audio(x))  # now the shape is 151*160*64*1
    x = np.array(new_x)

    embedding = None
    embed = model.predict_on_batch(x)
    if embedding is None:
        embedding = embed.copy()  # now the shape is 151*512

    return embedding,member


def add_enroll(model,enroll_dir = c.AISHELL_train_dir,wavfile = c.dynamic_enroll):
    oldembeddings,member = enroll(model,enroll_dir)

    libri = pd.DataFrame()
    libri['filename'] = find_files(wavfile, pattern="**/*.wav")
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split(".")[0].split("-")[0])
    unique_speakers = list(libri['speaker_id'].unique())
    member_numbers = len(unique_speakers)


    new_member = {}
    for i in range(member_numbers):
        new_member[i+len(member)] = unique_speakers[i]


    dictMerged = member.copy()
    dictMerged.update(new_member)


    new_members_anchor = None
    for ii in range(member_numbers):
        anchor_positive_file = libri[libri['speaker_id'] == unique_speakers[ii]]
        anchor_positive_file = anchor_positive_file.sample(n=1, replace=False)
        new_members_anchor = pd.concat([new_members_anchor, anchor_positive_file], axis=0)

    new_x = []
    for i in range(member_numbers):
        filename = new_members_anchor[i:i + 1]['filename'].values[0]
        raw_audio = read_audio(filename)
        feature = extract_features(raw_audio, target_sample_rate=c.SAMPLE_RATE)
        new_x.append(clipped_audio(feature))  # now the shape is new_numbers*160*64*1
    x = np.array(new_x)

    new_embedding = None
    embed = model.predict_on_batch(x)
    if new_embedding is None:
        new_embedding = embed.copy()  # now the shape is*512
    #oldembeddings = oldembeddings.append(new_embedding)
    oldembeddings = np.append(oldembeddings,new_embedding,axis=0)
    return oldembeddings,dictMerged



def batch_recognition(model,rec_dir = c.MY_WAV_DIR):
    enroll_embeddings,member=enroll(model)
    libri = pd.DataFrame()  # a DataStrcture of 2x2 like Excel Table
    libri['filename'] = find_files(rec_dir, pattern='*.wav')
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('.')[0].split('-')[0])
    unique_speakers = list(libri['speaker_id'].unique())
    family_number = len(unique_speakers)
    #
    # test_people = None
    # for ii in range(family_number):
    #     negative_file = libri[libri['speaker_id'] == unique_speakers[ii]]
    #     negative_file = negative_file.sample(n=1, replace=False)
    #     test_people = pd.concat([test_people, negative_file], axis=0)

    new_x = []
    for i in range(family_number):
        filename = libri[i:i + 1]['filename'].values[0]
        raw_audio = read_audio(filename)
        feature = extract_features(raw_audio, target_sample_rate=c.SAMPLE_RATE)
        new_x.append(clipped_audio(feature))  # now the shape is 29*160*64*1
    x = np.array(new_x)
    x = x.reshape(family_number, 160, 64, 1)
    test_embeddings = model.predict_on_batch(x)#now the shape is 30*512


    people = []
    anchor_length=enroll_embeddings.shape[0]
    accuracy=0.0
    one_shot = 0
    for em in test_embeddings:
        i=0

        result = []
        em = np.tile(em,(anchor_length,1))
        sim = np.array(batch_cosine_similarity(em,enroll_embeddings))
        index = np.argmax(sim)
        if member[index] == unique_speakers[i]:
            one_shot+=1
        if sim[index] > c.My_Famliy_Threshold:

            result.append(str(1))
            result.append(member[index])
            result.append(str(sim[index]))
        else:
            result.append(str(0))
        people.append(result)
        i+=1
    accuracy = float(one_shot/family_number)

    print("accuracy:",accuracy)
    return people







def test_recognifition(model,test_dir = c.MY_TEST_DIR,newwavfile = None):
    libri = data_catalog(test_dir)
    unique_speakers = list(libri['speaker_id'].unique())
    #np.random.shuffle(unique_speakers)
    family_number = len(unique_speakers)
    member={}
    family = {"19":"pengchong","27":"zhaonan","26":"tianxu"}
    for i in range(family_number):
        member[i]=family[unique_speakers[i]]

    family_anchor = None
    for ii in range(family_number):
        anchor_positive_file = libri[libri['speaker_id'] == unique_speakers[ii]]
        anchor_positive_file = anchor_positive_file.sample(n=1, replace=False)
        family_anchor = pd.concat([family_anchor,anchor_positive_file],axis=0)

    new_x = []
    for i in range(family_number):
        filename = family_anchor[i:i + 1]['filename'].values[0]
        x = np.load(filename)
        new_x.append(clipped_audio(x))#now the shape is 3*160*64*1
    x = np.array(new_x)

    embedding = None
    embed = model.predict_on_batch(x)
    if embedding is None:
        embedding = embed.copy()#now the shape is 3*512


    raw_audio = read_audio(newwavfile)
    feature = extract_features(raw_audio, target_sample_rate=c.SAMPLE_RATE)

    newmember = clipped_audio(feature)
    newmember = newmember.reshape(1,160,64,1)
    newembedding = model.predict_on_batch(newmember)

    newembedding = np.tile(newembedding,(3,1))
    sim = np.array(batch_cosine_similarity(embedding, newembedding))

    index = np.argmax(sim)
    if sim[index] > c.My_Famliy_Threshold:
        print("this is one of our famliy member.")
        print("----------********you are : {}!*******-----------".format(member[index]))
    else:
        print("you are not one of my famliy member.Please try again or try anotherone")




if __name__ == '__main__':
     model = convolutional_model()
     gru_model = None
     last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
     if last_checkpoint is not None:
         print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
         model.load_weights(last_checkpoint)
     add_enroll(model, c.AISHELL_train_dir, c.dynamic_enroll)
     # if c.COMBINE_MODEL:
     #     gru_model = recurrent_model()
     #     last_checkpoint = get_last_checkpoint_if_any(c.GRU_CHECKPOINT_FOLDER)
     #     if last_checkpoint is not None:
     #         print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
     #         gru_model.load_weights(last_checkpoint)

     fm, tpr, acc, eer = eval_model(model,test_dir=c.AISHELL_test_dir,check_partial=True,gru_model=gru_model)
    # print("f-measure = {0}, true positive rate = {1}, accuracy = {2}, equal error rate = {3}".format(fm, tpr, acc, eer))
    # model = convolutional_model()
    # gru_model = None
    # last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
    # if last_checkpoint is not None:
    #      print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
    #      model.load_weights(last_checkpoint)
    # #test_recognifition(model,c.TEST_DIR,"audio/LibriSpeechSamples/train-clean-100/19/227/19-227-0036.wav")
    # #test_recognifition(model, c.TEST_DIR, "LibriSpeech/test-clean/61/70968/61-70968-0001.wav")
    # people = batch_recognition(model,c.MY_WAV_DIR)
    # print(people)


