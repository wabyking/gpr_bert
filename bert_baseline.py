import os
import gc
import numpy as np
import pandas as pd 
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import modeling
import extract_features
import tokenization

val_df = pd.read_table('gap-validation.tsv', index_col='ID').reset_index(drop=True)
test_df  = pd.read_table('gap-validation.tsv', index_col='ID').reset_index(drop=True)
dev_df  = pd.read_table('gap-development.tsv', index_col='ID').reset_index(drop=True)

def count_char(text, offset):   
    count = 0
    for pos in range(offset):
        if text[pos] != " ": count +=1
    return count

def candidate_length(candidate):
    count = 0
    for i in range(len(candidate)):
        if candidate[i] !=  " ": count += 1
    return count

def count_token_length_special(token):
    count = 0
    special_token = ["#", " "]
    for i in range(len(token)):
        if token[i] not in special_token: count+=1
    return count

def embed_by_bert(df, path_to_bert='d:/dataset/bert/uncased_L-12_H-768_A-12', embed_size=768, batch_size=8,
                 layers='-1', max_seq_length=256):
    
    text = df['Text']
    text.to_csv('input.txt', index=False, header=False)
    os.system(f"python extract_features.py \
               --input_file=input.txt \
               --output_file=output.jsonl \
               --vocab_file={path_to_bert}/vocab.txt \
               --bert_config_file={path_to_bert}/bert_config.json \
               --init_checkpoint={path_to_bert}/bert_model.ckpt \
               --layers={layers} \
               --max_seq_length={max_seq_length} \
               --batch_size={batch_size}")
    
    bert_output = pd.read_json("output.jsonl", lines=True)
    bert_output.head()
    
    os.system("rm input.txt")
    os.system("rm output.jsonl")
    
    index = df.index
    columns = ["emb_A", "emb_B", "emb_P", "label"]
    emb = pd.DataFrame(index = index, columns = columns)
    emb.index.name = "ID"
    
    for i in tqdm(range(len(text))):
        
        features = bert_output.loc[i, "features"]
        P_char_start = count_char(df.loc[i, 'Text'], df.loc[i, 'Pronoun-offset'])
        A_char_start = count_char(df.loc[i, 'Text'], df.loc[i, 'A-offset'])
        B_char_start = count_char(df.loc[i, 'Text'], df.loc[i, 'B-offset'])
        A_length = candidate_length(df.loc[i, 'A'])
        B_length = candidate_length(df.loc[i, 'B'])
        
        emb_A, emb_B, emb_P = np.zeros(embed_size), np.zeros(embed_size), np.zeros(embed_size)
        char_count, cnt_A, cnt_B = 0, 0, 0
        
        for j in range(2, len(features)):
            token = features[j]["token"]
            token_length = count_token_length_special(token)
            if char_count == P_char_start:
                emb_P += np.asarray(features[j]["layers"][0]['values']) 
            if char_count in range(A_char_start, A_char_start + A_length):
                emb_A += np.asarray(features[j]["layers"][0]['values'])
                cnt_A += 1
            if char_count in range(B_char_start, B_char_start + B_length):
                emb_B += np.asarray(features[j]["layers"][0]['values'])
                cnt_B += 1                
            char_count += token_length
        
        if cnt_A > 0:
            emb_A /= cnt_A
        if cnt_B > 0:
            emb_B /= cnt_B
        
        label = "Neither"
        if (df.loc[i,"A-coref"] == True):
            label = "A"
        if (df.loc[i,"B-coref"] == True):
            label = "B"

        emb.iloc[i] = [emb_A, emb_B, emb_P, label]
        
    return emb



val_bert_emb = embed_by_bert(val_df)
test_bert_emb = embed_by_bert(test_df)
dev_bert_emb = embed_by_bert(dev_df)


val_bert_emb["emb_A"].head().map(np.asarray).values[0].astype('float').shape
def featurize(embedding_df):
    
    pronoun_embs, a_embs, b_embs, labels = [], [], [], []
    
    for i in tqdm(range(len(embedding_df))):
        
        pronoun_embs.append(embedding_df.loc[i, "emb_P"])
        a_embs.append(embedding_df.loc[i, "emb_A"])
        b_embs.append(embedding_df.loc[i, "emb_B"])

        label_map = {'A': 0, 'B': 1, 'Neither': 2}
        labels.append(label_map[embedding_df.loc[i, "label"]])

    
    a_embs = np.asarray(a_embs).astype('float')
    b_embs = np.asarray(b_embs).astype('float') 
    pronoun_embs = np.asarray(pronoun_embs).astype('float')
    
    return np.concatenate([a_embs, b_embs, pronoun_embs], axis=1), np.asarray(labels)

X_train, y_train = featurize(pd.concat([val_bert_emb, dev_bert_emb]).sort_index().reset_index())
X_train.shape, y_train.shape
logit = LogisticRegression(C=1e-2, random_state=17, solver='lbfgs', 
                           multi_class='multinomial', max_iter=100,
                          n_jobs=4)
logit.fit(X_train, y_train)

stage1_test_df  = pd.read_table('gap-development.tsv', index_col='ID').reset_index(drop=True)

stage1_test_bert_emb = embed_by_bert(stage1_test_df)
X_test, y_test = featurize(stage1_test_bert_emb)
logit_test_pred = logit.predict_proba(X_test)
log_loss(y_test, logit_test_pred)
submission = pd.read_csv("../input/sample_submission_stage_1.csv", index_col = "ID")
submission["A"] = logit_test_pred[:, 0]
submission["B"] = logit_test_pred[:, 1]
submission["NEITHER"] = logit_test_pred[:, 2]
submission.to_csv("submission.csv")