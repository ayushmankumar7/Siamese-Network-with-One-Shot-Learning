import tensorflow as tf 
import numpy as np 
import pickle

from model import siamese_network


model = siamese_network((105,105, 1))

print(model.summary())

save_path = './data/'


with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)
    
print("Training alphabets: \n")
print(list(train_classes.keys()))



with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))



def get_batch(batch_size,s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    
    categories = np.random.choice(n_classes,size=(batch_size,),replace=False)
    
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
    
    targets=np.zeros((batch_size,))

    
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = np.random.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = np.random.randint(0, n_examples)
        
        
        if i >= batch_size // 2:
            category_2 = category  
        else: 
            
            category_2 = (category + np.random.randint(1,n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)
    
    return pairs, targets

