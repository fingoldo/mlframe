import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

def plot_learning_curve(model_name:str,hist:object)->None:
    plt.plot(hist.history['accuracy']);
    plt.plot(hist.history['val_accuracy']);
    plt.title(f"[{model_name}] model's accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
class KerasDictConverter( BaseEstimator, TransformerMixin):

    """
        Idea of this transformer is similar to the one of ColumnTransformer: to apply different pipelines to various column subsets, and to merge results into single dict,
        which can be passed directly to a multi-input Keras model.
        There should be a signature which both pipeline and Keras model know.
        Goal here is that part of inputs (product name) gets tokenized,
        and other part (price) gets passed to an imputer, then to a scaler.
    """
    
    def __init__( self, transformers, remainder='drop', tokenizer=None, tokenizer_kwargs=None):
        """
            transformers: list of (name, transformer, columns) tuples
                transformer: {'drop', 'passthrough', 'tokenizer'} or estimator
        """
        self.dimensions={}
        self.remainder = remainder
        self.tokenizer = tokenizer                        
        self.transformers = transformers
        self.tokenizer_kwargs=tokenizer_kwargs
        if self.tokenizer_kwargs is None: self.tokenizer_kwargs = dict(return_tensors='np',truncation=True,padding=True,max_length=self.tokenizer.model_max_length)
        
        self.fit_just_made = False

    def fit( self, X, y = None ):
        for name, transformer, columns in self.transformers:
            if hasattr(transformer,'fit'):
                transformer.fit(X[columns],y)
        self.fit_just_made = True
        return self 
    
    def transform( self, X ):
        res={}
        for name, transformer, columns in self.transformers:
            if hasattr(transformer,'transform'):
                res[name]=transformer.transform(X[columns])   
                if self.fit_just_made:
                    self.dimensions[name]=res[name].shape[1]
            elif transformer=='tokenizer':
                assert len(columns)==1                          
                tokenized=self.tokenizer(X[columns[0]].to_list(),**self.tokenizer_kwargs)
                for key,arr in tokenized.items():
                    res[name+'_tokenized_'+key]=arr
                if self.fit_just_made:
                    if 'tokenized' not in self.dimensions: self.dimensions['tokenized']={}
                    self.dimensions['tokenized'][name]=arr.shape[1]
        self.fit_just_made = False
        return res      
        
def create_multiinput__keras_model(blockdimensions:dict,ModelTemplate:object,model_name:str,nclasses:int,transformers_trainable:bool=False)->object:
    global_inputs={}
    blocks=[]
    for block,ndim in blockdimensions.items():
        if block=='tokenized':            
            for var, size in ndim.items():
                # Local pretrained model
                new_holder={}
                
                new_holder['input_ids'] = tf.keras.Input(shape=(None,), dtype='int32')
                new_holder['attention_mask'] = tf.keras.Input(shape=(None,), dtype='int32')

                transformer = ModelTemplate.from_pretrained(model_name,num_labels=nclasses)
                transformer.layers[0].trainable=transformers_trainable
                
                logits = transformer({"input_ids": new_holder['input_ids'], "attention_mask": new_holder['attention_mask']})[0]
                #logits = tf.keras.layers.GlobalAveragePooling1D()(logits)  # reduce tensor dimensionality

                new_holder['model'] = tf.keras.models.Model(inputs = {"input_ids": new_holder['input_ids'], "attention_mask": new_holder['attention_mask']}, outputs = logits)        
                
                blocks.append(new_holder)
                
                for subkey in ['input_ids','attention_mask']:
                    global_inputs[var+'_tokenized_'+subkey]=new_holder[subkey]
        else:
            # Regular blocks!
            new_holder={}                
            new_holder['input'] = tf.keras.Input(shape=(ndim,), dtype='float32')            

            next_block = tf.keras.layers.Dense(ndim)(new_holder['input'])
            if ndim//2>0:
                next_block = tf.keras.layers.Dense(ndim//2)(next_block)                

            new_holder['model']=tf.keras.models.Model(inputs = new_holder['input'], outputs = next_block)
            blocks.append(new_holder)
            
            global_inputs[block]=new_holder['input']
            
    # combining branches
    if len(blocks)>1:
        output = tf.keras.layers.concatenate([block['model'].output for block in blocks], name='concatenate_all')
    else:
        output = blocks[0]['model'].output
    
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense(256, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.1)(output)    
    output = tf.keras.layers.Dense(nclasses, name='class_output',activation='softmax')(output) #
    


    final_model = tf.keras.models.Model(inputs=global_inputs, outputs=output)
    
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
    
    return final_model
