import time
import numpy as np
import tensorflow as tf
import csv
import time
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback



class TimeHistory(Callback):

    def __init__(self, csv_file):
        super(TimeHistory, self).__init__()
        self.csv_file = csv_file

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Time'])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_time_start
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1} training time: {epoch_time:.2f} seconds")

        # Write to CSV file
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_time])




class ModelTrainer:
    def __init__(self, 
                    model,

                    train_data_mri,

                    train_data_mask,

                    validation_data_mri,

                    validation_data_mask, 

                    optimizer, 

                    loss,

                    metrics,

                    metrics_names,

                    checkpoint_path,

                    checkpoint_name,

                    patience,

                    num_epochs

                    ):
        
        self.model = model


        self.train_data_mri = train_data_mri

        self.train_data_mask = train_data_mask

        self.validation_data_mri = validation_data_mri

        self.validation_data_mask = validation_data_mask

        self.num_epochs = num_epochs

        self.patience = patience

        self.metric_names = metrics_names

        self.checkpoint_path = checkpoint_path

        self.checkpoint_name = checkpoint_name



        self.first_csv_write=True

        self.best_val_loss = float('inf')

        self.wait = 0
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)




    def train(self):


        csv_file = f'{self.checkpoint_path}/epoch_times.csv'

        time_callback = TimeHistory(csv_file)


        csv_logger = CSVLogger(f'{self.checkpoint_path}/{self.checkpoint_name}.csv',append=True , separator=';')



        checkpointer=tf.keras.callbacks.ModelCheckpoint(f'{self.checkpoint_path}/{self.checkpoint_name}.h5',verbose=1,save_best_only=True)
        

        callbacks=[checkpointer,tf.keras.callbacks.EarlyStopping(patience=30,monitor='val_loss')]




        self.model.fit(self.train_data_mri,self.train_data_mask,validation_data=(self.validation_data_mri,self.validation_data_mask),shuffle=True,batch_size=32,epochs=self.num_epochs,callbacks=[callbacks,csv_logger,time_callback])
