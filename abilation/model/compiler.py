import time
import numpy as np
import tensorflow as tf
import csv
import time








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
        for epoch in range(self.num_epochs):



            self.time_train_start= time.time()

            # for batch in range(self.train_data_mask.shape[0]):
            all_train_results = {name: [] for name in self.metric_names}

            for batch in range(self.train_data_mask.shape[0]):

                if batch < self.train_data_mask.shape[0] - 1:

                    batch_results = self.model.train_on_batch(

                        self.train_data_mri[batch], self.train_data_mask[batch], return_dict=True
                    )
                    for name, value in batch_results.items():

                        all_train_results[name].append(value)

                    self._print_progress_bar(batch)

                if batch == self.train_data_mask.shape[0] - 1:

                    batch_results = self.model.train_on_batch(

                        self.train_data_mri[batch], self.train_data_mask[batch], return_dict=True
                    )
                    for name, value in batch_results.items():

                        all_train_results[name].append(value)

                    
                    self.time_train_end= time.time()


                    val_results = self._evaluate_validation_set()

                    train_results={name: np.mean(values) for name, values in all_train_results.items()}

                    results_str = ' '.join([f'{name}: {value:.4f}' for name, value in train_results.items()])

                    val_results_str = ' '.join([f'Val_{name}: {value:.4f}' for name, value in val_results.items()])

                    self.save_to_csv(train_results,val_results)

                    print(f'{results_str} {val_results_str}')
                    
                    self._save_best_model(val_results['loss'])  # Assuming val_loss is under 'loss'

                    
                    if self._early_stopping(val_results['loss']):

                        print(f'Early stopping triggered after epoch {epoch+1}')

                        return
    
    def _print_progress_bar(self, batch):

        percent = (batch + 1) / self.train_data_mask.shape[0]

        bar_length = 40

        bar = '-' * int(bar_length * percent) + '>' + '_' * (bar_length - int(bar_length * percent))

        print(f'\r{" " * (bar_length + 10)}', end='')  # Clear the line

        print(f'\r[{bar}] {int(percent * 100)} % ', end='', flush=True)  # Print the new progress

        time.sleep(0.1)
    
    def _evaluate_validation_set(self):

        all_val_results = {name: [] for name in self.metric_names}
      # for val_batch in range(self.validation_data_mask.shape[0]):

        for val_batch in range(self.validation_data_mask.shape[0]):

            batch_results = self.model.test_on_batch(

                self.validation_data_mri[val_batch], self.validation_data_mask[val_batch], return_dict=True
            )

            for name, value in batch_results.items():

                all_val_results[name].append(value)
        
        return {name: np.mean(values) for name, values in all_val_results.items()}
    
    def _save_best_model(self, val_loss):

        if val_loss < self.best_val_loss:

            self.best_val_loss = val_loss

            self.model.save(f'{self.checkpoint_path}/{self.checkpoint_name}.h5')

            print(f'Saved best model with val_loss: {val_loss:.4f}')

            self.wait = 0

        else:

            self.wait += 1
    
    def save_to_csv(self,train_result,validation_result):

        if self.first_csv_write:


            duration= self.time_train_end - self.time_train_start

            result=[train_result | {f'val_{name}':value for name,value in validation_result.items()} | {'time':duration}]

            lesion_coronal_distribution = f'{self.checkpoint_path}/{self.checkpoint_name}.csv'

            with open(lesion_coronal_distribution, 'w', newline='') as csvfile:
                fieldnames = result[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Writing header
                writer.writeheader()

                # Writing data
                writer.writerows(result)

            self.first_csv_write= False
            
        if not  self.first_csv_write:

            duration= self.time_train_end - self.time_train_start

            result=[train_result | {f'val_{name}':value for name,value in validation_result.items()} | {'time':duration} ]

            lesion_coronal_distribution = f'{self.checkpoint_path}/{self.checkpoint_name}.csv'

            with open(lesion_coronal_distribution, 'a') as csvfile:

                fieldnames = result[0].keys()

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                

                # Writing data
                writer.writerow(result[0])



    def _early_stopping(self, val_loss):
        
        return self.wait >= self.patience