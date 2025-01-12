# har-sensor-cls

Hand Activity Recognition (HAR) based on sensor data (accelerometer, gyroscope) from the smartphone and smartwatch using different machine learning models. The goal of this project is to classify hand activities (especially related to learning activities).


The 3 directories *HA24/*, *CXT22/* and *WISDM/* contain the corresponding dataset and data specific python files, for example for extracting data from the raw data files.

*preprocessings.py*, *split_utils.py*, *load_data.py* and *eval_utils.py* are modular components of the project.

- *preprocessings.py* contains functions for splitting sequences into windows, for extracting features from sequences and for creating spectrograms of the time-series data.
- *split_utils.py* contains functions for splitting data into training, validation and test sets considering different aspects. 
- *load_data.py* contains two functions: *load_df()*  for loading data from the Parquet file(s) and returning it in a dataframe and *get_inputs()* for loading all relevant data for a training process, returning the input data (sequences/spectrograms), labels, features, etc. as numpy arrays.
- *eval_utils.py* contains functions for evaluation, calculation of specific metrics, etc.

*visualize_data.py* can be used for plotting sequences from the data files. It isn’t relevant for the classification task. Nevertheless, using the Spyder IDE, the file can be executed block-wise to plot data from selected users and activities.
*features_rf.py* was used to assess the value of higher features extracted from sequences using Random Forest. It also isn’t relevant for the classification task.

The files *train.py*, *kfold_cv_eval.py*, *train_im_personal.py* and *eval_other_ds.py* are files, which each load a dataset, split it and then train or evaluate one or multiple models. 
- *train.py* loads the data using the function from *load_data.py*. A split option can be defined for the desired type of split. If **train** is set to True a training process is started, else the model is loaded from the provided file. Then follows a short evaluation, calculating loss, accuracy and top-3 accuracy. If **evaluate** is set True, an extended evaluation is conducted, printing recall, precision, F1 scores for each set and class, including confusion matrices.
- *kfold_cv_eval.py* loads the data using the function from *load_data_st24.py*. Then in a loop k training processes are conducted successively. 
- *train_im_personal.py* works similarly but trains personal/impersonal models in a for loop, user after user. Keras models are saved to *trained_models/users/*. If **personal** is set to True, will train a model for each user appearing in the dataset. Otherwise impersonal models are trained, where the model trains with all data except the chosen user’s one as it is reserved for the test set.
- *eval_other_ds.py*  is used for loading the CXT22 or WISDM dataset, aligned to the data used for training the HA24 dataset, and evaluating it.

*models/* contains files for building different keras models. *trained_models/* contains trained keras models and accompanying log files. 

## HA24/

The data is from a study conducted in 2024 by students of Goethe-Universität Frankfurt, containing sensor data for various hand activities associated with learning or likely to be occurring during learning scenarios. The dataset comprises accelerometer and gyroscope data from the smartphone and smartwatch, which participants have worn on their dominant arm. Consequently, the setup yields data from four sensors: Accelerometer Watch (Acc. Watch), Accelerometer Phone (Acc. Phone), Gyroscope Watch (Gyr. Watch) and Gyroscope Phone (Gyr. Phone). Due to technical restrictions, sampling rates for phone and watch could not be matched. For the phone it is around 100 Hz and for the watch 104 Hz. In total 10 participants were recorded. 7 of 10 participants have been recorded twice and the remaining 3 participants once, for each of the 15 categories. One recording lasted 2.5 minutes. This results in 75 minutes and 37.5 minutes of record time per user for each of the two groups respectively, and in 637.5 minutes overall recording time. Each category was recorded the same amount of times, ensuring a perfectly balanced set with 17 recordings per category. Participants of this study were aged between 22 and 56 years. The weighted average age is at 33.8 years. Around 52.9% of the data is from male, and 47.1% from female participants. All of the participants were right-handed. Categories are listed below.

| num | hand activity | German description |
| :-------- | :------- | :------- |
| 1 | idle hands | Stille Hände |
| 2 | typing on a smartphone | Tippen am Smartphone |
| 3 | typing on a tablet | Tippen am Tablet |
| 4 | typing on the keyboard | Tippen am Computer |
| 5 | scrolling on a smartphone | Scrollen am Smartphone |
| 6 | scrolling on a tablet | Scrollen am Tablet |
| 7 | using the computer mouse | Bedienung der Computermaus |
| 8 | using the touchpad | Bedienung des Touchpads |
| 9 | writing with a pen | Schreiben mit Stift |
| 10 | reading a book | Lesen in einem Buch |
| 11 | making a phone call | Telefonieren |
| 12 | eating | Essen |
| 13 | drinking | Trinken |
| 14 | scratching | Kratzen |
| 15 | fidgeting | Zappeln |

The dataset is located in a compressed form at *HA24/data.zip*. For recreating the Parquet files in *study24_seq/* and *study24.parquet*, it needs to be unzipped.

The file *extract_data.py* is for reading the original JSON files and saving the metadata, containing activity ID, label, user ID, starttime, endtime and elapsed seconds into a file named *study24.parquet*. At the same time the 3-dimensional sequences are separately saved into Parquet files as well. Once the program is executed, the file *study24.parquet* and the directory *study24_seq/*, containing all sequences, are created. Each of the 255 files have file names <activity_ID>_<user_ID>_<sensor_name>_<activity_name>.parquet.

The file *auxiliary.py* is a module containing a dictionary with English category names and sequence length values for phone / watch. The *data_inspect.py* file can be used for analyzing the *study24.parquet* metadata table and creating queries or statistics. By using the Spyder IDE, the file can be executed block-wise.

## CXT22/

The dataset CXT22 contains data gathered by students from Goethe-Universität Frankfurt in 2022. It includes selfreport, usage and sensor data recorded in learning environments, or learning sessions. It can be downloaded from https://hessenbox-a10.rz.uni-frankfurt.de/getlink/fiAkZUUTsKBeAhY3vV46nH/ (password: password). It comprises accelerometer, gyroscope and magnetometer data for the smartphone and smartwatch, recorded during learning related activities. For the study the actual data collection took place when participants initiated a learning session using an app on a smartwatch. Each session began with an initiating questionnaire, followed by an In-Session questionnaire every 30 minutes. Upon completion of the session, participants filled out a concluding questionnaire. The questionnaires asked the participant to evaluate various statements on their learning behavior and environment, while sensor data collection simultaneously utilized the smartwatch and smartphone sensors. The relevant sensor data, accelerometer and gyroscope data, was collected right at the start of the In-Session questionnaire, where participants categorized their hand activity conducted in the last 3 minutes. However, it needs to be taken into account that an activity was recorded for only 2.5 minutes. Data was gathered from a sample of 8 participants. However, necessary sensor data is available for only 4 participants. Altogether it includes sensor data for 33 cases with labeled hand activities. Below is the class distribution is shown.

| count	| hand activity | German description |
| :-------- | :------- | :------- |
 8 	|	 typing on the keyboard | Tippen_am_Computer |
 7 	|	 idle hands | stille_Hände |
 6 	|	 writing with a pen | Schreiben_mit_Stift |
 6 	|	 typing on a smartphone | Tippen_am_Smartphone |
 3 	|	 surfing the internet on a computer | Surfen_am_Computer |
 1 	|	 writing with a digital pen | Schreiben_mit_digitalem_Stift |
 1 	|  surfing the internet on a tablet | Surfen_am_Tablet |
 1 	|	 unsure | unsicher |

The dataset can be downloaded from https://hessenbox-a10.rz.uni-frankfurt.de/getlink/fiAkZUUTsKBeAhY3vV46nH/ (password: password). 

The *Sessions/* folder, containing the raw data files, comprises subfolders sorted by user name. Raw data files are JSON files, one for each recorded session. 

The *extract_data.py* file reads in all JSON files one after another, and saves the selfreport as well as the sensor data and time-series data. The two modules *mappings.py*, containing dictionaries and abbreviations for saving data from the session files, and *schemas.py*, containing the schemas used for the Parquet files, are utilized during this process.

Depending on the duration of the session, a different number of rows / samples is saved per file, as for each session interval one row is created. The extracted data is saved into two Parquet files: *sessions_selfreport.parquet* and *sessions_sensory.parquet*.

The sequential data is saved separately into the sessions_seq/ directory analogously to the primary dataset. The files *sessions_notifs.parquet*, containing the number of received notifications, and the *sessions_seq/usage/* directory, containing used apps, were also saved longside, but are not relevant for the examined classification task.

The *data_inspect.py* file can be used for gaining insights of the data by loading the *sessions_selfreport.parquet* and *sessions_sensory.parquet*, and creating queries or statistics. Using the Spyder IDE, the file can be executed block-wise.

### on/off-task classification

The file *train_activity_rate.py* is an attempt to estimate, whether a learning interval can be assigned to 'active learning' or 'no active learning'. It could also be described as on/off-task learning or, for continuous values, learning activity rates can be calculated. There are no labels available.

## WISDM/

The WISDM dataset is placed in the directory har-sensor-cls/WISDM/wisdm-dataset/ and is from Gary M. Weiss:
"Weiss,Gary. (2019). WISDM Smartphone and Smartwatch Activity and Biometrics Dataset . UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59." It includes 18 different full body and hand activities Accelerator and Gyroscope data, for both phone and watch.

To give an example of possible input data, in this graphic sequential data windows of user 1619, labeled with the activity 'writing' are plotted:
</br></br>
<img src=https://github.com/tiefseezeitung/classifying-hand-activities/assets/56825457/fbd00f0e-955a-4e2c-9031-8c2c7ed34cfd width="700">

The dataset in *wisdm-dataset/* is in the same folder structure as it was downloaded from (https://doi.org/10.24432/C5HK59).
- *extract_data.py* is executed to read in the raw txt files to produce all listed Parquet files. Some activities are not available for all sensors. In order to save only data samples for which every sensor is available, the dataframes are merged and saved into *wisdm_tables/wisdm_merged.py* and corresponding sequence files are saved into *wisdm_seq/*, grouped after their sensor type.
- *wisdm_activities.py* encompasses auxiliary variables, like a dictionary for translating the single letter labels into string descriptions and lists as subsets of labels like hand activities or hand activities related to learning.
- *data_inspect.py* can be used for gaining insights of the data by loading the *wisdm_merged.parquet*, and creating queries or statistics. Using the Spyder IDE, the file can be executed block-wise.

## models/
The *models/* directory contains 5 python files with functions for loading different neural network model architectures as Keras models.

models/

├── sequential.py

├── CNN.py 

├── ViT.py 

├── Swin.py

├── CSWin.py

*sequential.py* contains a broader spectrum of models. For the time-series data (3 dimensional sequences) it encompasses a simple LSTM with self-attention, a 2 layered LSTM with self-attention and batch normalization, a simple Transformer and a 1-dimensional CNN. It also contains a simple Feedforward model, for receiving the 1-dimensional feature arrays (for inputs as features extracted from the time-series data).

*CNN.py* contains a function for building a 2D CNN, which includes a 2 layered convolution block and residual connections. Additionally the file contains *build_CNN_FF()* for building the same CNN model, combined with the Feedforward network as in *sequential.py*.

*ViT.py* contains the function *build_ViT1()* for building a Keras model of a Vision Transformer, including positional embeddings and a class token.

*Swin.py* implements the functionality of a Swin Transformer. It contains 2 different functions for building a Swin model. *build_Swin_orig()* applies PatchMerging after every Transformer block (except the last) as described in the original paper. *build_Swin()* applies **PatchMerging** after two Transformer blocks. Non-shifted and shifted Transformer blocks are always applied in turns and one iteration consists of two Transformer blocks.

*CSWin.py* implements the functionality of a CSWin Transformer. The function *build_CSWin()* builds and returns the Keras model.
