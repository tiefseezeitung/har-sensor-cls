# har-sensor-cls
Hand Activity Recognition (HAR) based on sensor data (accelerometer, gyroscope) from the smartphone and smartwatch using different machine learning models. The goal of this project is to classify hand activities (especially related to learning activities).

## HA24/
The data is from a study conducted in 2024, containing sensor data for various hand activities associated with learning or likely to be occurring during learning scenarios. The dataset comprises accelerometer and gyroscope data from the smartphone and smartwatch, which participants have worn on their dominant arm. Consequently, the setup yields data from four sensors: Accelerometer Watch (Acc. Watch), Accelerometer Phone (Acc. Phone), Gyroscope Watch (Gyr. Watch) and Gyroscope Phone (Gyr. Phone). Due to technical restrictions, sampling rates for phone and watch could not be matched. For the phone it is around 100 Hz and for the watch 104 Hz. In total 10 participants were recorded. 7 of 10 participants have been recorded twice and the remaining 3 participants once, for each of the 15 categories. One recording lasted 2.5 minutes. This results in 75 minutes and 37.5 minutes of record time per user for each of the two groups respectively, and in 637.5 minutes overall recording time. Each category was recorded the same amount of times, ensuring a perfectly balanced set with 17 recordings per category. Participants of this study were aged between 22 and 56 years. The weighted average age is at 33.8 years. Around 52.9% of the data is from male, and 47.1% from female participants. All of the participants were right-handed. Categories are listed below.
| num | hand activity | German description |
| -------- | ------- |------- |
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

## CXT22/
Data was gathered in 2022. It comprises accelerometer, gyroscope and magnetometer data for phone and watch, recorded during learning related activities. It could either expand the recent dataset or could serve only as a test set. For the study the actual data collection took place when participants initiated a learning session using an app on a smartwatch. Each session began with an initiating questionnaire, followed by an In-Session questionnaire every 30 minutes. Upon completion of the session, participants filled out a concluding questionnaire. The questionnaires asked the participant to evaluate various statements on their learning behavior and environment, while sensor data collection simultaneously utilized the smartwatch and smartphone sensors. The relevant sensor data, accelerometer and gyroscope data, was collected right at the start of the In-Session questionnaire, where participants categorized their hand activity conducted in the last 3 minutes (see Table 3.2). However, it needs to be taken into account that an activity was recorded for only 2.5 minutes. Data was gathered from a sample of 8 participants. However, necessary sensor data is available for only 4 participants. Altogether it includes accelerometer, gyroscope and magnetometer recordings for 33 cases with labeled hand activities. Below is the class distribution.
|count	| hand activity|
| -------- | ------- |
 8 	|	 typing on the keyboard / Tippen_am_Computer |
 7 	|	 idle hands / stille_Hände |
 6 	|	 writing with a pen / Schreiben_mit_Stift |
 6 	|	 typing on a smartphone / Tippen_am_Smartphone |
 3 	|	 surfing the internet on a computer / Surfen_am_Computer |
 1 	|	 writing with a digital pen / Schreiben_mit_digitalem_Stift |
 1 	|  surfing the internet on a tablet / Surfen_am_Tablet |
 1 	|	 unsure / unsicher |
 
### on/off-task classification
The file "train_activity_rate.py" is an attempt to estimate, whether a learning interval was including active learning or not. It could also be described as on/off-task learning or (for continuous values) learning activity rate. There are no labels available.

## WISDM/
The WISDM dataset is used in the directory classifying-hand-activities/WISDM/wisdm-dataset/ from Gary M. Weiss:
"Weiss,Gary. (2019). WISDM Smartphone and Smartwatch Activity and Biometrics Dataset . UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59." It includes 18 different full body and hand activities Accelerator and Gyroscope data, for both phone and watch.

Run the python file "extract_data.py" to generate the parquet files used as the input for a model.

To give an example of (possible) input data,  in this graphic sequential data windows of user 1619, labeled with the activity 'writing' are plotted:
</br></br>
<img src=https://github.com/tiefseezeitung/classifying-hand-activities/assets/56825457/fbd00f0e-955a-4e2c-9031-8c2c7ed34cfd width="700">

