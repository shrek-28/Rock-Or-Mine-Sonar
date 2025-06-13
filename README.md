# Did the Sonar Signal Come from a Rock or Mine?

## Introduction 
The Rock vs. Mine (Sonar) dataset is a binary classification dataset consisting of 208 sonar signal samples, each with 60 numerical features representing the energy of sonar signals at various frequencies. The goal is to classify whether the signal was reflected from a metal cylinder (mine) or a rock. Each instance is labeled as either ```"M"``` for mine or ```"R"``` for rock, based on the object the sonar signal bounced off. The dataset is commonly used in machine learning to test algorithms in signal processing, especially for distinguishing subtle patterns in high-dimensional data.

```pandas``` was utilized for data analysis operations, with ```scikit-learn``` being used for the machine learning operations using SVM. ```LIME``` was utilized for model explainability.

## Dataset Features
The dataset was obtained from the following link: https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks

The dataset's features include:
| Feature No. | Attribute Name | Description |
|-------------|----------------|-------------|
| 1           | Attribute_1    | Energy in frequency band 1 |
| 2           | Attribute_2    | Energy in frequency band 2 |
| 3           | Attribute_3    | Energy in frequency band 3 |
| 4           | Attribute_4    | Energy in frequency band 4 |
| 5           | Attribute_5    | Energy in frequency band 5 |
| 6           | Attribute_6    | Energy in frequency band 6 |
| 7           | Attribute_7    | Energy in frequency band 7 |
| 8           | Attribute_8    | Energy in frequency band 8 |
| 9           | Attribute_9    | Energy in frequency band 9 |
| 10          | Attribute_10   | Energy in frequency band 10 |
| 11          | Attribute_11   | Energy in frequency band 11 |
| 12          | Attribute_12   | Energy in frequency band 12 |
| 13          | Attribute_13   | Energy in frequency band 13 |
| 14          | Attribute_14   | Energy in frequency band 14 |
| 15          | Attribute_15   | Energy in frequency band 15 |
| 16          | Attribute_16   | Energy in frequency band 16 |
| 17          | Attribute_17   | Energy in frequency band 17 |
| 18          | Attribute_18   | Energy in frequency band 18 |
| 19          | Attribute_19   | Energy in frequency band 19 |
| 20          | Attribute_20   | Energy in frequency band 20 |
| 21          | Attribute_21   | Energy in frequency band 21 |
| 22          | Attribute_22   | Energy in frequency band 22 |
| 23          | Attribute_23   | Energy in frequency band 23 |
| 24          | Attribute_24   | Energy in frequency band 24 |
| 25          | Attribute_25   | Energy in frequency band 25 |
| 26          | Attribute_26   | Energy in frequency band 26 |
| 27          | Attribute_27   | Energy in frequency band 27 |
| 28          | Attribute_28   | Energy in frequency band 28 |
| 29          | Attribute_29   | Energy in frequency band 29 |
| 30          | Attribute_30   | Energy in frequency band 30 |
| 31          | Attribute_31   | Energy in frequency band 31 |
| 32          | Attribute_32   | Energy in frequency band 32 |
| 33          | Attribute_33   | Energy in frequency band 33 |
| 34          | Attribute_34   | Energy in frequency band 34 |
| 35          | Attribute_35   | Energy in frequency band 35 |
| 36          | Attribute_36   | Energy in frequency band 36 |
| 37          | Attribute_37   | Energy in frequency band 37 |
| 38          | Attribute_38   | Energy in frequency band 38 |
| 39          | Attribute_39   | Energy in frequency band 39 |
| 40          | Attribute_40   | Energy in frequency band 40 |
| 41          | Attribute_41   | Energy in frequency band 41 |
| 42          | Attribute_42   | Energy in frequency band 42 |
| 43          | Attribute_43   | Energy in frequency band 43 |
| 44          | Attribute_44   | Energy in frequency band 44 |
| 45          | Attribute_45   | Energy in frequency band 45 |
| 46          | Attribute_46   | Energy in frequency band 46 |
| 47          | Attribute_47   | Energy in frequency band 47 |
| 48          | Attribute_48   | Energy in frequency band 48 |
| 49          | Attribute_49   | Energy in frequency band 49 |
| 50          | Attribute_50   | Energy in frequency band 50 |
| 51          | Attribute_51   | Energy in frequency band 51 |
| 52          | Attribute_52   | Energy in frequency band 52 |
| 53          | Attribute_53   | Energy in frequency band 53 |
| 54          | Attribute_54   | Energy in frequency band 54 |
| 55          | Attribute_55   | Energy in frequency band 55 |
| 56          | Attribute_56   | Energy in frequency band 56 |
| 57          | Attribute_57   | Energy in frequency band 57 |
| 58          | Attribute_58   | Energy in frequency band 58 |
| 59          | Attribute_59   | Energy in frequency band 59 |
| 60          | Attribute_60   | Energy in frequency band 60 |
| 61          | Class          | Target label — 'R' for Rock, 'M' for Mine |

## Methodology:
1. Initial data description was conducted, using ```df.describe()``` and ```df.info()```.
2. A simple heatmap was utilized to identify the correlations present in between the different features.
3. The dataset was split into a 9:1 ratio, for training and testing, and scaled using ```StandardScaler()```.
4. ```GridSearch``` was utilized in order to test for different ```k``` values. Accuracy was visualized for different valeus of ```k```.
5. KNN was utilized with cross validation to get superior results.
6. The model was evaluated with confusion matrix, AUC-ROC Score and classification reports.
7. The model was explained using ```LIME```.

## Results:
The confusion matrix displayed a remarkable 19 true values, and 2 false values, in the test dataset of 21 values, and the accuracy score was obtained to be 0.91.
The classification report was as follows:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| M     | 0.92      | 0.92   | 0.92     | 13      |
| R     | 0.88      | 0.88   | 0.88     | 8       |
| **Accuracy** |       |        | **0.90**  | 21      |
| **Macro Avg** | 0.90      | 0.90   | 0.90     | 21      |
| **Weighted Avg** | 0.90      | 0.90   | 0.90     | 21      |

The area under the AUC-ROC curve was obtained to be 0.9. 
LIME was used to explain the model. 
Key insights from the explanation:
- **`Freq_36 > 0.57`** had the **strongest positive influence**, strongly supporting the prediction of 'M'.
- Other features such as **`Freq_19 <= 0.30`**, **`Freq_22 <= 0.42`**, and **`Freq_35 > 0.59`** also contributed positively.
- Conversely, features like **`Freq_29 > 0.85`**, **`Freq_41 > 0.39`**, and **`Freq_28 > 0.90`** contributed negatively, pulling the prediction toward class 'R'.
This interpretation provided transparency into the model’s decision-making and validated the reliability of its predictions for this instance.

The LIME plot can be obtained from the Jupyter Notebook. 
