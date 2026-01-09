This application classifies emails as spam or legitimate using Logistic Regression trained on 70% of the data and validated on 30%.

&nbsp;Files Structure

1\. `create\_email\_dataset.py` - Creates the email dataset

2\. `email\_spam\_classifier.py` - Main application with Logistic Regression

3\. `test\_classifier.py` - Test script to verify requirements

4\. `email\_dataset.csv` - Generated dataset (after running create\_email\_dataset.py)

5\. `model\_coefficients.csv` - Model coefficients (generated after training)

6\. `email\_classifier\_model.pkl` - Saved model (optional)



\## How to Run



\### Step 1: Create the dataset

```bash

python create\_email\_dataset.py

