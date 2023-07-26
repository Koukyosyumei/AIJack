# AIValut

AIValut is a DBMS that allows you to debug your ML models. We implement the core engine and parser of DBMS from scratch with reference to [bogoDB](https://github.com/ad-sho-loko/bogoDB) and [db_tutorial](https://github.com/cstack/db_tutorial). 

## Install

```bash
./script/build.sh
```

## Basic Usage

You can start AIValut with simple commands.


```bash
# server side
./aivalut -i localhost -p 8889 -s
```

```bash
# client side
./aivalut -i localhost -p 8889
```

The above command opens an interactive window for the client to query the server. You can use SQL-like queries as follows:

```bash
# We support Int, Float, and Varchar as data types.
Create Table `table_name` {`primary_key_name` `data_type` Primary Key, `col_name` `data_type`, ...}

Insert Into `table_name` Values (`value_1`, `value_2`, ...)

# We support * symbol.
# We support Eq, Geq, and Leq as operators.
Select `col_name` From `table_name`
Select `col_name` From `table_name` Where `col_name` `operator` `value`
Select `col_name` From `table_name_x` Join `table_name_y` On `key_name_of_x` Eq `key_name_of_y`

# You can also query multiple commands from a text file
# Some sample files can be found in the `example` directory.
source query.avi

# You can safely exit with the following command
exit
```

-   Example 1

```sql
>>Create Table a {aid Int Primary Key, ascore Int}
>>Insert Into a Values (1, 1)
>>Insert Into a Values (2, 10)
>>Insert Into a Values (3, 10)

>>Create Table b {bid Int Primary Key, bscore Int}
>>Insert Into b Values (2, 31)
>>Insert Into b Values (3, 12)

>>Select * From a Join b On aid Eq bid Where ascore Geq 1
```

The main feature of AIValut is that it can internally train and debug an ML model with SQL-like Query.

```bash
# Training Logistic Regression
Logreg `model_name` `primary_key_name` `target_column_name` `number_of_iteration` `learning_rate` From Select `primary_key_name`, `feature_name` From `table_name`

# Debugging with Rain
# We currently support `SUM` aggregation for Logistic Regression for binary classification
# `target_class` specifies the desired class that you want the model to predict for samples satisfying `Where` constraints 
Complaint `complaint_name` Shouldbe `target_class` Remove `number_of_removed_records` Against Logreg `model_name_to_be_debugged` `primary_key_name` `target_column_name` `number_of_iteration` `learning_rate` From Select `primary_key_name`, `feature_name` From `table_name` Where `condition`
```

-   Example 2

```sql
# We train an ML model to classify whether each customer will go bankrupt or not based on their age and debt.
# You want the trained model to classify the customer as positive when he/she has more debt than or equal to 100.
# The 10th record seems problematic for the above constraint.
>>Select * From bankrupt
id age debt y
1 40 0 0
2 21 10 0
3 22 10 0
4 32 30 0
5 44 50 1
6 30 100 1
7 63 310 1
8 53 420 1
9 39 530 1
10 49 1000 0

# Train Logistic Regression with the number of iterations of 100 and the learning rate of 1.
# The name of the target feature is `y`, and We use all other features as training data.
>>Logreg lrmodel id y 100 1 From Select * From bankrupt
Trained Parameters:
 (0) : 2.771564
 (1) : -0.236504
 (2) : 0.967139
AUC: 0.520000
Prediction on the training data is stored at `prediction_on_training_data_lrmodel`

# Remove one record so that the model will predict `positive (class 1)` for the samples with `debt` greater or equal to 100.
>>Complaint comp Shouldbe 1 Remove 1 Against Logreg lrmodel id y 100 1 From Select * From bankrupt Where debt Geq 100
Fixed Parameters:
 (0) : -4.765492
 (1) : 8.747224
 (2) : 0.744146
AUC: 1.000000
Prediction on the fixed training data is stored at `prediction_on_training_data_comp_lrmodel`
```
