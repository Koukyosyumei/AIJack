# AIValut

AIValut is a DBMS, which allows you to debugg your ML models. We implement the core engine and parser of DBMS from scratch with reference to [bogoDB](https://github.com/ad-sho-loko/bogoDB) and [db_tutorial](https://github.com/cstack/db_tutorial). 

## Install

```
./script/build.sh
```

## Basic Usage

You can start AIValut with simple commands.


```
# server side
./aivalut -i localhost -p 8889 -s
```

```
# client side
./aivalut -i localhost -p 8889
```

The above command opens an interative window for the client to query the server. You can use SQL-like queries as follows:

```bash
# we support Int, Float, and Varchar as data types.
Create Table `table_name` {`primary_key_name` `data_type` Primary Key, `col_name` `data_type`, ...}

Insert Into `table_name` Values (`value_1`, `value_2`, ...)

# We support * symbol.
# We support Eq, Geq, and Leq as operators.
Select `col_name` From `table_name`
Select `col_name` From `table_name` Where `col_name` `operator` `value`
Select `col_name` From `table_name_x` Join `table_name_y` On `key_name_of_x` Eq `key_name_of_y`

# You can also query multiple commands from a text file
# Some sample files can be found at the `example` directory.
source query.avi

# You can safely exit with the following command
exit
```

The main feature of AIValut is that it can internally train and debug a ML model with SQL-like Query.

```bash
# Training Logistic Regression
Logreg `model_name` `primary_key_name` `target_column_name` `number_of_iteration` `learning_rate` From Select `primary_key_name`, `feature_name` From `table_name`

# Debuging
Complaint `complaint_name` `target_class` `number_of_removed_records` Logreg `model_name_to_be_debugged` `primary_key_name` `target_column_name` `number_of_iteration` `learning_rate` From Select `primary_key_name`, `feature_name` From `table_name` Where `condition`
```

- Example 1

```bash
Create Table a {aid Int Primary Key, ascore Int}
Insert Into a Values (1, 1)
Insert Into a Values (2, 10)
Insert Into a Values (3, 10)
greg lrmodel id y 100 1 From Select * From dummy

Create Table b {bid Int Primary Key, bscore Int}
Insert Into b Values (2, 31)
Insert Into b Values (3, 12)

Select * From a Join b On aid Eq bid Where ascore Geq 1
```
