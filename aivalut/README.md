# AIValut

AIValut is a DBMS, which allows you to debugg your ML models. We implement this DBMS from scratch with reference to [bogoDB](https://github.com/ad-sho-loko/bogoDB) and [db_tutorial](https://github.com/cstack/db_tutorial). 

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
Create Table `table_name` {`primary_key_name` `data_type` Primary Key, `col_name` `data_type`}

# We support * symbol.
# We support Eq, Geq, and Leq as operators.
Select `col_name` From `table_name`
Select `col_name` From `table_name` Where `col_name` `operator` `value`
Select `col_name` From `table_name_x` Join `table_name_y` On `key_name_of_x` Eq `key_name_of_y`
```

```bash
Create Table a {aid Int Primary Key, ascore Int}
Insert Into a Values (1, 1)
Insert Into a Values (2, 10)
Insert Into a Values (3, 10)

Create Table b {bid Int Primary Key, bscore Int}
Insert Into b Values (2, 31)
Insert Into b Values (3, 12)

Select * From a Join b On aid Eq bid Where ascore Geq 1
```
