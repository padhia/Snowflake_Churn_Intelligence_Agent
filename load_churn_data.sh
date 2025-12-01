#! /bin/sh

cat <<EOF
use role $SF_ROLE;

create database if not exists $SF_DB;
create schema if not exists $SF_DB.$SF_SCH;
use $SF_DB.$SF_SCH;

create stage if not exists data;

put file://$PWD/CUSTOMER_CHURN_DATASET.csv @data/ overwrite = true;

create or replace table customer_churn (
    CUSTOMERID        INTEGER,
    AGE               INTEGER,
    GENDER            TEXT,
    TENURE            INTEGER,
    USAGE_FREQUENCY   INTEGER,
    SUPPORT_CALLS     INTEGER,
    PAYMENT_DELAY     INTEGER,
    SUBSCRIPTION_TYPE TEXT,
    CONTRACT_LENGTH   TEXT,
    TOTAL_SPEND       INTEGER,
    LAST_INTERACTION  INTEGER,
    CHURN             INTEGER
);

copy into customer_churn
   from @data file_format = (type = 'CSV' parse_header = true error_on_column_count_mismatch = false)
   match_by_column_name = case_insensitive
;
EOF
