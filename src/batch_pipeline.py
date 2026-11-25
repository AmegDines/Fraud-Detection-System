"""
Batch Analytics & SQL Reporting (Spark SQL & Window Functions)
Academic & Technical Reference: Unit IV (Relational Analytics & Windowing)

Executes distributed relational queries utilizing:
1. Window functions (velocity checks, rolling 5-minute sums, account-level ranks)
2. Complex GROUP BY, HAVING, and aggregate filters to spot high-risk behavior
3. Relational JOINs across senders, receivers, and geographic transaction streams
4. Outputs comprehensive summary reporting to 'data/outputs/sql_metrics.csv'
"""

import os
import sys
import csv
import json
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PYSPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


def get_spark_session(app_name="FraudDetection_BatchSQL"):
    """Initializes Spark Session for SQL processing."""
    if not PYSPARK_AVAILABLE:
        return None
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_batch_pipeline_pyspark(transactions_csv="data/sample_transactions.csv",
                               output_csv="data/outputs/sql_metrics.csv"):
    """
    Executes Spark SQL queries to compute windowed velocity metrics,
    spikes, and account aggregations.
    """
    print("\n" + "="*70)
    print("  UNIT IV: RELATIONAL ANALYTICS & SPARK SQL WINDOWING")
    print("="*70)
    
    spark = get_spark_session()
    print(f"[INFO] Ingesting transactions into Spark DataFrame from '{transactions_csv}'...")
    
    df = spark.read.csv(transactions_csv, header=True, inferSchema=True)
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    
    # 1. Register temporary SQL view
    df.createOrReplaceTempView("transactions")
    print("[SPARK SQL] Registered temporary view 'transactions'.")
    
    # 2. Window Aggregation: Velocity Spikes
    # Calculate rolling transactions and amounts per account
    print("[SPARK SQL] Executing window velocity check query...")
    window_query = """
    SELECT 
        transaction_id,
        timestamp,
        sender_id,
        receiver_id,
        amount,
        location,
        payment_channel,
        is_fraud,
        COUNT(*) OVER (
            PARTITION BY sender_id 
            ORDER BY CAST(timestamp AS TIMESTAMP) 
            RANGE BETWEEN INTERVAL 10 MINUTES PRECEDING AND CURRENT ROW
        ) AS tx_velocity_10min,
        ROUND(SUM(amount) OVER (
            PARTITION BY sender_id 
            ORDER BY CAST(timestamp AS TIMESTAMP) 
            RANGE BETWEEN INTERVAL 10 MINUTES PRECEDING AND CURRENT ROW
        ), 2) AS amount_velocity_10min,
        ROUND(AVG(amount) OVER (PARTITION BY sender_id), 2) AS sender_avg_amount,
        ROW_NUMBER() OVER (PARTITION BY sender_id ORDER BY amount DESC) AS sender_rank_by_amount
    FROM transactions
    """
    windowed_df = spark.sql(window_query)
    windowed_df.createOrReplaceTempView("transactions_windowed")
    
    # 3. High-Risk Account Aggregation with GROUP BY and HAVING
    print("[SPARK SQL] Executing Account Risk Aggregation with GROUP BY and HAVING...")
    agg_query = """
    SELECT 
        sender_id AS account_id,
        COUNT(*) AS total_transactions,
        ROUND(SUM(amount), 2) AS total_volume_usd,
        ROUND(AVG(amount), 2) AS avg_transaction_usd,
        ROUND(MAX(amount), 2) AS max_single_transaction,
        MAX(tx_velocity_10min) AS max_10min_velocity,
        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_events_count,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS fraud_percentage,
        COUNT(DISTINCT location) AS distinct_locations_count,
        COUNT(DISTINCT payment_channel) AS distinct_channels_count
    FROM transactions_windowed
    GROUP BY sender_id
    HAVING 
        COUNT(*) >= 5 
        OR SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) > 0 
        OR MAX(tx_velocity_10min) >= 3
    ORDER BY fraud_events_count DESC, max_10min_velocity DESC, total_volume_usd DESC
    """
    metrics_df = spark.sql(agg_query)
    
    print("[SPARK SQL] Sample of high-risk velocity metrics:")
    metrics_df.show(5, truncate=False)
    
    # 4. Save to CSV output sink
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pandas_df = metrics_df.toPandas()
    pandas_df.to_csv(output_csv, index=False)
    
    # Also save location distribution summary
    loc_query = """
    SELECT 
        location,
        COUNT(*) AS tx_count,
        ROUND(SUM(amount), 2) AS total_amount,
        SUM(is_fraud) AS fraud_count,
        ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
    FROM transactions
    GROUP BY location
    ORDER BY fraud_count DESC, total_amount DESC
    """
    loc_df = spark.sql(loc_query).toPandas()
    loc_output = "data/outputs/location_metrics.csv"
    loc_df.to_csv(loc_output, index=False)
    
    print(f"[SUCCESS] Spark SQL metrics successfully exported to '{output_csv}' and '{loc_output}'.")
    spark.stop()
    return output_csv


def run_batch_pipeline_fallback(transactions_csv="data/sample_transactions.csv",
                                output_csv="data/outputs/sql_metrics.csv"):
    """
    High-performance fallback implementing the identical Spark SQL Windowing,
    Velocity, and Aggregation logic using Pandas for immediate execution.
    """
    print("\n" + "="*70)
    print("  UNIT IV: RELATIONAL ANALYTICS & VELOCITY WINDOWING")
    print("="*70)
    
    import pandas as pd
    import numpy as np
    
    print(f"[INFO] Ingesting transactions from '{transactions_csv}'...")
    df = pd.read_csv(transactions_csv)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["sender_id", "timestamp_dt"])
    
    print("[SQL ENGINE] Calculating rolling 10-minute velocity window aggregates...")
    
    # Rolling 10-minute velocity count & sum
    velocity_counts = []
    velocity_amounts = []
    
    grouped = df.groupby("sender_id")
    for sender_id, group in grouped:
        times = group["timestamp_dt"].values
        amounts = group["amount"].values
        n = len(times)
        
        counts = np.zeros(n, dtype=int)
        sums = np.zeros(n, dtype=float)
        
        for i in range(n):
            t_curr = times[i]
            t_min = t_curr - np.timedelta64(10, 'm')
            
            # Find elements in [t_min, t_curr]
            mask = (times[:i+1] >= t_min)
            counts[i] = np.sum(mask)
            sums[i] = np.sum(amounts[:i+1][mask])
            
        velocity_counts.extend(counts)
        velocity_amounts.extend(sums)
        
    df["tx_velocity_10min"] = velocity_counts
    df["amount_velocity_10min"] = np.round(velocity_amounts, 2)
    
    print("[SQL ENGINE] Executing Account Risk Aggregation (GROUP BY & HAVING)...")
    summary = df.groupby("sender_id").agg(
        total_transactions=("transaction_id", "count"),
        total_volume_usd=("amount", lambda x: round(x.sum(), 2)),
        avg_transaction_usd=("amount", lambda x: round(x.mean(), 2)),
        max_single_transaction=("amount", lambda x: round(x.max(), 2)),
        max_10min_velocity=("tx_velocity_10min", "max"),
        fraud_events_count=("is_fraud", "sum"),
        distinct_locations_count=("location", "nunique"),
        distinct_channels_count=("payment_channel", "nunique")
    ).reset_index().rename(columns={"sender_id": "account_id"})
    
    summary["fraud_percentage"] = np.round(
        summary["fraud_events_count"] * 100.0 / summary["total_transactions"], 2
    )
    
    # HAVING filter:
    # COUNT(*) >= 5 OR fraud_events_count > 0 OR max_10min_velocity >= 3
    filtered = summary[
        (summary["total_transactions"] >= 5) |
        (summary["fraud_events_count"] > 0) |
        (summary["max_10min_velocity"] >= 3)
    ].sort_values(by=["fraud_events_count", "max_10min_velocity", "total_volume_usd"], ascending=[False, False, False])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    
    # Location summary
    loc_summary = df.groupby("location").agg(
        tx_count=("transaction_id", "count"),
        total_amount=("amount", lambda x: round(x.sum(), 2)),
        fraud_count=("is_fraud", "sum")
    ).reset_index()
    loc_summary["fraud_rate_pct"] = np.round(loc_summary["fraud_count"] * 100.0 / loc_summary["tx_count"], 2)
    loc_summary = loc_summary.sort_values(by=["fraud_count", "total_amount"], ascending=[False, False])
    loc_summary.to_csv("data/outputs/location_metrics.csv", index=False)
    
    print(f"[SUCCESS] Relational analytics exported to '{output_csv}' ({len(filtered)} high-risk accounts).")
    return output_csv


def run_batch_pipeline(transactions_csv="data/sample_transactions.csv", output_csv="data/outputs/sql_metrics.csv"):
    if PYSPARK_AVAILABLE:
        try:
            return run_batch_pipeline_pyspark(transactions_csv, output_csv)
        except Exception as e:
            print(f"[WARNING] PySpark SQL pipeline encountered an issue ({e}). Using fallback relational engine...")
            return run_batch_pipeline_fallback(transactions_csv, output_csv)
    else:
        print("[INFO] PySpark not installed yet. Running high-performance relational analytics engine...")
        return run_batch_pipeline_fallback(transactions_csv, output_csv)


if __name__ == "__main__":
    run_batch_pipeline()
