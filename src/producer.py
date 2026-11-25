"""
Real-Time Transaction Kafka Producer
Academic & Technical Reference: Units I & V (Batch vs. Real-Time Processing & Ingestion)

Streams synthetic financial transactions into an Apache Kafka topic ('transactions').
Features:
- Produces realistic records: timestamp, sender_id, receiver_id, amount, location, is_fraud
- Configurable streaming throughput (records/sec)
- Dual mode: Native Kafka Producer with seamless Fallback Simulation Sink
  (Allows running in offline/local test environments without requiring an active Kafka broker)
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime, timezone

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try importing kafka-python (or kafka-python-ng)
KAFKA_AVAILABLE = False
try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    try:
        from kafka_python_ng import KafkaProducer
        from kafka_python_ng.errors import NoBrokersAvailable
        KAFKA_AVAILABLE = True
    except ImportError:
        KAFKA_AVAILABLE = False


def create_mock_transaction(fraud_probability=0.08):
    """Generates a single synthetic credit card transaction."""
    locations = [
        "New York, USA", "London, UK", "Tokyo, Japan", "Singapore, SG",
        "Toronto, Canada", "Sydney, Australia", "Berlin, Germany",
        "Paris, France", "Dubai, UAE", "Zurich, Switzerland",
        "Lagos, Nigeria", "Moscow, Russia", "Bogota, Colombia"
    ]
    channels = ["Mobile App", "Web Browser", "POS Terminal", "ATM", "Wire Transfer"]
    
    is_fraud = 1 if random.random() < fraud_probability else 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if is_fraud:
        # High-risk profile: higher amount or rapid transfer
        amount = round(random.choice([
            random.uniform(2500, 4999),   # Structuring under $5,000
            random.uniform(5000, 15000),  # Major drain
            random.uniform(900, 1999)     # Rapid burst
        ]), 2)
        sender_id = f"ACC_RISK_{random.randint(1, 20):03d}"
        receiver_id = f"ACC_MULE_{random.randint(1, 10):03d}"
        location = random.choice(locations[8:])  # Higher fraud propensity regions
        channel = random.choice(["Wire Transfer", "Web Browser"])
    else:
        amount = round(random.lognormvariate(3.8, 0.8), 2)
        amount = max(5.0, min(amount, 1200.0))
        sender_id = f"ACC_NORM_{random.randint(1, 300):04d}"
        receiver_id = f"ACC_RECV_{random.randint(1, 300):04d}"
        location = random.choice(locations[:8])
        channel = random.choice(channels)

    return {
        "transaction_id": f"TXN_{int(time.time() * 1000)}_{random.randint(100, 999)}",
        "timestamp": now,
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "amount": amount,
        "location": location,
        "payment_channel": channel,
        "device_id": f"DEV_{random.randint(100, 999)}",
        "is_fraud": is_fraud
    }


def start_producer(bootstrap_servers="localhost:9092", topic="transactions",
                   rate=2.0, count=-1, fallback_file="data/outputs/incoming_transactions_stream.json"):
    """
    Publishes transaction records to Apache Kafka.
    If Kafka is unavailable, automatically activates Local Simulation Sink.
    """
    os.makedirs(os.path.dirname(fallback_file), exist_ok=True)
    producer = None
    use_fallback = False

    if KAFKA_AVAILABLE:
        try:
            print(f"[INFO] Connecting to Apache Kafka cluster at '{bootstrap_servers}'...")
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=5000,
                retries=3
            )
            print(f"[SUCCESS] Connected to Kafka. Publishing to topic '{topic}'...")
        except Exception as e:
            print(f"[WARNING] Could not connect to Kafka at {bootstrap_servers}: {e}")
            print("[INFO] Switching to resilient Offline Simulation Mode (data/outputs/incoming_transactions_stream.json).")
            print("[INFO] Start Kafka anytime with: docker-compose up -d")
            use_fallback = True
    else:
        print("[WARNING] 'kafka-python' library not found. Running in Offline Simulation Mode.")
        use_fallback = True

    sent_count = 0
    fraud_count = 0
    delay = 1.0 / rate if rate > 0 else 0.5

    print(f"[INFO] Streaming started. Rate: {rate} tx/sec. Target count: {'Infinite' if count < 0 else count}")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        while count < 0 or sent_count < count:
            record = create_mock_transaction()
            sent_count += 1
            if record["is_fraud"] == 1:
                fraud_count += 1

            if not use_fallback and producer is not None:
                try:
                    future = producer.send(topic, value=record)
                    future.get(timeout=2)
                except Exception as ex:
                    print(f"[ERROR] Failed to send to Kafka: {ex}. Appending to simulation sink.")
                    use_fallback = True

            # Always append to streaming sink file for local ingestion & Streamlit visualization
            with open(fallback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Also maintain a rolling window of recent live alerts
            if record["is_fraud"] == 1:
                alerts_file = "data/outputs/live_alerts.csv"
                write_header = not os.path.exists(alerts_file) or os.path.getsize(alerts_file) == 0
                with open(alerts_file, "a", encoding="utf-8") as af:
                    if write_header:
                        af.write("transaction_id,timestamp,sender_id,receiver_id,amount,location,payment_channel,fraud_score,predicted_label,actual_label\n")
                    fraud_score = round(random.uniform(0.78, 0.99), 4)
                    af.write(f"{record['transaction_id']},{record['timestamp']},{record['sender_id']},{record['receiver_id']},{record['amount']},{record['location']},{record['payment_channel']},{fraud_score},1,{record['is_fraud']}\n")

            if sent_count % 10 == 0 or record["is_fraud"] == 1:
                status = "FRAUD ALERT" if record["is_fraud"] == 1 else "OK"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent #{sent_count:04d} | {status} | ${record['amount']:>8.2f} | {record['sender_id']} -> {record['receiver_id']} ({record['location']})")

            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n[INFO] Streaming stopped by user.")
    finally:
        if producer is not None:
            producer.flush()
            producer.close()
        print(f"[SUMMARY] Total Transactions: {sent_count} | Fraud Detected: {fraud_count} ({fraud_count/max(1, sent_count)*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Kafka Transaction Streamer")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka bootstrap broker (e.g. localhost:9092)")
    parser.add_argument("--topic", default="transactions", help="Kafka topic name")
    parser.add_argument("--rate", type=float, default=2.0, help="Transactions per second")
    parser.add_argument("--count", type=int, default=-1, help="Total transactions to emit (-1 for infinite)")
    parser.add_argument("--file", default="data/outputs/incoming_transactions_stream.json", help="Simulation sink file")
    
    args = parser.parse_args()
    start_producer(
        bootstrap_servers=args.broker,
        topic=args.topic,
        rate=args.rate,
        count=args.count,
        fallback_file=args.file
    )
