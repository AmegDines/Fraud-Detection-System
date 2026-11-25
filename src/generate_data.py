"""
Synthetic Financial Transaction Dataset Generator
Produces realistic transaction logs containing normal banking behavior as well as
crafted fraud topologies:
1. High-velocity bursts (rapid drain of accounts)
2. Smurfing / structuring under thresholds
3. Multi-hop money-laundering fraud rings (cyclic and hub-and-spoke rings)
4. Anomalous location / channel transactions
"""

import os
import random
import csv
from datetime import datetime, timedelta

def generate_dataset(output_path="data/sample_transactions.csv", num_records=8000, seed=42):
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    locations = [
        "New York, USA", "London, UK", "Tokyo, Japan", "Singapore, SG",
        "Toronto, Canada", "Sydney, Australia", "Berlin, Germany",
        "Paris, France", "Dubai, UAE", "Zurich, Switzerland",
        "Lagos, Nigeria", "Moscow, Russia", "Bogota, Colombia"
    ]
    channels = ["Mobile App", "Web Browser", "POS Terminal", "ATM", "Wire Transfer"]
    
    # 1. Base Accounts
    normal_senders = [f"ACC_NORM_{i:04d}" for i in range(1, 401)]
    normal_receivers = [f"ACC_RECV_{i:04d}" for i in range(1, 501)]
    
    # 2. Fraud Ring 1: Cyclic mule network (5 nodes loop)
    ring_1_nodes = [f"ACC_RING_A{i}" for i in range(1, 6)]
    
    # 3. Fraud Ring 2: Hub & Spoke money laundering syndicate (1 hub, 6 satellites)
    ring_2_hub = "ACC_HUB_MULE_01"
    ring_2_nodes = [f"ACC_MULE_SATELLITE_{i}" for i in range(1, 7)]
    
    # 4. Fraud Ring 3: Multi-hop cascading chain (6 nodes chain)
    ring_3_nodes = [f"ACC_CHAIN_{i}" for i in range(1, 7)]
    
    # 5. Velocity burst fraudsters
    burst_senders = [f"ACC_BURST_FRAUD_{i:02d}" for i in range(1, 11)]

    records = []
    base_time = datetime(2026, 10, 20, 8, 0, 0)
    current_time = base_time

    # Generate Fraud Ring 1 transactions (Cyclic laundering)
    for cycle in range(25):
        for i in range(len(ring_1_nodes)):
            s = ring_1_nodes[i]
            r = ring_1_nodes[(i + 1) % len(ring_1_nodes)]
            t = base_time + timedelta(hours=cycle * 2, minutes=i * 12 + random.randint(1, 5))
            amount = round(random.uniform(4200, 4980), 2)  # Just under 5k limit
            records.append({
                "transaction_id": f"TXN_RING1_{cycle}_{i}",
                "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": s,
                "receiver_id": r,
                "amount": amount,
                "location": random.choice(["Dubai, UAE", "Zurich, Switzerland", "Singapore, SG"]),
                "payment_channel": "Wire Transfer",
                "device_id": f"DEV_RING1_{i}",
                "is_fraud": 1
            })

    # Generate Fraud Ring 2 transactions (Hub-and-spoke funnels)
    for funnel in range(20):
        t_batch = base_time + timedelta(hours=3 + funnel * 3)
        # Inflow to hub
        for idx, sat in enumerate(ring_2_nodes[:3]):
            records.append({
                "transaction_id": f"TXN_RING2_IN_{funnel}_{idx}",
                "timestamp": (t_batch + timedelta(minutes=idx * 4)).strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": sat,
                "receiver_id": ring_2_hub,
                "amount": round(random.uniform(3000, 4800), 2),
                "location": "Lagos, Nigeria",
                "payment_channel": "Mobile App",
                "device_id": f"DEV_SAT_{idx}",
                "is_fraud": 1
            })
        # Outflow from hub to other satellites
        for idx, sat in enumerate(ring_2_nodes[3:]):
            records.append({
                "transaction_id": f"TXN_RING2_OUT_{funnel}_{idx}",
                "timestamp": (t_batch + timedelta(minutes=30 + idx * 5)).strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": ring_2_hub,
                "receiver_id": sat,
                "amount": round(random.uniform(2800, 4700), 2),
                "location": "Dubai, UAE",
                "payment_channel": "Wire Transfer",
                "device_id": "DEV_HUB_MAIN",
                "is_fraud": 1
            })

    # Generate Fraud Ring 3 (Cascading chain transfers)
    for chain_round in range(15):
        t_chain = base_time + timedelta(days=1, hours=chain_round * 2)
        for i in range(len(ring_3_nodes) - 1):
            s = ring_3_nodes[i]
            r = ring_3_nodes[i + 1]
            records.append({
                "transaction_id": f"TXN_RING3_{chain_round}_{i}",
                "timestamp": (t_chain + timedelta(minutes=i * 8)).strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": s,
                "receiver_id": r,
                "amount": round(random.uniform(6500, 9200), 2),
                "location": "Moscow, Russia",
                "payment_channel": "Web Browser",
                "device_id": f"DEV_CHAIN_{i}",
                "is_fraud": 1
            })

    # Generate Velocity burst fraud (Rapid fund drain)
    for b_idx, sender in enumerate(burst_senders):
        burst_time = base_time + timedelta(days=b_idx % 5, hours=random.randint(1, 23))
        for hit in range(12):  # 12 transactions in 10 minutes
            records.append({
                "transaction_id": f"TXN_BURST_{b_idx}_{hit}",
                "timestamp": (burst_time + timedelta(seconds=hit * 45)).strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": sender,
                "receiver_id": random.choice(normal_receivers),
                "amount": round(random.uniform(900, 2500), 2),
                "location": random.choice(locations),
                "payment_channel": "Mobile App",
                "device_id": f"DEV_BURST_{b_idx}",
                "is_fraud": 1
            })

    # Generate remaining normal & low-risk transactions
    remaining = num_records - len(records)
    for i in range(remaining):
        current_time += timedelta(seconds=random.randint(20, 180))
        sender = random.choice(normal_senders)
        receiver = random.choice(normal_receivers)
        channel = random.choice(channels)
        location = random.choice(locations[:10])  # mostly standard cities
        
        # Log-normal distribution for amounts (mostly small, occasional medium)
        amount = round(random.lognormvariate(4.2, 0.9), 2)
        amount = max(5.0, min(amount, 4500.0))
        
        # 1.5% random isolated fraud (compromised card)
        is_fraud = 1 if random.random() < 0.015 else 0
        if is_fraud:
            amount = round(random.uniform(1200, 5000), 2)
            location = random.choice(locations[10:])  # anomalous location
        
        records.append({
            "transaction_id": f"TXN_GEN_{i:05d}",
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "location": location,
            "payment_channel": channel,
            "device_id": f"DEV_{random.randint(100, 999)}",
            "is_fraud": is_fraud
        })

    # Sort all transactions by timestamp to simulate real sequence
    records.sort(key=lambda x: x["timestamp"])

    # Write out to CSV
    fieldnames = [
        "transaction_id", "timestamp", "sender_id", "receiver_id",
        "amount", "location", "payment_channel", "device_id", "is_fraud"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    fraud_count = sum(1 for r in records if r["is_fraud"] == 1)
    print(f"Generated {len(records)} transactions in '{output_path}'.")
    print(f"Total Fraudulent Transactions: {fraud_count} ({fraud_count/len(records)*100:.2f}%)")
    return output_path

if __name__ == "__main__":
    generate_dataset()
