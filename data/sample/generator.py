import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd

def generate_single_csv(index=1):
    # Reset random seed based on microseconds to guarantee unique datasets on each run
    np.random.seed(int(time.time() * 1000000) % 2**32)
    
    # 1. Completely random sample size for each file (between 150 and 2500 rows)
    num_samples = np.random.randint(150, 2500)
    
    # 2. Randomly select a network traffic scenario to produce distinct statistics
    scenario = np.random.choice(['heavy_attack', 'mostly_normal', 'balanced', 'dos_flood'])
    
    if scenario == 'heavy_attack':
        # High cyber threat scenario
        weights = [0.15, 0.45, 0.20, 0.10, 0.10]
    elif scenario == 'mostly_normal':
        # Mostly safe network traffic scenario
        weights = [0.85, 0.05, 0.04, 0.03, 0.03]
    elif scenario == 'dos_flood':
        # Massive DoS attack scenario
        weights = [0.10, 0.75, 0.05, 0.05, 0.05]
    else:
        # Random balanced distribution scenario
        random_p = np.random.dirichlet(np.ones(5))
        weights = list(random_p)

    classes = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'Botnet']
    labels = np.random.choice(classes, size=num_samples, p=weights)
    
    data = []
    for label in labels:
        if label == 'Normal':
            duration = round(float(np.random.exponential(scale=2.0)), 2)
            protocol = np.random.choice(['tcp', 'udp', 'icmp'], p=[0.7, 0.25, 0.05])
            service = np.random.choice(['http', 'smtp', 'dns', 'ftp', 'ssh', 'other'])
            flag = np.random.choice(['SF', 'S0', 'REJ'], p=[0.92, 0.05, 0.03])
            src_bytes = int(np.random.exponential(scale=1500) + 40)
            dst_bytes = int(np.random.exponential(scale=5000) + 100)
            wrong_fragment, urgent, hot, num_failed_logins, num_compromised, num_root = 0, 0, 0, 0, 0, 0
            count = int(np.random.randint(1, 35))
            srv_count = int(np.random.randint(1, 35))
            serror_rate = round(float(np.random.beta(0.1, 5.0)), 2)
            rerror_rate = round(float(np.random.beta(0.1, 5.0)), 2)
            same_srv_rate = round(float(np.random.uniform(0.8, 1.0)), 2)
            diff_srv_rate = round(float(np.random.uniform(0.0, 0.15)), 2)
            dst_host_count = int(np.random.randint(1, 100))
            dst_host_srv_count = int(np.random.randint(50, 255))
            
        elif label == 'DoS':
            duration = 0.0
            protocol = np.random.choice(['tcp', 'udp', 'icmp'], p=[0.8, 0.1, 0.1])
            service = np.random.choice(['http', 'private', 'other'])
            flag = np.random.choice(['S0', 'REJ', 'SF'], p=[0.75, 0.2, 0.05])
            src_bytes = int(np.random.choice([0, 64, 512, 1024]))
            dst_bytes = 0
            wrong_fragment = int(np.random.choice([0, 1, 3], p=[0.8, 0.1, 0.1]))
            urgent, hot, num_failed_logins, num_compromised, num_root = 0, 0, 0, 0, 0
            count = int(np.random.randint(150, 512))
            srv_count = int(np.random.randint(150, 512))
            serror_rate = round(float(np.random.uniform(0.75, 1.0)), 2)
            rerror_rate = round(float(np.random.uniform(0.0, 0.2)), 2)
            same_srv_rate = round(float(np.random.uniform(0.85, 1.0)), 2)
            diff_srv_rate = round(float(np.random.uniform(0.0, 0.1)), 2)
            dst_host_count = 255
            dst_host_srv_count = int(np.random.randint(10, 255))

        elif label == 'PortScan':
            duration = round(float(np.random.uniform(0.0, 0.5)), 2)
            protocol = 'tcp'
            service = np.random.choice(['private', 'other', 'http', 'ftp', 'ssh'])
            flag = np.random.choice(['REJ', 'RSTO', 'S0'], p=[0.5, 0.3, 0.2])
            src_bytes = int(np.random.choice([0, 40, 60]))
            dst_bytes = 0
            wrong_fragment, urgent, hot, num_failed_logins, num_compromised, num_root = 0, 0, 0, 0, 0, 0
            count = int(np.random.randint(50, 300))
            srv_count = int(np.random.randint(1, 10))
            serror_rate = round(float(np.random.uniform(0.2, 0.8)), 2)
            rerror_rate = round(float(np.random.uniform(0.5, 1.0)), 2)
            same_srv_rate = round(float(np.random.uniform(0.0, 0.2)), 2)
            diff_srv_rate = round(float(np.random.uniform(0.7, 1.0)), 2)
            dst_host_count = int(np.random.randint(100, 255))
            dst_host_srv_count = int(np.random.randint(1, 20))

        elif label == 'BruteForce':
            duration = round(float(np.random.uniform(1.0, 15.0)), 2)
            protocol = 'tcp'
            service = np.random.choice(['ssh', 'ftp', 'smtp'])
            flag = np.random.choice(['SF', 'REJ'], p=[0.7, 0.3])
            src_bytes = int(np.random.randint(100, 800))
            dst_bytes = int(np.random.randint(100, 1200))
            wrong_fragment, urgent = 0, 0
            hot = int(np.random.randint(1, 6))
            num_failed_logins = int(np.random.randint(2, 6))
            num_compromised = int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
            num_root = 0
            count = int(np.random.randint(10, 100))
            srv_count = int(np.random.randint(10, 100))
            serror_rate = round(float(np.random.uniform(0.0, 0.2)), 2)
            rerror_rate = round(float(np.random.uniform(0.0, 0.4)), 2)
            same_srv_rate = round(float(np.random.uniform(0.8, 1.0)), 2)
            diff_srv_rate = round(float(np.random.uniform(0.0, 0.1)), 2)
            dst_host_count = int(np.random.randint(20, 150))
            dst_host_srv_count = int(np.random.randint(20, 150))

        elif label == 'Botnet':
            duration = round(float(np.random.uniform(10.0, 120.0)), 2)
            protocol = np.random.choice(['tcp', 'udp'], p=[0.7, 0.3])
            service = np.random.choice(['private', 'other', 'dns'])
            flag = np.random.choice(['SF', 'S0'], p=[0.8, 0.2])
            src_bytes = int(np.random.randint(50, 400))
            dst_bytes = int(np.random.randint(50, 400))
            wrong_fragment, urgent = 0, 0
            hot = int(np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05]))
            num_failed_logins = 0
            num_compromised = int(np.random.choice([0, 1], p=[0.9, 0.1]))
            num_root = 0
            count = int(np.random.randint(20, 150))
            srv_count = int(np.random.randint(5, 50))
            serror_rate = round(float(np.random.uniform(0.0, 0.3)), 2)
            rerror_rate = round(float(np.random.uniform(0.0, 0.3)), 2)
            same_srv_rate = round(float(np.random.uniform(0.4, 0.8)), 2)
            diff_srv_rate = round(float(np.random.uniform(0.2, 0.6)), 2)
            dst_host_count = 255
            dst_host_srv_count = int(np.random.randint(5, 50))

        data.append({
            'duration': duration, 'protocol_type': protocol, 'service': service, 'flag': flag,
            'src_bytes': src_bytes, 'dst_bytes': dst_bytes, 'wrong_fragment': wrong_fragment,
            'urgent': urgent, 'hot': hot, 'num_failed_logins': num_failed_logins,
            'num_compromised': num_compromised, 'num_root': num_root, 'count': count,
            'srv_count': srv_count, 'serror_rate': serror_rate, 'rerror_rate': rerror_rate,
            'same_srv_rate': same_srv_rate, 'diff_srv_rate': diff_srv_rate,
            'dst_host_count': dst_host_count, 'dst_host_srv_count': dst_host_srv_count,
            'attack_type': label
        })
        
    df = pd.DataFrame(data)
    
    # Inject ~1% random missing values for preprocessor validation testing
    missing_idx = np.random.choice(df.index, size=max(1, int(0.01 * num_samples)), replace=False)
    df.loc[missing_idx, 'src_bytes'] = np.nan
    
    out_dir = os.path.dirname(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"traffic_sample_{timestamp}_{index}.csv"
    out_path = os.path.join(out_dir, file_name)
    
    df.to_csv(out_path, index=False)
    print(f"✅ [{index}] File: {file_name} | Scenario: {scenario} | Rows: {num_samples}")
    time.sleep(1) # Delay 1 second to ensure unique timestamps

if __name__ == '__main__':
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    else:
        try:
            count = int(input("How many CSV files do you want to generate? "))
        except ValueError:
            count = 1

    print(f"\n🚀 Generating {count} CSV file(s) with distinct scenarios...")
    for i in range(1, count + 1):
        generate_single_csv(index=i)
    print("✨ Successfully completed!\n")