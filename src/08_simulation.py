"""
08_simulation.py
----------------
Simulates error-rate improvements under three enhancements:
1) Lower slot-filling threshold
2) Model confidence fallback
3) API retry mechanism

Input: outputs/vivollo_final.csv
Output: outputs/simulation_result.csv
Usage:
    python src/08_simulation.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

def simulate():
    df = pd.read_csv(Path('outputs') / 'vivollo_final.csv', parse_dates=['created_at'])
    user = df[df.sender_type=='user'].copy()
    n = len(user)

    # 1) Slot-filling scenarios
    np.random.seed(0)
    user['filled_slots'] = np.random.randint(0,5,size=n)
    orig = (user['filled_slots']<4).mean()
    new_thresh = (user['filled_slots']<2).mean()

    # 2) Model fallback scenario
    user['conf'] = np.random.rand(n)
    fallback = (user['conf']<0.4).mean()

    # 3) API retry scenario
    base_err = 0.15
    post_err = 0.03
    user['api'] = np.random.rand(n)<base_err
    final_api = (user['api'] & (np.random.rand(n)<post_err)).mean()

    # Results
    res = pd.DataFrame({
        'Scenario':['Original slot','Thresh=2','Model fallback','API retry'],
        'Error rate (%)':[(orig*100).round(2),(new_thresh*100).round(2),
                         (fallback*100).round(2),(final_api*100).round(2)]
    })
    res.to_csv(Path('outputs')/'simulation_result.csv', index=False)
    print(res)

if __name__=='__main__': simulate()
