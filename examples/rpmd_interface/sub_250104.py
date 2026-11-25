import os
import subprocess
import numpy as np
import json
import threading
import time

# Global variables
process1 = None
process2 = None
last_call_time = time.time()
timeout = 1800  # 30 minutes in seconds

def monitor_processes():
    """Monitor subprocesses and terminate them if inactive for more than 30 minutes."""
    global process1, process2, last_call_time
    while True:
        time.sleep(300)  # Check every 5 minutes
        if time.time() - last_call_time > timeout:
            if process1:
                process1.terminate()
                process1 = None
            if process2:
                process2.terminate()
                process2 = None
            break  # Stop monitoring after termination
def initialize_potential():
    global process1, process2, last_call_time

    python_path = os.environ["PHYSNET_PYTHON"]
    here = os.path.dirname(os.path.abspath(__file__))

    process1 = subprocess.Popen([python_path, os.path.join(here, 'PES_DME_20250118.py')],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    init_cmd = json.dumps({"command": "initialize_potential"})
    process1.stdin.write(init_cmd + '\n')
    process1.stdin.flush()

    process2 = subprocess.Popen([python_path, os.path.join(here, 'PES_diff_20250224.py')],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    process2.stdin.write(init_cmd + '\n')
    process2.stdin.flush()

    # Update last call time
    last_call_time = time.time()

    # Start monitoring thread (Python 2 fix)
    monitor_thread = threading.Thread(target=monitor_processes)
    monitor_thread.daemon = True  # Correct way to set daemon in Python 2
    monitor_thread.start()


def get_potential(q):
    global last_call_time

    if process1 is None or process2 is None:
        raise RuntimeError("Processes have been terminated due to inactivity. Reinitialize the potential.")

    q_json = json.dumps({"command": "get_potential", "data": q.tolist()})

    process1.stdin.write(q_json + '\n')
    process1.stdin.flush()
    output1 = process1.stdout.readline().strip()
    result1 = json.loads(output1)
    V1 = np.array(result1['V'])
    dVdq1 = np.array(result1['dVdq'])

    process2.stdin.write(q_json + '\n')
    process2.stdin.flush()
    output2 = process2.stdout.readline().strip()
    result2 = json.loads(output2)
    V2 = np.array(result2['V'])
    dVdq2 = np.array(result2['dVdq'])

    # Update last call time
    last_call_time = time.time()

    V = V1 + V2
    dVdq = dVdq1 + dVdq2
    return V, dVdq



def close_potential():
    """Close both persistent model workers after the trajectory finishes."""
    global process1, process2
    for process in (process1, process2):
        if process is not None:
            process.stdin.close()
            process.wait()
    process1 = process2 = None
