import subprocess
import threading
import time

import progressbar
import psutil


class Miner:
    def __init__(self, mining_command):
        self.enabled = True
        self.miner = None
        self.mining_command = mining_command

        self.miner_thread = threading.Thread(target=self.mine)
        self.miner_thread.start()

    def start(self):
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self.miner is not None:
            print('killing miner...')
            p = psutil.Process(self.miner.pid)
            for child in p.children(recursive=True):
                child.terminate()
            p.terminate()
            self.miner = None

    def mine(self):
        with open('./miner.log', 'w') as f:
            while True:
                if self.enabled:
                    if self.miner is None:
                        print('starting miner...')
                        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
                        i = 0
                        self.miner = subprocess.Popen(self.mining_command, shell=True, stdout=f)
                        print('miner pid: ', self.miner.pid)
                        # lets give some time to the process to start :)
                        time.sleep(1)
                    if self.miner is not None:
                        if self.miner.poll() is not None:
                            # miner crashed
                            self.miner = None
                        bar.update(i)
                        i = i + 1
                time.sleep(1)