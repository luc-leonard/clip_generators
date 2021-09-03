import subprocess
import threading
import time

import progressbar


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
            self.miner.terminate()
            self.miner = None

    def mine(self):
        while True:
            if self.enabled:
                if self.miner is None:
                    print('starting miner...')
                    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
                    i = 0
                    self.miner = subprocess.Popen(self.mining_command, shell=True, stdout=subprocess.PIPE)
            if self.miner is not None:
                bar.update(i)
                i = i + 1
            time.sleep(1)