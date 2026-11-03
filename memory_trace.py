"""Record machine memory while a render is being submitted and monitored."""
import csv
import sys
import threading
import time
from pathlib import Path


class MemoryTrace:
    def __init__(self, run):
        self.path = Path(run) / 'memory-samples.csv'
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def _record(self):
        try:
            with self.path.open('a', buffering=1, newline='') as stream:
                writer = csv.writer(stream)
                if stream.tell() == 0:
                    writer.writerow(['unix_ns', 'monotonic_ns', 'available_kib', 'free_kib', 'cached_kib', 'shmem_kib', 'swap_free_kib', 'memory_full_stall_us'])
                while not self.stop_event.is_set():
                    info = {}
                    for line in Path('/proc/meminfo').read_text().splitlines():
                        key, _, value = line.partition(':')
                        if key in {'MemAvailable', 'MemFree', 'Cached', 'Shmem', 'SwapFree'}:
                            info[key] = int(value.split()[0])
                    if 'MemAvailable' not in info or 'MemFree' not in info:
                        raise ValueError('Required memory counters are missing')
                    stall = ''
                    try:
                        for line in Path('/proc/pressure/memory').read_text().splitlines():
                            if line.startswith('full '):
                                stall = next(v.split('=', 1)[1] for v in line.split() if v.startswith('total='))
                    except (OSError, StopIteration):
                        pass
                    writer.writerow([time.time_ns(), time.monotonic_ns(), info['MemAvailable'], info['MemFree'],
                                     info.get('Cached', ''), info.get('Shmem', ''), info.get('SwapFree', ''), stall])
                    self.stop_event.wait(0.05)
        except (OSError, ValueError) as error:
            print('Memory trace unavailable: ' + str(error), file=sys.stderr, flush=True)


def summarize(path):
    path = Path(path) / 'memory-samples.csv'
    if not path.is_file():
        raise ValueError('No memory trace for this run')
    count = 0
    first = last = minimum = minimum_at = min_free = None
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            timestamp = int(row['unix_ns'])
            available, free = int(row['available_kib']), int(row['free_kib'])
            first = timestamp if first is None else first
            last = timestamp
            if minimum is None or available < minimum:
                minimum, minimum_at = available, timestamp
            min_free = free if min_free is None else min(min_free, free)
            count += 1
    if not count:
        raise ValueError('Memory trace has no samples')
    return {'samples': count, 'duration_seconds': (last - first) / 1e9,
            'min_available_kib': minimum, 'min_available_unix_ns': minimum_at,
            'min_free_kib': min_free, 'trace_path': str(path)}
