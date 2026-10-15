"""Regression: quick commands must not overwrite a slow worker's unread RPC."""
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.reduction import ForkingPickler
import multiprocessing as mp
import pickle
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
import test_mtp_cpu as base


write = base.method("nanovllm/engine/model_runner.py", "ModelRunner", "write_shm",
                    {"ForkingPickler": ForkingPickler, "monotonic": monotonic, "sleep": sleep})


def reader(name, event, output):
    memory = SharedMemory(name=name)
    try:
        commands = []
        for _ in range(3):
            if not event.wait(timeout=10):
                raise RuntimeError("Did not receive TP command")
            sleep(.05)  # Deliberately defer consuming the shared payload.
            size = int.from_bytes(memory.buf[:4], "little")
            commands.append(pickle.loads(bytes(memory.buf[4:4 + size])))
            event.clear()
        output.put(commands)
    finally:
        memory.close()


class CommandQueueTests(unittest.TestCase):
    def test_unread_commands_survive_back_to_back_publish(self):
        context = mp.get_context("spawn")
        event, output = context.Event(), context.Queue()
        memory = SharedMemory(create=True, size=4096)
        process = context.Process(target=reader, args=(memory.name, event, output))
        try:
            process.start()
            publisher = SimpleNamespace(world_size=2, rank=0, shm=memory, event=[event])
            for value in (11, 22, 33):
                write(publisher, "release_then_decode", value)
            self.assertEqual(output.get(timeout=10), [["release_then_decode", 11],
                                                      ["release_then_decode", 22],
                                                      ["release_then_decode", 33]])
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            output.close()
            memory.close()
            memory.unlink()


if __name__ == "__main__":
    unittest.main()
