"""ZeroMQ transport implementation for Prime-RL process communication."""

import json
from typing import Any, Dict

import zmq


class ZMQPublisher:
    """ZeroMQ Publisher socket for broadcasting data/weights."""

    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")

    def send_json(self, data: Dict[str, Any]):
        msg = json.dumps(data)
        self.socket.send_string(msg)

    def close(self):
        self.socket.close()
        self.context.term()


class ZMQSubscriber:
    """ZeroMQ Subscriber socket for receiving broadcast data."""

    def __init__(self, host: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def recv_json(self, timeout_ms: int = 1000) -> Dict[str, Any] | None:
        if self.socket.poll(timeout=timeout_ms):
            msg = self.socket.recv_string()
            return json.loads(msg)
        return None

    def close(self):
        self.socket.close()
        self.context.term()


class ZMQPushSender:
    """ZeroMQ PUSH socket for queueing tasks/batches."""

    def __init__(self, host: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(f"tcp://{host}:{port}")

    def send_json(self, data: Dict[str, Any]):
        msg = json.dumps(data)
        self.socket.send_string(msg)

    def close(self):
        self.socket.close()
        self.context.term()


class ZMQPullReceiver:
    """ZeroMQ PULL socket for receiving queued tasks/batches."""

    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(f"tcp://*:{port}")

    def recv_json(self, timeout_ms: int = 1000) -> Dict[str, Any] | None:
        if self.socket.poll(timeout=timeout_ms):
            msg = self.socket.recv_string()
            return json.loads(msg)
        return None

    def close(self):
        self.socket.close()
        self.context.term()
