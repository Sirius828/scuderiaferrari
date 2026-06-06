#!/usr/bin/env python3
import threading
import queue
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


class RaceControlUi(Node):
    def __init__(self):
        super().__init__('race_control_ui')

        self.emergency_pub = self.create_publisher(Bool, '/race/emergency_stop', 10)
        self.stop_request_sub = self.create_subscription(
            Bool,
            '/perception/stop_request',
            self.stop_request_callback,
            10
        )
        self.enabled_client = self.create_client(SetBool, '/line_follower/set_enabled')

        self.tk_root = tk.Tk()
        self.tk_root.title('Race Control')
        self.tk_root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.status_var = tk.StringVar(value='Standby')
        self.stop_request_var = tk.StringVar(value='Perception stop: false')
        self.service_var = tk.StringVar(value='Line follower service: waiting')

        self._closing = False
        self._autonomous_enabled = False
        self._emergency_active = False
        self._ui_queue = queue.Queue()
        self._build_ui()
        self._tick()

    def _build_ui(self):
        frame = ttk.Frame(self.tk_root, padding=16)
        frame.grid(row=0, column=0, sticky='nsew')
        self.tk_root.columnconfigure(0, weight=1)
        self.tk_root.rowconfigure(0, weight=1)

        ttk.Label(frame, text='Race Control', font=('TkDefaultFont', 18, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 12)
        )
        ttk.Label(frame, textvariable=self.status_var, font=('TkDefaultFont', 14)).grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(0, 8)
        )
        ttk.Label(frame, textvariable=self.stop_request_var).grid(row=2, column=0, columnspan=2, sticky='w')
        ttk.Label(frame, textvariable=self.service_var).grid(row=3, column=0, columnspan=2, sticky='w', pady=(0, 12))

        self.start_button = ttk.Button(frame, text='Start', command=self.on_start)
        self.start_button.grid(row=4, column=0, sticky='ew', padx=(0, 8), ipady=10)

        self.stop_button = ttk.Button(frame, text='Emergency Stop', command=self.on_emergency_stop)
        self.stop_button.grid(row=4, column=1, sticky='ew', padx=(8, 0), ipady=10)

        self.clear_button = ttk.Button(frame, text='Clear Emergency', command=self.on_clear_emergency)
        self.clear_button.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(12, 0), ipady=6)

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def stop_request_callback(self, msg: Bool):
        self.post_ui(lambda: self.stop_request_var.set(f'Perception stop: {str(bool(msg.data)).lower()}'))

    def post_ui(self, fn):
        self._ui_queue.put(fn)

    def process_ui_queue(self):
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            fn()

    def publish_emergency(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.emergency_pub.publish(msg)
        self._emergency_active = bool(active)

    def call_set_enabled(self, enabled: bool):
        if not self.enabled_client.service_is_ready():
            self.enabled_client.wait_for_service(timeout_sec=0.2)
        if not self.enabled_client.service_is_ready():
            self.post_ui(lambda: self.service_var.set('Line follower service: unavailable'))
            return

        req = SetBool.Request()
        req.data = bool(enabled)
        future = self.enabled_client.call_async(req)
        future.add_done_callback(lambda fut: self.on_set_enabled_done(fut, enabled))
        self.service_var.set('Line follower service: request sent')

    def on_set_enabled_done(self, future, enabled: bool):
        try:
            response = future.result()
            def update():
                if response.success:
                    self._autonomous_enabled = bool(enabled)
                    state = 'Running' if enabled else 'Standby'
                    if self._emergency_active:
                        state = 'Emergency Stop'
                    self.status_var.set(state)
                self.service_var.set(f'Line follower service: {response.message}')
            self.post_ui(update)
        except Exception as exc:
            self.post_ui(lambda: self.service_var.set(f'Line follower service error: {exc}'))

    def on_start(self):
        self.publish_emergency(False)
        self.status_var.set('Starting...')
        self.call_set_enabled(True)

    def on_emergency_stop(self):
        self.publish_emergency(True)
        self.status_var.set('Emergency Stop')
        self.call_set_enabled(False)

    def on_clear_emergency(self):
        self.publish_emergency(False)
        self._autonomous_enabled = False
        self.status_var.set('Standby')
        self.call_set_enabled(False)

    def _tick(self):
        if self._closing:
            return
        self.process_ui_queue()
        if self.enabled_client.service_is_ready():
            if 'waiting' in self.service_var.get():
                self.service_var.set('Line follower service: ready')
        else:
            self.service_var.set('Line follower service: waiting')
        self.tk_root.after(200, self._tick)

    def on_close(self):
        self._closing = True
        self.call_set_enabled(False)
        self.publish_emergency(True)
        self.tk_root.after(100, self.tk_root.destroy)

    def run_ui(self):
        self.tk_root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = RaceControlUi()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_ui()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
