import time
import asyncio

class SimpleThrottle:
    def __init__(self, coro, delay):
        # Initialize the SimpleThrottle instance with a coroutine and a delay
        self.coro = coro
        self.delay = delay
        self.last_call = None
        self.update_task = None
        self.queue_count = 0

    async def _wrapper(self):
        # Internal wrapper function to handle throttling
        if self.queue_count > 0:
            self.queue_count -= 1

        if self.last_call is not None:
            # Calculate elapsed time since the last call
            elapsed_time = time.time() - self.last_call
            if elapsed_time < self.delay:
                # If elapsed time is less than the delay, sleep to throttle
                await asyncio.sleep(self.delay - elapsed_time)

        await self.coro()
        # Update the last call time and reset the update task
        self.last_call = time.time()
        self.update_task = None

        if self.queue_count > 0:
            # If there are queued calls, execute them
            await self.call()

    async def call(self):
        # Method to initiate a call and handle queuing
        if self.update_task is None:
            # If no update task is running, create a new one
            self.update_task = asyncio.ensure_future(self._wrapper())
        else:
            # If an update task is already running, increment the queue count
            self.queue_count = min(self.queue_count + 1, 1)

    async def call_and_wait(self):
        # Method to wait for the completion of the current update task and then initiate a new call
        if self.update_task is not None:
            # If an update task is running, wait for its completion
            await self.update_task
        await self.coro()
