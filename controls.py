import time
import pydirectinput

pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.0

class KeyController:
    def __init__(self, up="w", down="s", left="a", right="d"):
        self.up, self.down, self.left, self.right = up, down, left, right
        self._held = set()

    def release_all(self):
        for k in list(self._held):
            pydirectinput.keyUp(k)
        self._held.clear()

    def hold(self, keys):
        keys = set(keys)
        for k in list(self._held - keys):
            pydirectinput.keyUp(k)
            self._held.remove(k)
        for k in list(keys - self._held):
            pydirectinput.keyDown(k)
            self._held.add(k)

    def tap(self, key, duration=0.02):
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)
