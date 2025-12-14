# src/pi/gpio_ctrl.py
import os
USE_MOCK = os.environ.get('USE_GPIO_MOCK','1') == '1'

if USE_MOCK:
    class GPIOController:
        def __init__(self, mapping):
            self.mapping = mapping
            self.state = {k: False for k in mapping}
            print('[GPIO MOCK] initialized with pins:', mapping)
        def set(self, key, value: bool):
            self.state[key] = bool(value)
            print(f'[GPIO MOCK] {key} ->', 'ON' if value else 'OFF')
        def cleanup(self):
            print('[GPIO MOCK] cleanup')
else:
    import RPi.GPIO as GPIO
    class GPIOController:
        def __init__(self, mapping):
            self.mapping = mapping
            GPIO.setmode(GPIO.BCM)
            for k, pin in mapping.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
        def set(self, key, value: bool):
            GPIO.output(self.mapping[key], GPIO.HIGH if value else GPIO.LOW)
        def cleanup(self):
            GPIO.cleanup()
