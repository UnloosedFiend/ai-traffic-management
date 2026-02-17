# Raspberry Pi 4 Model B - Traffic Light Wiring Guide

## System Overview

This guide covers wiring **4 traffic lanes** with **4 LEDs each** (Red, Yellow, Green, Blue) to a **Raspberry Pi 4 Model B (2018)**.

- **Total LEDs**: 16 (4 lanes × 4 colors)
- **GPIO Pins Used**: 16
- **Power**: 3.3V from Pi (with current-limiting resistors)

---

## Components Required

| Component | Quantity | Notes |
|-----------|----------|-------|
| Raspberry Pi 4 Model B | 1 | 2GB/4GB/8GB RAM |
| Red LEDs (5mm) | 4 | One per lane |
| Yellow LEDs (5mm) | 4 | One per lane |
| Green LEDs (5mm) | 4 | One per lane |
| Blue LEDs (5mm) | 4 | Emergency indicator |
| 330Ω Resistors | 16 | One per LED |
| Breadboard | 1-2 | For prototyping |
| Jumper Wires | ~40 | Male-to-female |
| Common Ground Bus | 1 | Or use breadboard rails |

---

## Raspberry Pi 4 GPIO Pinout

```
               3V3  (1) (2)  5V
          GPIO2/SDA (3) (4)  5V
         GPIO3/SCL  (5) (6)  GND
             GPIO4  (7) (8)  GPIO14/TXD
                GND (9) (10) GPIO15/RXD
            GPIO17 (11) (12) GPIO18
            GPIO27 (13) (14) GND
            GPIO22 (15) (16) GPIO23
               3V3 (17) (18) GPIO24
   GPIO10/SPI_MOSI (19) (20) GND
    GPIO9/SPI_MISO (21) (22) GPIO25
   GPIO11/SPI_SCLK (23) (24) GPIO8/CE0
               GND (25) (26) GPIO7/CE1
          GPIO0/ID (27) (28) GPIO1/ID
             GPIO5 (29) (30) GND
             GPIO6 (31) (32) GPIO12
            GPIO13 (33) (34) GND
            GPIO19 (35) (36) GPIO16
            GPIO26 (37) (38) GPIO20
               GND (39) (40) GPIO21
```

---

## GPIO Pin Assignments

### Lane 0 (North)
| Light | GPIO Pin | Physical Pin | Wire Color (suggested) |
|-------|----------|--------------|------------------------|
| RED | GPIO 17 | Pin 11 | Red |
| YELLOW | GPIO 27 | Pin 13 | Yellow |
| GREEN | GPIO 22 | Pin 15 | Green |
| BLUE | GPIO 5 | Pin 29 | Blue |

### Lane 1 (East)
| Light | GPIO Pin | Physical Pin | Wire Color (suggested) |
|-------|----------|--------------|------------------------|
| RED | GPIO 6 | Pin 31 | Red |
| YELLOW | GPIO 13 | Pin 33 | Yellow |
| GREEN | GPIO 19 | Pin 35 | Green |
| BLUE | GPIO 26 | Pin 37 | Blue |

### Lane 2 (South)
| Light | GPIO Pin | Physical Pin | Wire Color (suggested) |
|-------|----------|--------------|------------------------|
| RED | GPIO 12 | Pin 32 | Red |
| YELLOW | GPIO 16 | Pin 36 | Yellow |
| GREEN | GPIO 20 | Pin 38 | Green |
| BLUE | GPIO 21 | Pin 40 | Blue |

### Lane 3 (West)
| Light | GPIO Pin | Physical Pin | Wire Color (suggested) |
|-------|----------|--------------|------------------------|
| RED | GPIO 4 | Pin 7 | Red |
| YELLOW | GPIO 18 | Pin 12 | Yellow |
| GREEN | GPIO 23 | Pin 16 | Green |
| BLUE | GPIO 24 | Pin 18 | Blue |

### Ground Pins (use any)
- Pin 6, 9, 14, 20, 25, 30, 34, 39

---

## Circuit Diagram (ASCII)

```
For EACH LED:

    GPIO Pin ──────[330Ω]──────(+)LED(-)────── GND
                   resistor     anode cathode

LED Polarity:
  - Anode (+): Longer leg, connects to resistor
  - Cathode (-): Shorter leg (flat side), connects to GND
```

### Complete Wiring for Lane 0:

```
                    Raspberry Pi 4
                    ┌─────────────┐
                    │             │
    Pin 11 (GPIO17) │─────[330Ω]──┼──(+) RED LED (-)───┐
                    │             │                     │
    Pin 13 (GPIO27) │─────[330Ω]──┼──(+) YEL LED (-)───┤
                    │             │                     │
    Pin 15 (GPIO22) │─────[330Ω]──┼──(+) GRN LED (-)───┤
                    │             │                     │
    Pin 29 (GPIO5)  │─────[330Ω]──┼──(+) BLU LED (-)───┤
                    │             │                     │
    Pin 6 (GND)     │─────────────┼─────────────────────┘
                    │             │        (common ground)
                    └─────────────┘
```

---

## Breadboard Layout

```
        Lane 0          Lane 1          Lane 2          Lane 3
    ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  R Y G B  │   │  R Y G B  │   │  R Y G B  │   │  R Y G B  │
    │  ○ ○ ○ ○  │   │  ○ ○ ○ ○  │   │  ○ ○ ○ ○  │   │  ○ ○ ○ ○  │
    │  │ │ │ │  │   │  │ │ │ │  │   │  │ │ │ │  │   │  │ │ │ │  │
    │ [R][R][R][R]  │ [R][R][R][R]  │ [R][R][R][R]  │ [R][R][R][R]
    │  │ │ │ │  │   │  │ │ │ │  │   │  │ │ │ │  │   │  │ │ │ │  │
    └──┼─┼─┼─┼──┘   └──┼─┼─┼─┼──┘   └──┼─┼─┼─┼──┘   └──┼─┼─┼─┼──┘
       │ │ │ │         │ │ │ │         │ │ │ │         │ │ │ │
       │ │ │ │         │ │ │ │         │ │ │ │         │ │ │ │
    To Raspberry Pi GPIO Pins (see table above)
    
    [R] = 330Ω Resistor
    ○ = LED (R=Red, Y=Yellow, G=Green, B=Blue)
    
    All LED cathodes (-) connect to common GND rail
```

---

## Physical Wiring Steps

### Step 1: Prepare the Breadboard
1. Place 4 LEDs per lane (16 total) on the breadboard
2. Keep cathodes (short leg, flat side) on the same row for easy grounding

### Step 2: Connect Resistors
1. Connect a 330Ω resistor to each LED's anode (long leg)
2. The other end of each resistor goes to a jumper wire

### Step 3: Wire to GPIO
Connect jumper wires from resistors to GPIO pins:

| Lane | RED | YELLOW | GREEN | BLUE |
|------|-----|--------|-------|------|
| 0 | Pin 11 | Pin 13 | Pin 15 | Pin 29 |
| 1 | Pin 31 | Pin 33 | Pin 35 | Pin 37 |
| 2 | Pin 32 | Pin 36 | Pin 38 | Pin 40 |
| 3 | Pin 7 | Pin 12 | Pin 16 | Pin 18 |

### Step 4: Connect Ground
1. Connect all LED cathodes to a common ground rail
2. Connect the ground rail to Pi GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)

---

## Software Configuration

Update `src/pi/gpio_ctrl.py` with the correct pin mapping:

```python
# GPIO BCM pin mapping for 4 lanes × 4 lights
PIN_MAP = {
    # Lane 0 (North)
    'lane0_red': 17,
    'lane0_yellow': 27,
    'lane0_green': 22,
    'lane0_blue': 5,
    
    # Lane 1 (East)
    'lane1_red': 6,
    'lane1_yellow': 13,
    'lane1_green': 19,
    'lane1_blue': 26,
    
    # Lane 2 (South)
    'lane2_red': 12,
    'lane2_yellow': 16,
    'lane2_green': 20,
    'lane2_blue': 21,
    
    # Lane 3 (West)
    'lane3_red': 4,
    'lane3_yellow': 18,
    'lane3_green': 23,
    'lane3_blue': 24,
}
```

---

## Testing the Circuit

### Quick Test Script

Save as `test_leds.py` on Raspberry Pi:

```python
#!/usr/bin/env python3
"""Test all 16 LEDs one by one"""
import RPi.GPIO as GPIO
import time

# All GPIO pins used
PINS = {
    'Lane0_RED': 17, 'Lane0_YEL': 27, 'Lane0_GRN': 22, 'Lane0_BLU': 5,
    'Lane1_RED': 6,  'Lane1_YEL': 13, 'Lane1_GRN': 19, 'Lane1_BLU': 26,
    'Lane2_RED': 12, 'Lane2_YEL': 16, 'Lane2_GRN': 20, 'Lane2_BLU': 21,
    'Lane3_RED': 4,  'Lane3_YEL': 18, 'Lane3_GRN': 23, 'Lane3_BLU': 24,
}

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup all pins as output
for name, pin in PINS.items():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

print("Testing all LEDs...")
try:
    for name, pin in PINS.items():
        print(f"  {name} (GPIO {pin})")
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(pin, GPIO.LOW)
    
    print("\nAll LEDs tested! Press Ctrl+C to exit.")
    
    # Blink all green lights
    while True:
        for lane in range(4):
            GPIO.output(PINS[f'Lane{lane}_GRN'], GPIO.HIGH)
        time.sleep(0.5)
        for lane in range(4):
            GPIO.output(PINS[f'Lane{lane}_GRN'], GPIO.LOW)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    GPIO.cleanup()
```

Run with:
```bash
python3 test_leds.py
```

---

## Traffic Light States

### Normal Operation
| State | Lane Active | Active Lane | Other Lanes |
|-------|-------------|-------------|-------------|
| GREEN | Lane N | 🟢 GREEN | 🔴 RED |
| YELLOW | Transition | 🟡 YELLOW | 🔴 RED |
| RED | Next lane | 🔴 RED | (next gets green) |

### Emergency Mode
| State | Emergency Lane | Other Lanes |
|-------|----------------|-------------|
| EMERGENCY | 🟢 GREEN | 🔴 RED + 🔵 BLUE |

The BLUE light indicates to drivers: "An emergency vehicle is approaching from another direction"

---

## Troubleshooting

### LED Not Lighting
1. Check polarity (long leg = anode = to resistor)
2. Verify GPIO pin number matches code
3. Check resistor connections
4. Test LED with 3.3V directly

### Dim LED
1. Reduce resistor value (try 220Ω)
2. Check for loose connections

### GPIO Error
```bash
# Check if GPIO is accessible
ls -la /dev/gpiomem

# Add user to gpio group
sudo usermod -a -G gpio $USER
```

### Multiple LEDs Flickering
1. Ensure adequate power supply (5V 3A recommended)
2. Don't power too many LEDs from 3.3V rail
3. Consider using a separate 5V supply with transistors

---

## Power Considerations

### Direct GPIO (Current Setup)
- Max 16mA per GPIO pin
- Max 50mA total from 3.3V rail
- With 330Ω resistors: ~10mA per LED ✓

### For Brighter LEDs (Optional)
Use transistors (2N2222) or MOSFETs:
```
GPIO ──[1kΩ]──┬── Base (2N2222)
              │
              ├── Emitter ──── GND
              │
              └── Collector ──(LED)──[100Ω]── 5V
```

---

## Safety Notes

1. **Always use resistors** - Never connect LEDs directly to GPIO
2. **Double-check wiring** before powering on
3. **Use 3.3V logic** - Pi GPIO is NOT 5V tolerant
4. **Disconnect power** when modifying circuit
5. **Static discharge** - Ground yourself before handling Pi

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────┐
│          TRAFFIC LIGHT GPIO QUICK REFERENCE            │
├──────────┬───────┬────────┬───────┬───────┬───────────┤
│   LANE   │  RED  │ YELLOW │ GREEN │ BLUE  │    GND    │
├──────────┼───────┼────────┼───────┼───────┼───────────┤
│ Lane 0   │ GPIO17│ GPIO27 │ GPIO22│ GPIO5 │ Pin 6,9,  │
│ (North)  │ Pin 11│ Pin 13 │ Pin 15│ Pin 29│ 14,20,25, │
├──────────┼───────┼────────┼───────┼───────┤ 30,34,39  │
│ Lane 1   │ GPIO6 │ GPIO13 │ GPIO19│ GPIO26│           │
│ (East)   │ Pin 31│ Pin 33 │ Pin 35│ Pin 37│           │
├──────────┼───────┼────────┼───────┼───────┤           │
│ Lane 2   │ GPIO12│ GPIO16 │ GPIO20│ GPIO21│           │
│ (South)  │ Pin 32│ Pin 36 │ Pin 38│ Pin 40│           │
├──────────┼───────┼────────┼───────┼───────┤           │
│ Lane 3   │ GPIO4 │ GPIO18 │ GPIO23│ GPIO24│           │
│ (West)   │ Pin 7 │ Pin 12 │ Pin 16│ Pin 18│           │
└──────────┴───────┴────────┴───────┴───────┴───────────┘
```
