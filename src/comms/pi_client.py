import requests


class PiClient:
    def __init__(self, pi_ip, port=5000):
        self.url = f"http://{pi_ip}:{port}/set_signal"

    def send(self, lane, duration, emergency=False):
        payload = {
            "lane": int(lane),
            "green_duration": int(duration),
            "emergency": bool(emergency)
        }

        try:
            response = requests.post(self.url, json=payload, timeout=2)
            if response.ok:
                print("[PI] Command sent successfully")
                return True
            else:
                print("[PI] Error response:", response.status_code)
                return False
        except requests.exceptions.RequestException:
            print("[PI] Raspberry Pi not reachable")
            return False
