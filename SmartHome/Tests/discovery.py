import time
from zeroconf import ServiceBrowser, Zeroconf
import requests
from bs4 import BeautifulSoup

class MyListener:
    def remove_service(self, zeroconf, type, name):
        print(f"Service {name} removed")

    def add_service(self, zeroconf, type, name):
        if "shelly" in name.lower():
            info = zeroconf.get_service_info(type, name)
            if info:
                address = info.parsed_addresses()[0]
                port = info.port
                print(f"Service {name} added, address {address}, port {port}")

                # Beispiel-URL, die den HTML-Code enthält
                url = f"http://{address}"

                # Wartezeit, um sicherzustellen, dass der Display-Name übermittelt wurde
                time.sleep(5)

                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        device_name = soup.find('div', id='device_name').h1.text
                        print(f"Device Name: {device_name}")

                        # Warte 3 Sekunden bevor der Inhalt geschrieben wird
                        time.sleep(3)

                        with open("shelly.txt", "w", encoding="utf-8") as file:
                            file.write(response.text)
                    else:
                        print(f"Failed to retrieve HTML from {url}")
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching HTML: {e}")

zeroconf = Zeroconf()
listener = MyListener()
browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)

try:
    input("Press enter to exit...\n\n")
finally:
    zeroconf.close()
