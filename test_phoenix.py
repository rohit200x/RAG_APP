import phoenix as px
import time

print("Starting Phoenix...")

px.launch_app(port=7000)

print("Phoenix started")

while True:
    time.sleep(10)