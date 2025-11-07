# send_sms.py
import subprocess, time

numbers = ["+447519266843"]
msgs = []
sig = '\n\n\nFrom RawDogs R&D Department'
for x in range(1, 6):
    msgs.append(f'Automated message {x} \n<Proof of concept: cell phone decoy testing> {sig}')

delay = 5

for n in numbers:
    for m in msgs:
        print("Sending to", n)
        print(m)
        subprocess.run(["termux-sms-send", "-n", n, m])
        time.sleep(delay)
