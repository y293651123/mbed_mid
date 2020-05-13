import numpy as np
import serial
import time

waitTime = 0.1

# generate the waveform table
signalLength = 106
#t = np.linspace(0, 2*np.pi, signalLength)
signalTable = [

        261, 261, 392, 392, 440, 440, 392,
        349, 349, 330, 330, 294, 294, 261,
        392, 392, 349, 349, 330, 330, 294,
        392, 392, 349, 349, 330, 330, 294,
        261, 261, 392, 392, 440, 440, 392,
        349, 349, 330, 330, 294, 294, 261,

        261, 293, 329, 261, 261, 293, 329, 261, 329, 349, 392, 329, 349, 392, 392, 440, 392, 349,
        329, 261, 392, 440, 392, 349, 329, 261, 293, 392, 261, 261, 293, 392, 261, 440,

        329, 293, 261, 329, 329, 329, 100, 293, 293, 293, 100, 329, 392, 392, 100,
        329, 293, 261, 329, 329, 329, 329, 293, 293, 329, 293, 261, 100, 100, 100]

# output formatter
formatter = lambda x: "%d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
for data in signalTable:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("Signal sended")
