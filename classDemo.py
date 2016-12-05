import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import argparse
import imutils
from imutils.video import VideoStream
import telepot
import sys, traceback
import time
from pprint import pprint


def handle(msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        print(content_type, chat_type, chat_id)

        if content_type == 'text':
            message = msg['text']
        if (message == '/control'):
                show_keyboard = {'keyboard': [['Picture','Video'], ['Stop','Analytics']]}
                bot.sendMessage(285007767, 'This is a custom keyboard', reply_markup=show_keyboard)
        elif (message == "Picture"):
            filename="A26-CurvedArrow-Orange.png"
            file_id = open(filename,'rb')
            response=bot.sendPhoto(285007767, file_id)
            pprint(response)
            file_id.close()
        elif (message == "Video"):
            bot.sendMessage(chat_id, msg['text'])
            filename="Megamind.avi"
            file_id = open(filename,'rb')
            response=bot.sendVideo(285007767, file_id)
            #response=bot.sendVideo(285007767, u'BAADBAADEAADyp-2Dwqi54nc6OpBAg')
            pprint(response)
            file_id.close()
        else:
            bot.sendMessage(chat_id, "No clue what you meant")


class Timer:
	def __init__(self):
		self.timeLastBeenReset = time.clock()
	def reset(self):
		self.timeLastBeenReset = time.clock()
	def lap(self):
		return time.clock()-self.timeLastBeenReset

class houseCamera:
    def __init__(self, piCamera):
        self.width = 320
        self.minArea = 10
        self.piCamera = piCamera
        self.motionDetect = False
        self.lastBlur = None
        self.firstRun = True

        self.cap = VideoStream(self.piCamera > 0).start()
        time.sleep(2.0)

    def setWidth(self, width):
    	self.width = width

    def grabFrame(self): 
    	frame = self.cap.read()
    	frame=cv2.resize(frame, (self.width,int(self.width*frame.shape[0]/frame.shape[1])))
    	return frame

    def motionDetected(self):
		# Our operations on the frame come here
   		frame = self.cap.read()
   		frame=cv2.resize(frame, (self.width,int(self.width*frame.shape[0]/frame.shape[1])))		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(25,25),0)
		if (self.firstRun):
			self.lastBlur = blur
			self.firstRun = False

		frameDelta = cv2.absdiff(blur, self.lastBlur)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)

		if (self.piCamera):
			(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
		else:
			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)


		motion = 0
		
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < self.minArea:
				continue

			motion = motion + 1
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		self.lastBlur = blur
		#print motion
		if (motion > 0):
			return True
		else:
			return False






camera = houseCamera(False)
camera.setWidth(640)

bot = telepot.Bot('263626698:AAEfH7dm6M_fTzi603pLsaSXGzmoHJSshRk')
bot.message_loop(handle)

timer = Timer()

while True:
	if (camera.motionDetected()):
		cv2.imshow('frame',camera.grabFrame())
		if (timer.lap() > 3):
			print "Time done, resseting"
			print timer.lap()
			timer.reset()
			cv2.imwrite('image.png',camera.grabFrame())
			file_id = open('image.png','rb')
			response=bot.sendPhoto(285007767, file_id)
			pprint(response)




	keyPressed = cv2.waitKey(1) & 0xFF
	if keyPressed == ord('q'):
		break



sys.exit(0)


def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id)

    if content_type == 'text':
        bot.sendMessage(chat_id, msg['text'])

bot = telepot.Bot('263626698:AAEfH7dm6M_fTzi603pLsaSXGzmoHJSshRk')
bot.message_loop(handle)
#ID TEL Accenture: 285007767
bot.sendMessage(285007767, 'Motion detection started')

WIDTH = 320
HEIGHT = 240
MINAREA = 10

# import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the cammera sensor to warmup
cap = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

'''
cap = cv2.VideoCapture(0)
cap.set(3,WIDTH)
cap.set(4,HEIGHT)
'''
last = 0

firstRun = True

time.sleep(2.0)
'''
for i in range(10):
	ret, frame = cap.read()
'''
frame = cap.read()
#frame=cv2.resize(frame, (WIDTH,int(HEIGHT*frame.shape[0]/frame.shape[1])))
frame=cv2.resize(frame, (WIDTH,int(WIDTH*frame.shape[0]/frame.shape[1])))

lastMotion = frame

filename = 'output '+time.asctime(time.localtime(time.time()))+'.avi'
print filename

# Define the codec and create VideoWriter object
if (args["picamera"] > 0):
	fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
else:
	fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
	
out = cv2.VideoWriter(filename,fourcc, 20.0, (WIDTH,int(WIDTH*frame.shape[0]/frame.shape[1])))


serie = np.array([])



while(True):
	# Capture frame-by-frame
	frame = cap.read()
	frame=cv2.resize(frame, (WIDTH,int(WIDTH*frame.shape[0]/frame.shape[1])))



	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(25,25),0)
	if (firstRun):
		lastBlur = blur
		current = last;
		firstRun = False

	last = np.mean(blur)
	frameDelta = cv2.absdiff(blur, lastBlur)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)

	if (args["picamera"] > 0):
		(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
	else:
		(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)


	motion = 0
	
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < MINAREA:
			continue

		motion = motion + 1
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	print motion


	if (motion > 0):
		cv2.putText(frame, time.asctime(time.localtime(time.time()) ),(5,HEIGHT-20), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255))
		out.write(frame)
		print time.asctime( time.localtime(time.time()) )
		#bot.sendMessage(285007767, 'Motion Detected')



	if (abs (current-last) > 1): 
		#print time.asctime( time.localtime(time.time()) )
		lastMotion = frame
		#cv2.putText(frame, time.asctime(time.localtime(time.time()) ),(5,220), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255))
		#out.write(frame)


	suma = np.sum(thresh > 1)
	print suma
	serie=np.append(serie,suma)



	current = last

	diffBlur = abs(blur-lastBlur)
	diffBlur[diffBlur < 200]=0

	# Display the resulting frame
	cv2.putText(frame, time.asctime(time.localtime(time.time()) ),(5,HEIGHT-20), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255))

	cv2.imshow('frame',thresh)
	cv2.imshow('original',frame)

	keyPressed = cv2.waitKey(1) & 0xFF

	if keyPressed == ord('q'):
		break
	if keyPressed == ord('p'):
		plt.plot(serie)
		plt.ion()
		plt.show()


	lastBlur = blur

# When everything done, release the capture
#cap.release()
out.release()
cv2.destroyAllWindows()
bot.sendMessage(285007767, 'Finished. Sending Video')
file_id = open(filename,'rb')
response=bot.sendVideo(285007767, file_id)
pprint(response)
