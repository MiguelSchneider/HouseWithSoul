import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import argparse
import imutils
from imutils.video import VideoStream
import telepot
import sys, traceback
from pprint import pprint
import json
import logging



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
		self._motionDetect = False
		self.lastBlur = None
		self.lastFrame = None
		self.firstRun = True

		if (self.piCamera == True):
			self.cap = VideoStream(usePiCamera=True).start()
			print ("RASPI CAMERA")
		else:
			self.cap = VideoStream(0).start()
		time.sleep(2.0)

	def setWidth(self, width):
		self.width = width

	def grabFrame(self): 
		return self.lastFrame

	def motionDetection(self):
		motion = 0
		cnts = None
		frame = self.cap.read()
		self.lastFrame = frame

		frame=cv2.resize(frame, (self.width,int(self.width*frame.shape[0]/frame.shape[1])))		

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(25,25),0)
		if (self.firstRun):
			self.lastBlur = blur
			self.firstRun = False

		frameDelta = cv2.absdiff(blur, self.lastBlur)
		self.lastBlur = blur
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)

		(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
		'''
		if (self.piCamera):
			(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
		else:
			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
		'''	
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < self.minArea:
				continue
			motion = motion + 1
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		

		_motion = False
	
		if (motion>0):
			_motion = True

		return (_motion, frame, cnts)



	def process(self):
		(motion, frame, cnts) = self.motionDetection();
		return (motion, frame, cnts)



class House:
	def __init__(self, config_file_name):


		with open(config_file_name) as data_file:    
			data = json.load(data_file)

		self.camera = houseCamera(data["Camera"]["piCamera"])
		self.camera.setWidth(data["Camera"]["width"])
		
		self.chatBot = telepot.Bot('263626698:AAEfH7dm6M_fTzi603pLsaSXGzmoHJSshRk')
		self.chatBot.message_loop(self.botHandle)

		self.acceptedIDs=data["Telegram"]["AcceptedIDs"]

		self.timer = Timer()

		self.sendMotionDetectionMessage = data["House"]["motionDetectMessage"]



	def botHandle(self,msg):
		content_type, chat_type, chat_id = telepot.glance(msg)
		print(content_type, chat_type, chat_id)

		if (chat_id in self.acceptedIDs):
			if content_type == 'text':
				message = msg['text']
			if (message == '/control'):
					show_keyboard = {'keyboard': [['Picture','Motion Detection'], ['Stop','Analytics']]}
					self.chatBot.sendMessage(chat_id, 'This is a custom keyboard', reply_markup=show_keyboard)
			elif (message == "Picture"):
				#cv2.imwrite('image.png',self.camera.grabFrame())
				cv2.imwrite('image.png',self.camera.grabFrame())
				file_id = open('image.png','rb')
				response=self.chatBot.sendPhoto(chat_id, file_id)
				pprint(response)
				file_id.close()
			elif (message == "Motion Detection"):
				if (self.sendMotionDetectionMessage):
					self.sendMotionDetectionMessage=False
					self.chatBot.sendMessage(chat_id, "Stopped detecting motion")
				else:
					self.sendMotionDetectionMessage=True
					self.chatBot.sendMessage(chat_id, "Started detecting motion")
			else:
				self.chatBot.sendMessage(chat_id, "No clue what you meant")
		else:
			self.chatBot.sendMessage(chat_id, "You are not accepted in this chat")


	def run(self):
		motionSaveTimer = Timer()

		moved = 0
		while True:
			(motion, frame, cnts)=self.camera.process()
			if (motion):
				moved = moved + 1
				if ( (self.timer.lap() > 3) and self.sendMotionDetectionMessage):
					self.timer.reset()
					cv2.imwrite('image.png',frame)
					file_id = open('image.png','rb')
					for sendToID in self.acceptedIDs:
						response=self.chatBot.sendPhoto(sendToID, file_id)
						pprint(response)
			if (motionSaveTimer.lap() > 3):
				logFileMotion = open ('motionLog.log','a')
				savingStr=time.asctime(time.localtime(time.time()))+'   '+str(moved)+ '    '+str(round(np.mean(frame)))   +'\n'
				logFileMotion.write(savingStr)
				motionSaveTimer.reset()
				moved = 0



			'''
			cv2.imshow('frame',self.camera.grabFrame())
			keyPressed = cv2.waitKey(1) & 0xFF
			if keyPressed == ord('q'):
				break
			'''



if __name__ == "__main__":
	myHouse = House("config.json")
	myHouse.run()
