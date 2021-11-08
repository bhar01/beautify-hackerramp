import cv2
import numpy as np
import dlib
from imutils import face_utils, translate
from PIL import Image, ImageDraw
import face_recognition


class Camera(object):
	def __init__(self):
		self.camera = cv2.VideoCapture(0)

		p = "/Users/aishwarya/PycharmProjects/flaskProject3/shape_predictor_68_face_landmarks.dat"
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(p)
		self.effect = "contours"
		self.r=100
		self.g=0
		self.b=0


	def __del__(self):
		self.camera.release()


	def return_jpg(self, frame):
		ret, jpeg = cv2.imencode('.jpeg', frame)
		return jpeg.tobytes()

	def return_effect(self):
		frame =self.create_effect()
		return frame


	def create_effect(self):



		ret, frame = self.camera.read()
		if not ret:
			print("no")
			return False




		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = self.detector(gray)
		imgColorLips=frame.copy()
		face_landmarks_list = face_recognition.face_landmarks(frame)
		for face_landmarks in face_landmarks_list:
			rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(rgb_img)
			draw = ImageDraw.Draw(pil_img, 'RGBA')
			draw.polygon(face_landmarks['top_lip'], fill=(self.r, self.g, self.b, 100))
			draw.polygon(face_landmarks['bottom_lip'], fill=(self.r, self.g, self.b, 100))
			imgColorLips = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

		return self.return_jpg(imgColorLips)


		'''for face in faces:
			x1, y1 = face.left(), face.top()
			x2, y2 = face.right(), face.bottom()
			landmarks = self.predictor(gray, face)
			myPoints = []
			for n in range(68):
				x = landmarks.part(n).x
				y = landmarks.part(n).y
				myPoints.append([x, y])
			myPoints = np.array(myPoints)


			mask = np.zeros_like(frame)
			points=myPoints[49:68]
			rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(rgb_img)
			draw = ImageDraw.Draw(pil_img, 'RGBA')
			draw.polygon(points, fill=(self.r, self.g , self.b, 100))
			imgColorLips1 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
			mask = cv2.fillPoly(mask, [points], (255, 255, 255))
			imgColorLips1 = np.zeros_like(mask)
			imgColorLips1[:] = self.b, self.g, self.r
			imgColorLips1 = cv2.bitwise_and(mask, imgColorLips1)
			imgColorLips1 = cv2.GaussianBlur(imgColorLips1, (7, 7), 10)
			for i in range(0, len(imgColorLips1)):
				if(imgColorLips1[i][0]!=0 and imgColorLips1[i][1]!=0 and imgColorLips1[i][2]!=0):
					frame[i][0]=imgColorLips1[i][0]
					frame[i][1] = imgColorLips1[i][1]
					frame[i][2] = imgColorLips1[i][2]'''


			#imgColorLips1= cv2.addWeighted(imgColorLips1, 0.75, frame, 1, 1)


