import time
from cv2 import cv2
import sys
from kafka import SimpleProducer, KafkaClient

# Connect to Kafka
kafka = KafkaClient("localhost:9092")
producer = SimpleProducer(kafka)

# Assigning a topic
topic = "Video-Relay"

# Function to relay video
def VideoEmitter(video):
	'''
	Functions that takes a video and breaks it into frames 
	and emits to the consumers
	
	@param:
		video => The video file
	
	@return:
	 	None
	'''
	try:
		video = cv2.VideoCapture(video)
		print("[+] Emitting")

		# Read the file
		while(video.isOpened):
			# Reading images of each frame
			success, image = video.read()
			# Check if file ended
			if not success:
				break
			# Convert image to .jpg
			_, buffer = cv2.imencode('.jpg', image)
			# Convert image to byte stream for sending
			producer.send_messages(topic, buffer.tobytes())
		
	except KeyboardInterrupt:
		print("[+] Relay closed on interrupt")

	except Exception as err:
		print("[+] Unhandled Exception")
		print(err)

	finally:
		# Clear the capture
		video.release()
		print("[+] Relay ended")

def RealTimeFeed():
	'''
	Functions that takes webcam feed and breaks it into frames 
	and emits to the consumers
	
	@param:
		None
	
	@return:
	 	None
	'''
	try:
		camera = cv2.VideoCapture(cv2.CAP_DSHOW)
		print("[+] Emitting Realtime")

		# Read the file
		while(True):
			# Reading images of each frame
			success, frame = camera.read()
			# Check if file ended
			if not success:
				break
			# Convert image to .jpg
			_, buffer = cv2.imencode('.jpg', frame)
			# Convert image to byte stream for sending
			producer.send_messages(topic, buffer.tobytes())
		
	except KeyboardInterrupt:
		print("[+] Realtime Relay closed on interrupt")

	except Exception as err:
		print("[+] Realtime Error: Unhandled Exception")
		print(err)

	finally:
		# Clear the capture
		camera.release()
		print("[+] Realtime Relay ended")

if __name__ == '__main__':
	if(len(sys.argv) > 1):
		video = sys.argv[1]
		VideoEmitter(video)
	else:
		print("[+] Fetching Realtime Feed!")
		RealTimeFeed()