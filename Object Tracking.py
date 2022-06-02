from darkflow.net.build import TFNet
import cv2
import tensorflow as tf
import airsim
import os

# Config TF, set True if using GPU
config = tf.ConfigProto(log_device_placement = False)
config.gpu_options.allow_growth = False

with tf.Session(config=config) as sess:
    options = {
            'model': './cfg/yolo.cfg',
            'load': './yolov2.weights',
            'threshold': 0.6,
            #'gpu': 1.0 # uncomment these if using GPU
               }
    tfnet = TFNet(options)


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
client.moveToPositionAsync(-10, 10, -10, 5).join()

msg=int(input("If you want to work with images press 1, else press 0"))
if(msg==1):
    # take images
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),
        airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
    print('Retrieved images: %d', len(responses))

    #Get RGB Image
    for color in responses:
        imgcolor = np.fromstring(color.image_data_uint8, dtype=np.uint8)
        imgcolor = imgcolor.reshape(responses[1].height, responses[1].width, -1)
        if imgcolor.shape[2] == 4:
            imgcolor = cv2.cvtColor(imgcolor,cv2.COLOR_BGR2RGB)
        results=tfnet.return_predict(img)
        print(results)
        for (i, result) in enumerate(results):
            x = result['topleft']['x']
            w = result['bottomright']['x'] - result['topleft']['x']
            y = result['topleft']['y']
            h = result['bottomright']['y'] - result['topleft']['y']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_position = (x + int(w / 2)), abs(y - 10)
            cv2.putText(img, result['label'], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Objet Detection YOLO", color)
else:
    #Working on a video
    cap = cv2.VideoCapture('all2.mp4')
    frame_number = 0
    while True:
        ret, frame = cap.read()
        frame_number += 1
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tfnet.return_predict(img)

            for (i, result) in enumerate(results):
                x = result['topleft']['x']
                w = result['bottomright']['x'] - result['topleft']['x']
                y = result['topleft']['y']
                h = result['bottomright']['y'] - result['topleft']['y']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_position = (x + int(w / 2)), abs(y - 10)
                cv2.putText(frame, result['label'], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Objet Detection YOLO", frame)
            if frame_number == 240:
                break
            if cv2.waitKey(1) == 10:
                break
    cap.release()
    cv2.destroyAllWindows()