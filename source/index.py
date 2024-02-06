import os
import cv2
import time
import flet as ft
import mediapipe as mp
from mediapipe.tasks.python import text
from mediapipe.tasks import python

class Ai():
    async def textToEmotion(data: str):
        absPath: str = os.path.abspath("index.py")
        modelAbsPath = absPath.replace("index.py", "models\\")
        bertAbsPath = modelAbsPath + "bert_classifier.tflite"

        base_option = python.BaseOptions(model_asset_path=bertAbsPath)
        options = text.TextClassifierOptions(base_options=base_option)

        with python.text.TextClassifier.create_from_options(options) as classifier:
            clas_result: str = classifier.classify(data)
        
        dictionary: dict = {}
        y = 0

        for x in clas_result.classifications[0].categories:
            score = x.score
            cat_name = x.category_name

            dictionary[y] = {'catName': cat_name, 'score': score}
            y+=1

        return dictionary
    
    async def textToImage(prompt: str):
        print("Hello World")

    async def objectDetector(image):
        modelPath = "C:\\Users\\lenoc\\Documents\\GitHub\\FaceDetector\\models\\efficientdet_lite0.tflite"

        # Honestly Idk what I'm doing
        baseOptions = mp.tasks.BaseOptions
        # detectionResult = mp.tasks.components.containers.detections.DetectionResult
        objectDetector = mp.tasks.vision.ObjectDetector
        objectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        visionRunningMode = mp.tasks.vision.RunningMode

        options = objectDetectorOptions(
            base_options=baseOptions(model_asset_path=modelPath),
            max_results=5,
            running_mode=visionRunningMode.IMAGE
            )

        with objectDetector.create_from_options(options) as detector:
            detection_result = detector.detect(image)

        #I need to get EVERY
        dictionary: dict = {}
        x = 0

        for detection in detection_result.detections:
            boundingBox = detection.bounding_box

            x1 = boundingBox.origin_x
            y1 = boundingBox.origin_y
            x2 = boundingBox.width
            y2 = boundingBox.height

            categories = detection.categories
            category = categories[0]
            score = category.score
            categoryName = category.category_name

            dictionary[x] = {"catName": categoryName, "score": score, "X1": x1, "Y1": y1, "X2": x2, "Y2": y2}
            x+=1

        return dictionary

class UI():
    async def main(page: ft.Page):
        await page.update_async() #Constant updated

        page.title = "Demo IA"
        field = ft.Text(width=100)
        fieldText = ft.TextField(width=100)

        myImage = ft.Image(
            src=False,
            width=320,
            height=320,
            fit=ft.ImageFit.COVER
        )

        async def takeCapture(e):
            cap = cv2.VideoCapture(0) #????
            cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL) #???? x2
            cv2.resizeWindow("Webcam", 320, 320)

            try:
                while True:
                    ret, frame = cap.read()

                    cv2.imshow("Webcam", frame)
                    #myImage.src = ""
                    await page.update_async()

                    key = cv2.waitKey(1) # Wait a Key with 1 (???) of delay

                    if key == ord("p"):
                        break
                    elif key == ord("q"):
                        data = await Ai.objectDetector(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                        )

                        for x in data:
                            cv2.rectangle(
                                frame,
                                (int(data[x]['X1']), int(data[x]['Y1'])),
                                (int(data[x]['X1']) + int(data[x]['X2']), int(data[x]['Y1']) + int(data[x]['Y2'])),
                                (246,209,81),
                                2)
                            
                            cv2.putText(frame, data[x]['catName'] + " " + str(round(data[x]['score'] * 100, 0)) + "%", 
                                        (data[x]['X1'], data[x]['Y1'] - 10), 
                                        cv2.QT_FONT_NORMAL, 1, 
                                        (255,209,81), 2)

                            print(data[x])
                            
                        cv2.imshow("Webcam", frame)
                        cv2.waitKey(1000)
                        await page.update_async()

                cap.release() #???? x4
                cv2.destroyAllWindows()
                await page.update_async()

            except Exception as e:
                print("SOMETHING FAILED!!!")
                print(e)

        async def enterClick(e):
            x = await Ai.textToEmotion(fieldText.value)

            p = x[0]['catName']
            pScore = round(x[0]['score'] * 100, 0)
            n = x[1]['catName']
            nScore = round(x[1]['score'] * 100, 0)

            field.value = ("Positive: " + str(pScore) + "%\nNegative: " + str(nScore) + "%")
            await page.update_async()

        await page.add_async(
            ft.Row(
                [
                    ft.Text(value="Object Detector: "),
                    ft.IconButton(ft.icons.ADD_A_PHOTO, on_click=takeCapture),
                    myImage
                ]
                ),
            ft.Row(
                [
                    ft.Text(value="Text to Emotion: "),
                    fieldText,
                    ft.IconButton(icon=ft.icons.ADD_BOX, on_click=enterClick),
                    field
                ]
            )
            )
    
    ft.app(target=main, view=ft.AppView.FLET_APP)

UI()