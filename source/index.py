import os
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
        
        try:
            x = eval(str(clas_result))
            print()
            print(x)
            print("- X")
        except:
            print()
            print(type(clas_result))
            print("- Class Result Type")
        
        return clas_result
    
    async def textToImage(prompt: str):
        print("Hello World")

    async def objectDetector():
        modelPath = "C:\Users\lenoc\Documents\GitHub\FaceDetector\models\efficientdet_lite0.tflite"

        # Honestly Idk what I'm doing
        baseOptions = mp.tasks.BaseOptions
        detectionResult = mp.tasks.components.containers.detections.DetectionResult
        objectDetector = mp.tasks.vision.ObjectDetector
        objectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        visionRunningMode = mp.tasks.vision.RunningMode

        def print_result(result: detectionResult, output_image: mp.Image, timestamp: int):
            print('Detection result {}'.format(result))

        options = objectDetectorOptions(
            base_options = baseOptions(model_asset_path = modelPath),
            running_mode = visionRunningMode.LIVE_STREAM,
            max_results = 5,
            result_callback = print_result
        )

        with objectDetector.create_from_options(options) as detector:
            return detector

class UI():
    async def main(page: ft.Page):
        page.title = "Demo IA"
        field = ft.Text(width=100)
        fieldText = ft.TextField(width=100)

        async def enterClick(e):
            x: str = await Ai.textToEmotion(fieldText.value)
            print(x)
            field.value = x
            await page.update_async()

        await page.add_async(ft.Row(
                [
                    fieldText,
                    ft.IconButton(ft.icons.ADD_A_PHOTO, on_click=enterClick),
                    field,
                ],
                    vertical_alignment=ft.MainAxisAlignment.CENTER
                )
            )
    
    ft.app(target=main, view=ft.AppView.FLET_APP)

UI()