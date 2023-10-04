import csv
from flask import Flask, request
from waitress import serve
from threading import Thread
import requests
import os
import json
import time

app = Flask(__name__)

@app.route('/EkinTrafficSystem/services/secondarySpeed/setCalculatedSpeed', methods = ['POST'])
def resultSpeed():
    try:
        if request.data:
            resultMessage = request.get_json()
            print(resultMessage)
            response = {"status":True}
            return response
        else:
            response = {"status":False}
            return response
    except Exception as e:
        print(e)
        return {"status":False}
    

def main():
    requestJson = {}
    violationPaths = "C:/Users/ekin/Documents/Verra/test2"
    #violationPaths = "E:/SpeedTest/Canada/Samples"
    #violationPaths = "C:/Users/ekin/Documents/july13/105/vio1"
    #violationPaths = "E:/SpeedTest/errorTest/"
    #violationPaths = "D:/Downloads/nee/Violation_data_05-25-2023_09_31_26.598_AM"
    #violationPaths = "C:/Users/ekin/Downloads/vio"
    #violationPaths = "C:/Users/ekin/Downloads/vioVerraJuly"
    violations = os.listdir(violationPaths)
    counter = 0
    totalCounter = 0
    
    for violation in violations:
        vio = os.path.join(violationPaths, violation)
        files = os.listdir(vio)
        jsonFile = next(x for x in files if x.endswith(".json"))
        ovipath = next(x for x in files if x.endswith("ovi.jpg"))
        initialPlatePhotoPath = next(x for x in files if x.endswith("i.jpg"))
        videofile = next(x for x in files if x.endswith(".mp4"))
        f = open(os.path.join(vio,jsonFile),)
        data = json.load(f)
        initialsn = data['initialSnapshot']
        speed = "id : " + str(data['id']) + ", radarSpeed : " +str(initialsn['speed'])
        result = open("demofile2.txt", "a")
        result.write(speed)
        result.close()
        print(initialsn['speed'])
        data['evidenceVideo'] = vio + "/" + videofile
        initialsn['overviewImagePath'] = vio + "/" + ovipath
        data['initialSnapshot'] = initialsn
        totalCounter += 1
        print(totalCounter)
        
        if "plateDetail" in data :
            plateDetail = data['plateDetail']
            imageInfo = plateDetail['imageInfo']
            imageInfo['imagePath'] = vio + "/" + initialPlatePhotoPath
            plateDetail['imageInfo'] = imageInfo
            data['plateDetail'] = plateDetail
        else:
            data['plateDetail'] = vio + "/" + initialPlatePhotoPath            
            counter += 1
            print(counter)
            
        # requestJson['radarEvent'] = data
        # requestJson['command'] = "visraAnalyzeSpeedEstimator"
        #json.dump(requestJson)
        command = "visraAnalyzeSpeedEstimator"
        requestJson = {'command':command,
                       'radarEvent': data}


        API_ENDPOINT = "http://127.0.0.1:9595/VisraServer"

        try:
            r = requests.post(url = API_ENDPOINT, json=requestJson)
            print(r.content)
        except:
            print("ERROR: post request")

        time.sleep(30)



if __name__ == '__main__':

    t = Thread(target=main,args=())
    t.start()
    serve(app, host='0.0.0.0', port = 8080)
    