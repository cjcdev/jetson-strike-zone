# Strike Zone Project Notes


- Starting Docker
```
cd jetson-inference
./docker/run.sh -v /home/chris/jetson-strike-zone/:/jetson-inference/jetson-strike-zone
```

## Training 

### Image annotation (Pascal VOC)
Use VCAT 
https://github.com/opencv/cvat
https://forums.developer.nvidia.com/t/successful-training-with-train-ssd-py-using-small-custom-data-set-but-error-on-full-data-set/156921

Install
```
git clone https://github.com/opencv/cvat
cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```
login to https://localhost:8080

1. Upload images, annotate them, export dataset PascalVoc.
2. Extract dataset zip
3. Create labels.txt in base directory of dataset with all the class names in it.
2. Need to create train.txt/test.txt/val.txt allocation files in ImageSets/Main.  defaults.txt is created by VCAT by default with all the names.
3. ImageSets/Main (create yourself)
    (filenames without extensions can be parsed using 
        find * -type f -print | sed ‘s/.[^.]*$//’
    )
    /test.txt  (10% of filenames without extensions)
    /train.txt (80% of filenames without extension)
    /val.text  (10% of filenames without extensions)
    /trainval.txt (combine train and val)


### Run Training

#### Run detection training
```
cd /jetson-interence/jetson-strike-zone/training

python3 /jetson-inference/python/training/detection/ssd/train_ssd.py --dataset-type=voc --data=data/homeplate-voc --model-dir=models/homeplate --batch-size=2 --workers=1 --epoch=200
```

#### Export onnx model
```
cd /jetson-interence/jetson-strike-zone/training

python3 /jetson-inference/python/training/detection/ssd/onnx_export.py --model-dir=models/homeplate
```

#### Test with detectnet
```
cd /jetson-interence/jetson-strike-zone/training

detectnet --model=models/homeplate/ssd-mobilenet.onnx --labels=models/homeplate/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0 rtp://chris-lin-xps15:8554 --threshold=0.3
```


## PoseNet Keypoints:
```
    "keypoints": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "neck"
    ],
```
