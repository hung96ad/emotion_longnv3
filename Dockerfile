FROM debian:latest

RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk vim procps curl cmake

RUN pip3 install opencv-python==3.2.0.8 dlib scikit-learn numpy Flask keras scipy pillow tensorflow pandas h5py statistics pyyaml pyparsing cycler matplotlib

ADD . /face_classifier

WORKDIR face_classifier

ENV PYTHONPATH=$PYTHONPATH:src
ENV FACE_CLASSIFIER_PORT=8084
EXPOSE $FACE_CLASSIFIER_PORT

ENTRYPOINT ["python3"]
CMD ["emotion_recognition.py"]
