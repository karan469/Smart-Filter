FROM python:3.7
COPY . /app
WORKDIR /app
COPY requirements.txt ./requirements.txt

RUN xargs -L 1 pip install < requirements.txt
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html

EXPOSE 5000
COPY . /app

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
