FROM gw000/keras:2.0.6-py2-tf-cpu

COPY ./requirements.txt /
RUN pip install -r /requirements.txt

CMD jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root
