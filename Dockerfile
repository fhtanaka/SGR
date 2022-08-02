FROM python:3.7-slim
WORKDIR /app
RUN apt-get update
RUN apt-get install git sed build-essential -y
RUN git clone --recurse-submodules https://github.com/EvolutionGym/evogym.git
WORKDIR /app/evogym
RUN sed -i -e 's/@.*//g'  requirements.txt
RUN pip install -r requirements.txt
RUN apt-get install  xorg-dev libglu1-mesa-dev  libglew-dev cmake -y
RUN python setup.py install
WORKDIR /app
ADD . /app
RUN pip install dill pathos
CMD ["python", "sgr_main.py"]