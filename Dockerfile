FROM python:3.9
COPY ./ rl_waypoint_mrta/
WORKDIR /rl_waypoint_mrta
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
RUN python main.py --help
# RUN python main.py --train