The carla runs as a server client mode.

Open a terminal and python run_carla_server.py. Note that to run the server you cannot run via ssh. You need to physically run at from the desktop. We are using the stable CARLA 0.8.2 version.

The client maneuvering the car agent can be run by executing python client_example.py.
Variations:
python client_example.py -a [This command will run the car in autopilot mode]
python client_example.py -i [This will record data to a folder called _out]

 
