just use docker : docker-compose up
otherwise if no docker use below : 

##note, i was not able to host it because model takes alot of ram and server was running out of free tier out of memory 

# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install your dependencies
pip install -r requirements.txt

# When done working:
deactivate

run command 
PORT=5002 USE_GPU=false python app.py


Please remember that processing may take some time, so please wait3-5 min
go to the directory in which you have the file, 
then execute the below command : 
curl -X POST -F "file=@test.mp3" -F "instruments=vocals,drums,bass,other" http://127.0.0.1:5001/separate
you will get a  job-id 

to check the status for the mp3 file
http://127.0.0.1:5002/status/{job-id}

http://127.0.0.1:5001/download/{job-id}
