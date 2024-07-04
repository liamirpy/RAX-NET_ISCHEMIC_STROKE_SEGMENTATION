FROM python:3.11-slim

# Set the working directory in the container
WORKDIR .

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME RUX

# Run segmentation model when the container launches
CMD ["python3", "fusion.py"]