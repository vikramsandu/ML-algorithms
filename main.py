"""
Dockerize your repo.

What is Docker?
==================
Docker is a lightweight virtual machine focused on packaging and running applications.
1. Docker packages the environment and code into a portable container.
2. It maintains reproducibility also easily deploy to the server.

Important commands:
===================
1. Build the docker image: "docker build -t ml_algorithms_image ."
2. Run the docker container: "docker run --rm ml_algorithms_image"
3. Save the docker image into tar: "docker save -o ml_algorithms_image.tar ml_algorithms_image"

Create a DockerHub account:
===========================

"""

if __name__ == "__main__":
    print("Running clustering and dimensionality reduction...")
