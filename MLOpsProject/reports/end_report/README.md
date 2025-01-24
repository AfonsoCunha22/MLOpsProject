# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [X] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [X] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [X] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [X] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [X] Check how robust your model is towards data drifting (M27)
* [X] Deploy to the cloud a drift detection API (M27)
* [X] Instrument your API with a couple of system metrics (M28)
* [X] Setup cloud monitoring of your instrumented application (M28)
* [X] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [X] Write some documentation for your application (M32)
* [X] Publish the documentation to GitHub Pages (M32)
* [X] Revisit your initial project description. Did the project turn out as you wanted?
* [X] Create an architectural diagram over your MLOps pipeline
* [X] Make sure all group members have an understanding about all parts of the project
* [X] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:
Group 39

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

*s233564, s233177, s233185, s233162*

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We decided to use the Hugging Face Transformers package, which wasn't covered in our course, to fulfill the project's requirements. This framework let us use pre-trained models for sentiment analysis, which sped up our development process significantly. By using Hugging Face, we quickly built a strong baseline and could then focus on fine-tuning and optimizing our models. This allowed us to concentrate on other course objectives without spending too much time on building models from scratch. The tools and resources provided by Hugging Face were crucial in helping us complete the project successfully, enhancing our model's accuracy and effectiveness.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used 'requirements.txt' and 'requirements_dev.txt' files to manage our project dependencies. Whenever we used a new package, we checked its version and included it in the appropriate file.
To get a complete copy of our development environment, a new team member would need to follow these steps:
Create a new virtual environment using:

```bash
conda create -n myenv python=3.12.8`
```

Then Activate the newly created environment:

```bash
conda activate myenv
```

And as a last step install the required dependencies by running

```bash
pip install -r requirements.txt
```

and

```bash
pip install -r requirements_dev.txt
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter template, we filled out the configs, data, docs, models, notebooks, reports, src, tests, and wandb folders. We added a .dvc folder for data version control and an onnx_deployment folder for creating, testing, and deploying APIs for ONNX models. Additionally, we included a front_end folder and a subfolder in reports to work on the final report. The wandb folder was added for files related to Weights & Biases. We also used the .github folder for GitHub workflows and actions, and a credentials folder for storing sensitive information. Some other files were placed outside of the overall structure due to a wrongly set root of the project when we started. We did not use the dockerfiles folder due to some issues we encountered, so we decided to keep them where they worked. We also did not use the notebooks folder, although in retrospect, we realize that organizing our code there could have made it clearer. Overall, our structure could be much improved, but we only realized the importance of it closer to the end of the project.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We made sure our code was clear, easy to read, and well-organized. We tried to use meaningful names for files, functions, and variables to make the purpose of each part obvious. We included good comments to explain important parts of the code without overloading it. Additionally, we added terminal commands in the comments needed to run the code, so we could quickly execute it when needed. We handled errors carefully to keep the application stable and avoid unexpected crashes. We also used Ruff for linting to maintain code quality and consistency. This approach made it easier for us to cooperate and ensured the code was easier to understand, debug, and maintain, especially when working in a team or on larger projects. Keeping the structure clean and readable helped ensure that future updates or new team members could quickly adapt to the project.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We have implemented a comprehensive set of tests using Pytest to ensure the reliability of our project. These tests cover various components, including data preprocessing, model functionality, and API endpoints. Specifically, we test data integrity, model initialization, forward pass, prediction, training, evaluation, and model saving/loading. Additionally, we use Ruff for linting and formatting, and pre-commit hooks to maintain code quality. Our GitHub Actions workflow runs these tests on multiple operating systems and Python versions, ensuring cross-platform compatibility and robustness.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our code is 61%, which includes all our source code. Our pytests cover various parts of the project, including the API, data processing, and model functions. Running ```coverage run --rcfile=MLOpsProject/.coveragerc -m pytest MLOpsProject/tests``` shows that all tests passed, with some files having 100% coverage and others less. While coverage gives a lot of insights about the code, it is not a guarantee that the code would be bug-free. This is because it is just a measure of how many lines of code are run when your tests are executed. However, it helps in identifying untested parts of the codebase and ensures that the most critical paths are covered, but it does not eliminate the possibility of bugs entirely.


### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:
We used different branches in our project, trying to keep it in a way that everybody would work on a separate branch to avoid conflicts. However, we did not use pull requests throughout the project. In retrospect, we see that using pull requests would have been valuable and could have saved us a lot of time, especially when we had to track who made changes that influenced a file. The reason we decided not to use pull requests was that we thought we would not interfere with each other's code as much, especially since we often worked on one computer. Additionally, our pull requests were often small, and we felt that having to approve each other's pull requests would prolong the work.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer:

We used DVC to manage data versioning in our project. It helped us track changes to datasets and ensured consistency between data and models. By storing data in remote storage and tracking it in the repository, we saved space and maintained a clear history. DVC made collaboration easier and ensured our pipeline was reproducible and reliable.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have implemented a robust continuous integration (CI) setup to ensure code quality, reliability, and consistency throughout the development lifecycle. Our CI pipeline is designed to automatically trigger workflows under specific conditions, such as changes to the data, model, or codebase.

In our project we focused in mainly three components:
- **Unit Testing**: We have set up unit tests to validate the functionality of our code. These tests are automatically run whenever changes are pushed, ensuring that no new commits break existing functionality.
- **Linting and Pre-commit Hooks**: Pre-commit hooks have been integrated into our repository to enforce code formatting and linting standards before commits are made.
- **Workflow Automation**: A dedicated workflow is triggered whenever changes to the data or model are detected. This workflow is responsible for running the updated data pipeline, ensuring all dependencies are tested thoroughly and checks statistics of the used data in order to ensure it's integrity.

Our workflows are configured using GitHub Actions, and we have implemented caching mechanisms to reduce build times and optimize performance. We currently test on a multiple operating system and , however a single Python version, nevertheless the setup is extensible to support additional configurations in the future.

Here it's an example to the data integrity workflow run: [Workflow Triggered](https://github.com/AfonsoCunha22/MLOpsProject/actions/runs/12912022248/job/36005827665)

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run an experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured our experiments using a *config.yaml* file located under *src/sentiment_analysis/conf/config.yaml*. This file centralizes all experiment settings, such as model parameters, training configurations, and data paths. To manage these configurations effectively, we utilized Hydra, which allows dynamic overrides and seamless experimentation. To achieve this, we integrated Hydra and Typer to get the most of easy configurability.

This approach enables flexible experimentation while maintaining consistency across multiple configuration setups. Different configurations were tested by creating, switching between multiple YAML files and playing with configuration values.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made sure our experiments can be repeated by using configuration files with Hydra. These files stored all important details like model settings, hyperparameters, and data paths. Every time we ran an experiment, the exact settings were saved so we could easily do it again later. We also used DVC to keep track of different versions of our data and model files. This way, we can go back to any version we need. To repeat an experiment, we just load the right configuration and data version, making everything clear and easy to reproduce.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project, we created multiple Docker images to manage different parts of the workflow effectively:

1. FastAPI Dockerfile: This image was used for inference and deployment of the FastAPI application. It included dependencies such as FastAPI and PyTorch. To run it locally, we used:
docker run -p 8000:8000 fastapi-api:latest.

2. Data Dockerfile: This image was designed for managing data preprocessing tasks. It ensured consistency in how data was processed across different environments.

3. Train Dockerfile: This image was used for training the machine learning model. It contained all necessary libraries and tools for running the training scripts in an isolated environment.

Each Dockerfile was built and tested to ensure smooth functionality, and the images were deployed to Google Cloud Run for scalability and serverless operation. Using Docker allowed us to ensure consistency and reproducibility across all stages of the project.

Here’s a link to one of our Dockerfiles: ***docker file link***.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When we encountered bugs in our experiments, our team used a variety of debugging techniques, depending on the complexity of the problem and our personal preferences. Some team members preferred to use print statements for quick and dirty insights, while others relied on integrated development environments (IDEs) with built-in debugging tools such as breakpoints and step-by-step execution to examine code behavior more thoroughly.

In addition, dealing with versions of different libraries specified in our requirements.txt often led to compatibility issues. Profiling and debugging were critical to identifying how different library versions affected the performance and functionality of our application. By methodically profiling our code, we were able to determine which library updates were causing performance degradation or unexpected behavior, allowing us to either roll back to more stable versions or adapt our code to accommodate new library updates.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

For our project, we used a number of Google Cloud Platform (GCP) services strategically integrated into various tasks and workflows. Here's how we used these services:

- **Google Cloud Source Repositories**: We used this to create a Git repository that was fully integrated with other GCP services, ensuring all team members had access and version control was streamlined (M5).

- **Google Cloud Build**: This service was essential for automatically building our Docker containers whenever there were changes to our repository, which helped with continuous integration and deployment (M21).

- **Google Cloud Storage (GCS)**: We used GCS for data storage, which was linked to our data version control setups. It was critical for securely and efficiently storing large data sets, models, and training artifacts (M21).

- **Vertex AI**: For training models directly in the cloud, with tried to implement Vertex AI which provided managed services for machine learning workflows that streamlined our model development and deployment processes (M21).

- **Google Cloud capabilities**: Used as the backend for our FastAPI application, enabling lightweight, serverless computing that responds to requests on-demand (M23).

- **Google Cloud Run**: An alternative to Cloud Functions for deploying containerized applications, providing flexibility and scalability for our API (M23).

- **Google Cloud Monitoring and Logging**: Integrated into our application to track system metrics and log events, enabling effective monitoring and alerting systems to ensure performance and reliability (M28).

- **Google Cloud Pub/Sub**: Used to handle event-driven interactions within our services, facilitating asynchronous messaging patterns critical to decoupling services that respond to data changes (M19).

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

For our latest project, we leveraged the Google Cloud Platform compute engine and deployed the "instancesentiment" VM instance, an n1-standard-1. This machine type, equipped with an Intel Haswell CPU, provided a balanced mix of compute and memory, ideal for our machine learning model experiments focused on sentiment analysis. It ran a PyTorch environment designed for CPU-based tasks from the pytorch-latest-cpu-v20241224 image, with a 100 GB SCSI persistent disk for adequate data storage.

The instance was connected to the default network in the europe-west1-b region, using an ephemeral external IP for temporary test access. We ensured high security by disabling HTTP and HTTPS traffic and disabling IP forwarding, which restricted the instance to direct communication only. This configuration was essential for efficiently evaluating different preprocessing techniques and model architectures without the need for high performance GPUs.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

Our Bucket contained the processed data that was hashed and pushed, using md5. As can be seen here [Overall bucket](figures/bucket_files.png) and [Processed Hashed Files](figures/hashed_files.png).


### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We attempted to train our model in the cloud using Vertex AI. We created a script to set up the training job which can be found as train_vertex_ai.py, define the required instances, and configure the training pipeline. This included specifying the container image and dependencies, along with setting up the input data and output paths in GCP.

While we successfully created the GCP Engine instances and initiated the process, the training ultimately failed due to connection and configuration issues. These challenges included difficulties in establishing consistent access to the data stored in our GCP Bucket and misconfigurations in the permissions and network settings. Despite troubleshooting and revisiting the setup, we couldn’t resolve these issues within the project timeline.

Although the training didn’t succeed on Vertex AI, we tried to learn and understand it's use and purpose and considered that, if revisited, we would focus on improving cloud permissions and networking settings to ensure a seamless training process.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We created an API for our sentiment analysis model using FastAPI. The API has a /predict/ endpoint that takes text input and returns the sentiment (Negative, Neutral, or Positive) along with probabilities. It also logs the predictions, including the text, predicted class, and timestamp, for future use. We added a /drift/ endpoint to check for data drift using Evidently. This compares the logged predictions with the training data and creates a drift report in HTML format. To monitor performance, we used Prometheus to track things like the number of requests, errors, and how long requests take. This setup makes our API easy to use and keeps the model reliable over time.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

For deployment, we wrapped our model into a FastAPI application and tested it locally using uvicorn to ensure all endpoints, including /predict/ and /drift/, worked as expected. After testing, we containerized the application with Docker and deployed it to Google Cloud Run, a serverless platform that simplified the deployment process by managing infrastructure for us.

The deployment steps involved building the Docker image, pushing it to Google Container Registry (GCR), and using gcloud commands to deploy the container to Cloud Run. Once deployed, Cloud Run provided us with a unique URL to invoke the endpoints.

To test the endpoints, we used PowerShell commands like Invoke-RestMethod for both /predict/ and /drift/ endpoints. For example, we sent HTTP POST requests to the /predict/ endpoint to get sentiment predictions and GET requests to /drift/ to generate and check data drift reports. Cloud Run provided reliable scalability and a smooth deployment experience for our application.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Yes, we managed to implement monitoring for our deployed model. For this, we utilized a combination of tools. We used Prometheus Client to expose metrics directly from our application, enabling us to track key performance indicators such as latency, request counts, and error rates. Additionally, we created custom reports to monitor specific metrics relevant to our application, such as prediction accuracy and input data drift.

However, the most intuitive and user-friendly solution was leveraging Google Cloud Monitoring and Alerts. This setup allowed us to create dashboards for visualizing real-time performance metrics and configure alerts for critical thresholds, such as increased latency or a drop in model accuracy. The integration with Google Cloud made it seamless to monitor the deployed model and quickly act on anomalies.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>

>
> Answer:

We used a total of $3.21 on Compute Engine, which was the most expensive service. Minor costs were incurred for Networking ($0.15) and Cloud Storage ($0.05), but these were offset by promotions or discounts. Working in the cloud allowed us to scale our project efficiently and focus on development without worrying about infrastructure. It was a cost-effective way to manage resources and deploy applications.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We focused on meeting the project requirements and did not implement any additional features beyond what was specified. Our efforts were concentrated on ensuring high-quality execution of the required tasks

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>

> Answer:

The biggest challenge in our project was setting up dependencies and resolving library conflicts, which delayed progress initially. We carefully updated the requirements.txt file, used virtual environments to isolate dependencies, and documented the setup process for efficiency. Learning new tools like DVC and Evidently for data versioning and drift detection was also time-consuming. We overcame these challenges by referring to documentation, collaborating as a team, and troubleshooting together. While the setup phase was time-intensive, it streamlined the rest of the project and ensured smoother progress.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
