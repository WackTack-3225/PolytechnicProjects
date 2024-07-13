# Introduction
**Project SMARTGUARD** is the final project for my Computer Vision. I worked in a team of 4 and my role was Project Leader

**Project SMARTGUARD** is a Object Detection and Face Recognition system in the context of Workplace health and safety management, with more emphasis on safety. It detects whether PPE is worn on the worker, while it conducts face recognition.

# My Responsibilities
My role in this project was Project Planning & Execution, figuring out the Object Detection Model and code integration.

For project planning, it took about 1 Week of ideation and thinking to produce this outcome with my group members. The workplan was decided in 1 meeting, where we agreed which section we would take but for the specific project nuances it took a week for collective agreement.

## **Initial Workplan & Data Pipeline Planning:**
![WorkPlan and Data Pipeline](/Computer%20Vision/Supporting%20Images/img1.jpg)

## **Individual Component Process and UI planning:**
![WorkPlan and Data Pipeline](/Computer%20Vision/Supporting%20Images/img2.jpg)

In the end, we dropped the warning system as we felt that though extremely relevant, would take up too much time for an additional component in our system as we were mainly graded on our CV techniques.

For Object Detection Model we used YoloV8 as it was best in class and provided almost real time feedback. The only downside being that it was extremely computationally intensive and required a good GPU to process smoothly. I then used the OpenCV library to draw the bounding boxes around the object before passing the work to the rest of my friends to finish.

As for the code integration, I had to combine all the packaged codes together and make sure they worked as intended. There was some instances of trouble, especially in getting the BLOB to work proerly but thats about it. My friends were able to complete their code without much errors so it went quite smoothly on my end.

Furthermore, I also made all the error and event handlings to ensure that our project started up smoothly between members as we packaged and sent each other. In hindsight, we should have collaborated on GitHub or a version control system, but at that point in time we never learnt the absolute benefits github does provide. I also made the automatic database update section after my friend completed the FaceRecognition Section of our code. 

# Details
I censored most if not all the code with ####\. and also general terms to protect the privacy of me and my teammates as the code uses Google Firebase, which does require links and also access certificates to use such services. 

Supporting Documents like the slides for this project and the user manual are found in the supporting material folder. 


# Project Conclusions/Refelction
I believe that all in all, this project is a good representation of what a access control system powered by AI should be, however it does need to be improved in its accuracy and Face Recognition capabilities. As the FaceRecogniton is a open sourced tool (FaceR C+ Library), and a blackbox model to us, it may not be able to detect each individual clearly. Furthermore, the annotation dataset we did had around 1000+ images only and most of it were from public datasets of low quality. We also manually labelled most of the data, of which I was not part of due to my role as planning the project outline and the research on Object Detection models.