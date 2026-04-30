# 42028: Deep Learning and Convolutional Neural Networks  
## Assignment 2 – Autumn 2026

---

## Assignment Specification

### Due Date
**Friday 11:59 PM, 08 May 2026**

### Demonstrations
Optional, if required.

### Marks
30% of the total marks for this subject.

### Submission

1. A report in PDF or MS Word document (~10 pages)  
   *(Part-B submission)*

2. Google Colab / iPython notebooks  
   *(Part-A submission)*

Submit via **Canvas assignment submission**.

> **Note:**  
> This assignment is an individual work. Your assignment is incomplete without both the Report and Code submission.  
> If you only submit either the code or the report, it will be considered incomplete and will not be marked.  
> Please make sure to submit both the report and code using the appropriate links/pages on Canvas.

---

# Summary

This assessment requires students to customize standard CNN architectures for image classification.

Standard CNNs such as:
- AlexNet
- GoogleNet
- ResNet
- etc.

should be used to create customized versions of the architectures.

Students are also required to implement a CNN architecture for:
- object detection
- localization

Both the customized CNNs:
- image classification
- object detection

should be trained and tested using the provided dataset.

Students must provide:
- Code (iPython / Colab Notebook)
- Final report

The report should:
- outline assumptions/intuitions used to create the customized CNNs
- discuss performance and experimental results

---

# Assignment Objectives

The purpose of this assignment is to demonstrate competence in the following skills:

- Ensure students have a firm understanding of CNNs and object detection algorithms.
- Facilitate learning of advanced research topics.
- Assist students in completing future projects.
- Ensure students can develop custom CNN architectures for different computer vision tasks.

---

# Tasks

## Description

### 1. Image Classification (2 Experiments)

Customize a foundational/baseline architecture, such as:
- AlexNet
- GoogleNet
- ResNet
- etc.

Requirements:
- Reduce or increase convolutional layers
- Train and test the model
- Use the given dataset
- Conduct 2 experiments

---

### 2. Object Detection / Localization

Implement and train/validate/test:
- Faster-RCNN
- SSD / YOLO / RF-DETR

Notes:
- Existing implementations are permitted
- Examples:
  - Google Object Detection API
  - TorchVision

---

### 3. Object Detection Experiments (2 Experiments)

Train and test:
- Faster-RCNN
- SSD / YOLO / RF-DETR

using the provided dataset.

---

# Dataset

Datasets can be downloaded using instructions available on Canvas.

Location:
- Canvas
- Assignment → Assignment-2

---

# Report Requirements

Write a short report describing:
- implementation details
- concepts learned in class
- assumptions/intuitions for customized CNNs
- performance discussion

Additional requirements:
- Include CNN architecture diagrams where necessary
- Include model summaries:
  - input parameters
  - output parameters
  - etc.
- Discuss:
  - results
  - limitations
  - constraints
  - observations

---

# Suggested Report Structure

## 1. Introduction

Include:
- Brief outline of the report
- Baseline CNN architectures used
- Object detection methods used

---

## 2. Dataset

Include:
- Dataset description
- Sample images for each class

---

## 3. Proposed CNN Architecture for Image Classification

### a. Foundational/Baseline Architecture (Experiment 1)

### b. Customized Architecture (Experiment 2)

### c. Assumptions / Intuitions

### d. Model Summary

---

## 4. CNN Architecture for Object Detection / Localization

### a. Faster-RCNN (Experiment 3)

### b. SSD / YOLO / RF-DETR (Experiment 4)

### c. Assumptions / Intuitions

### d. Model Summary

---

## 5. Experimental Results and Discussion

### a. Experimental Settings

#### i. Image Classification

#### ii. Object Detection

---

### b. Experimental Results

#### i. Image Classification

1. Performance on baseline/standard architecture
2. Performance on customized architecture

#### ii. Object Detection

1. Performance on Faster-RCNN
2. Performance on SSD / YOLO / customized architecture

---

### iii. Discussion

Discuss:
- performance
- accuracy
- understanding of results

Optional:
- include wrongly classified image samples

---

## 6. Conclusion

Provide a short summary of:
- experiments
- results
- overall understanding

---

# Deliverables

## a. Project Report
- Maximum 10 pages

## b. Google Colab / iPython Notebook
Requirements:
- Include code
- Output of each code cell must be visible

---

# Important Notes

- You may report accuracy on custom CNNs for object detection instead of SSD/YOLO/RF-DETR.
- Transfer learning is permitted.
- The complete network must still be trained/fine-tuned on the provided dataset.
- Students must only use the dataset assigned to them based on student ID.

---

# Additional Information

## Dataset Generation

Check Canvas instructions for dataset generation details.

---

## Image Classification Dataset

Each student receives:
- 10 unique classes
- images inside each class folder

Important:
- Classes differ depending on student ID
- Dataset is NOT pre-split

Students must:
- split into train/test/validation sets
- use student ID as the random seed

---

## Object Detection Dataset

Each student receives:
- unique train/test/validation sets
- pre-segregated datasets
- three annotation formats

---

## Dataset Consistency

For a given student ID:
- the same dataset will always be generated

Important:
- Only use your assigned dataset
- This will be cross-verified
- Any discrepancy results in:
  - 0 marks for the entire assignment

---

# Assessment Submission

Submission has two parts:

1. iPython / Colab notebooks  
   - zip file if multiple notebooks

2. Report

Submit both to Canvas before the due date.

Notes:
- You may submit multiple times before the deadline
- The final submission is the one that will be marked
- Experimental results in notebooks must match the report

Penalties apply for inconsistencies between:
- notebook results
- report results

---

# Important Notices

## PLEASE NOTE 1

It is your responsibility to thoroughly test your program on:
- AWS SageMaker

---

## PLEASE NOTE 2

Only your final Canvas submission will be marked.

Earlier submissions will be ignored.

Recommendation:
- Download your final submission from Canvas
- Test it thoroughly

---

# Return of Assessed Assignment

Marks are expected to be released:
- within 2 weeks after submission
- via Canvas

---

# Queries

If illness or other issues affect submission:
- contact the subject coordinator ASAP

## Subject Coordinator

**Dr. Nabin Sharma**

- Room: CB11.07.124
- Phone: 9514 1835
- Email: Nabin.Sharma@uts.edu.au

---

# Assignment Questions

If you have assignment questions:
- post them on the Canvas discussion forum

This allows everyone to see the response.

---

# Canvas Announcements

If serious problems are discovered:
- announcements/FAQs will be posted on Canvas

It is your responsibility to check Canvas regularly.

---

# PLEASE NOTE

If answers can already be found in:
- Subject information
- Assignment specification
- Canvas FAQ
- Assignment 2 FAQ
- Canvas discussion board

you will be redirected there instead of receiving a direct answer.