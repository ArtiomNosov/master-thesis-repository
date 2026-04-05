# Practical Technical Specification for the Implementation and Experimental Validation of an Intelligent Candidate Ranking Module for an ATS

## 1. Purpose of the document

This technical specification defines the **practical implementation scope** of the master’s thesis devoted to the development and deployment of an intelligent candidate ranking module based on semantic analysis of resumes and job descriptions for an ATS. This document does not describe a literature review; instead, it defines a **concrete engineering plan for implementation, experimentation, integration, and evaluation**.

**Important language requirement:** although the implementation plan is written in English, all **final conclusions, completed-stage summaries, experimental takeaways, and final result interpretations must be written in Russian**.

## 2. Goal of the practical part

It is required to **design, implement, train, test, and integrate** a software module that:

* accepts a job description and a set of resumes / candidate profiles;
* computes the degree of semantic match between a job and a candidate;
* produces a **ranked list of candidates**;
* returns the result in a format suitable for use inside an ATS;
* supports **reproducible experimental evaluation**;
* makes it possible to assess the **practical and economic impact of deployment**.

At the end of this stage, the **summary of what was implemented and achieved must be written in Russian**.

## 3. Object of development

As part of the practical implementation, it is necessary to create a **prototype or working module** for intelligent candidate ranking, including the following subsystems:

1. data preparation subsystem;
2. preprocessing subsystem for job descriptions and resumes;
3. feature extraction and normalization subsystem;
4. semantic job–resume matching model;
5. candidate ranking module;
6. API or another ATS integration interface;
7. logging, testing, and quality evaluation pipeline.

## 4. Practical deliverables to be produced

The outcome of the work must consist of **concrete engineering artifacts rather than theoretical discussion**:

* source code for the data preparation pipeline;
* a prepared and documented dataset of job descriptions and resumes;
* an implemented job–resume matching model;
* an implemented ranking service or ranking module;
* an interface for integration with an ATS or its test contour;
* a set of executed experiments with quality metrics;
* comparison results against baseline methods;
* integration testing results;
* an assessment of practical and economic efficiency.

For each major deliverable, the **result summary and conclusions must be provided in Russian**.

## 5. Scope of practical implementation

Within this thesis, at least **one full working cycle** must be implemented:

1. data acquisition and preparation;
2. model training or fine-tuning;
3. relevance scoring computation;
4. final candidate ranking generation;
5. experimental evaluation;
6. integration into an ATS or a test contour;
7. result analysis.

If production integration into a real ATS is not possible, it is acceptable to implement a **test bench / sandbox contour / pilot contour** that simulates ATS behavior and enables end-to-end validation.

The **final analysis of this full cycle must be written in Russian**.

## 6. Practical development tasks

### 6.1. Data preparation and profiling

It is necessary to practically perform:

* collection of a corpus of job descriptions and resumes from available sources;
* unification of input data formats;
* text cleaning from duplicates, noisy blocks, service fragments, garbage symbols, and irrelevant sections;
* dataset profiling: volume, average document length, domain distribution, language distribution, seniority distribution, role distribution, and skill distribution;
* de-identification or masking of personal data, if present;
* train / validation / test split generation;
* label construction for matching and ranking tasks.

After this stage, the **dataset preparation outcome, key observations, and conclusions must be written in Russian**.

### 6.2. Design and implementation of the preprocessing pipeline

A preprocessing pipeline must be implemented, potentially including:

* text normalization;
* document segmentation into semantic blocks;
* tokenization;
* removal of template and legal boilerplate sections;
* extraction of structured fields from resumes and job descriptions;
* extraction of key skills, technologies, roles, seniority markers, and other relevant entities;
* conversion of features into a unified format for model input.

The **practical conclusions for this stage must be written in Russian**.

### 6.3. Implementation of the semantic matching model

It is necessary to practically implement and validate a model that encodes job descriptions and resumes into comparable representations and computes a semantic matching score.

The following setups are acceptable:

* bi-encoder / dual-encoder architecture;
* siamese encoder architecture;
* sentence embedding model;
* two-stage retrieval + reranking scheme.

Within the practical scope, it is required to:

* choose a base architecture;
* implement training or fine-tuning;
* define input features, loss function, negative sampling strategy, and validation protocol;
* ensure reproducibility of training.

The **results of model selection, training observations, and stage-level conclusions must be written in Russian**.

### 6.4. Implementation of the ranking module

A module must be developed that:

* accepts a job description and a set of candidates, or a candidate pool from an index;
* computes a score for each job–candidate pair;
* produces an ordered top-K candidate list;
* supports filtering by mandatory constraints when required by the ATS scenario;
* returns the result in a machine-readable format.

If a multi-stage architecture is used, the module must be split into:

* **retrieval stage** — fast initial candidate selection;
* **reranking stage** — refined ranking of shortlisted candidates.

The **practical summary of the ranking module implementation must be written in Russian**.

### 6.5. Implementation of the API and integration layer

It is necessary to practically implement an interaction interface between the ranking module and an ATS or a test contour.

At minimum, the following operations must be supported:

* sending a job description to the ranking module;
* sending resumes / candidates to the index or processing layer;
* requesting top-K ranked candidates;
* receiving ranking results and service metadata;
* logging the model version, response time, and request parameters.

The integration format may be implemented as:

* REST API;
* internal service module;
* offline batch job;
* test harness / simulation contour.

The **integration-stage conclusions and practical findings must be written in Russian**.

## 7. Practical experiments that must be conducted

### 7.1. Baseline comparison experiments

It is necessary to compare the developed approach with **at least one simple baseline** and **at least one stronger baseline**.

Possible baselines include:

* keyword-based matching;
* TF-IDF / cosine similarity;
* BM25-like ranking;
* rule-based scoring;
* embedding-only retrieval without additional reranking.

The purpose of this block is to demonstrate that the developed module produces measurable improvements over simpler solutions.

The **comparison conclusions and interpretation of the results must be written in Russian**.

### 7.2. Model architecture experiments

It is necessary to practically investigate the impact of:

* encoder architecture choice;
* fine-tuning regime;
* training set size;
* positive / negative pair generation strategy;
* feature aggregation strategy;
* text-only input vs text + structured features.

This block must show **what exactly drives the final quality**.

The **experimental takeaways for this block must be written in Russian**.

### 7.3. Ranking quality experiments

For the final module, evaluation must be conducted using ranking metrics, at minimum:

* NDCG@K;
* Precision@K;
* Recall@K;
* optionally MRR@K, MAP, HitRate@K.

The evaluation must be performed:

* on the test set;
* under comparable conditions for all baselines and the target model;
* with fixed experiment parameters;
* with results saved in tables and/or logs.

The **interpretation of ranking metrics and the conclusions from this evaluation must be written in Russian**.

### 7.4. Performance and deployability experiments

It is necessary to practically measure:

* module response latency;
* throughput under batch or repeated processing;
* candidate indexing or re-indexing time;
* memory and computational requirements;
* system behavior as the size of the candidate pool grows.

The purpose is to demonstrate that the solution is not only accurate but also **operationally deployable**.

The **performance-related conclusions must be written in Russian**.

### 7.5. Integration experiments

An end-to-end validation of the full workflow must be conducted:

1. job submission;
2. candidate corpus retrieval;
3. rank computation;
4. result return into the ATS or test contour;
5. validation of output format and business logic.

This experiment must confirm that the module can actually be integrated into an applied workflow rather than only executed as an isolated notebook or script.

The **end-to-end integration results and conclusions must be written in Russian**.

### 7.6. Practical ablation experiments

It is desirable to perform an ablation study to separately demonstrate the contribution of:

* preprocessing;
* extraction / normalization block;
* semantic encoder;
* reranking layer;
* structured features;
* filtering and business rules.

The **ablation conclusions must be written in Russian**.

## 8. What must be practically developed

Within the thesis, at least the following software components must be developed:

1. data loading and normalization module;
2. dataset preparation module;
3. text preprocessing module;
4. feature or embedding construction module;
5. model training / fine-tuning module;
6. inference module;
7. top-K ranking module;
8. integration API or test service;
9. quality evaluation and reporting module.

For each implemented component, the **implementation result summary must be written in Russian**.

## 9. What must be practically investigated

The work must investigate:

* whether the semantic model outperforms simple lexical approaches;
* how fine-tuning affects ranking quality;
* how structured features influence performance together with text features;
* how metrics change for different K values;
* how stable the solution is across different job and resume types;
* whether the system is suitable for integration with respect to latency and operational constraints;
* what practical effect deployment has on the candidate screening workflow.

The **final conclusions of this investigation block must be written in Russian**.

## 10. Acceptance criteria for the practical part

The practical part is considered complete if all of the following conditions are satisfied:

1. a working software prototype or module exists;
2. a dataset or experimental corpus has been prepared and documented;
3. a reproducible training and/or inference pipeline has been implemented;
4. ranking quality metrics have been obtained and recorded;
5. comparison against baseline approaches has been performed;
6. integration into an ATS or a test contour has been completed;
7. end-to-end workflow operability has been demonstrated;
8. tables, diagrams, plots, and conclusions based on the experiments have been prepared;
9. practical and economic efficiency has been assessed.

**All acceptance-related conclusions and the final verdict on completion must be written in Russian.**

## 11. Expected output artifacts

By the end of the work, the following artifacts must be prepared:

* project source code;
* architecture description;
* API specification and integration scenario;
* dataset description and data preparation procedure;
* training and experiment run configurations;
* result tables for baselines and the target model;
* ranking quality report;
* integration testing report;
* performance report;
* practical and economic efficiency assessment;
* materials for inclusion in the thesis manuscript and defense.

The **executive summary for these output artifacts must be written in Russian**.

## 12. Recommended structure of the practical part in the thesis

To correctly present the practical results in the thesis manuscript, the practical part should be structured as follows:

1. engineering problem statement;
2. input data description and dataset preparation scheme;
3. implemented pipeline architecture description;
4. model and training setup description;
5. ranking module description;
6. ATS or test contour integration description;
7. experiment program description;
8. quality metric results;
9. performance and integration results;
10. practical impact assessment;
11. conclusions, limitations, and future work.

Even if intermediate drafting is done in English, the **final section conclusions, chapter summaries, and overall practical conclusions must be written in Russian**.

## 13. Final formulation of the practical scope

The essence of the practical part of this master’s thesis is to **not merely describe semantic analysis and ranking methods, but to implement a full applied candidate ranking module for an ATS, perform training and experimental validation, integrate it into a working or test contour, and then demonstrate its applicability using quality, operational, and business-oriented metrics**.

The **final overall summary of what was done, what was achieved, what limitations were found, and what conclusions follow from the practical work must be written in Russian**.
