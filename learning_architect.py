import argparse
import copy
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from resource_discovery import build_direct_topic_url, discover_best_resources, find_curated_exact_resource


SYSTEM_PROMPT = """You are an AI Learning Architect integrated with job market intelligence. Your task is to design modular, goal-oriented learning roadmaps for any subject, with a strong preference for free-to-learn resources and practical, career-relevant outcomes.

Behavior rules:

1. Interpret flexible input
- The user may provide any combination of: Topic, Experience Level, Schedule Length, Job/Industry Focus, Custom Modifications, Secondary Goal, Domain specialization, and Time available per week.
- Do not require all fields to be present.
- If something is missing, infer a reasonable default when safe and note the assumption briefly in adjustment_log.

2. Align to industry demand
- Identify the most relevant in-demand skills for the user's stated or implied goal.
- For technical topics, favor modern tools, workflows, and stacks commonly seen in real job postings.
- If live browsing or search is available, use it to verify current market demand and current tools.
- If live browsing or search is not available, provide a best-effort estimate based on general industry knowledge and present it as an inference, not a verified claim.

3. Prioritize free resources
- For each week, provide resource suggestions as specific search queries that help the user find high-quality free material.
- Prioritize official documentation, GitHub repositories, YouTube playlists, freeCodeCamp, Khan Academy, Coursera free tier, MIT OpenCourseWare, fast.ai, and other reputable free sources when relevant.
- Do not just list websites. Provide concrete search queries.

4. Build a realistic roadmap
- Create a week-by-week plan that matches the user's timeframe or an inferred reasonable timeframe.
- Make the sequence progressive and skill-building.
- Balance fundamentals, practice, and portfolio work.
- Keep projects small but meaningful.

5. Adapt for specialization
- If the user includes a secondary goal, specialization, or industry context, pivot the second half of the roadmap toward that use case.
- If the user includes a target job title, tailor tools, projects, and vocabulary to that role.

6. Output constraints
- Return only valid JSON.
- Do not include markdown.
- Do not include explanation outside the JSON.
- Keep industry_insight to exactly 2 sentences.
- Use the provided response schema exactly.
"""


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["industry_insight", "weekly_breakdown", "adjustment_log"],
    "properties": {
        "industry_insight": {
            "type": "string",
            "description": "Exactly two sentences explaining why the skill is trending and valuable.",
        },
        "weekly_breakdown": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "week",
                    "focus",
                    "why_this_week",
                    "priority_focus",
                    "time_budget",
                    "execution_plan",
                    "resources",
                    "hands_on_project",
                ],
                "properties": {
                    "week": {"type": "integer", "minimum": 1},
                    "focus": {"type": "string"},
                    "why_this_week": {"type": "string"},
                    "priority_focus": {"type": "string"},
                    "time_budget": {"type": "string"},
                    "execution_plan": {"type": "string"},
                    "resources": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "search_query",
                                "title",
                                "url",
                                "source_type",
                                "access_note",
                                "why_this_resource",
                                "contribution_to_path",
                                "primary_focus",
                                "time_fit",
                                "use_strategy",
                            ],
                            "properties": {
                                "search_query": {"type": "string"},
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "source_type": {"type": "string"},
                                "access_note": {"type": "string"},
                                "why_this_resource": {"type": "string"},
                                "contribution_to_path": {"type": "string"},
                                "primary_focus": {"type": "string"},
                                "time_fit": {"type": "string"},
                                "use_strategy": {"type": "string"},
                            },
                        },
                    },
                    "hands_on_project": {"type": "string"},
                },
            },
        },
        "adjustment_log": {
            "type": "string",
            "description": "Brief explanation of how the roadmap was adapted and what assumptions were inferred.",
        },
    },
}


FIELD_ALIASES = {
    "topic": "topic",
    "subject": "topic",
    "experience level": "experience_level",
    "experience_level": "experience_level",
    "level": "experience_level",
    "schedule length": "schedule_length",
    "schedule_length": "schedule_length",
    "timeframe": "schedule_length",
    "job focus": "job_industry_focus",
    "industry focus": "job_industry_focus",
    "job/industry focus": "job_industry_focus",
    "job_industry_focus": "job_industry_focus",
    "custom modifications": "custom_modifications",
    "custom_modifications": "custom_modifications",
    "secondary goal": "secondary_goal",
    "secondary_goal": "secondary_goal",
    "domain specialization": "domain_specialization",
    "domain_specialization": "domain_specialization",
    "specialization": "domain_specialization",
    "time available per week": "time_available_per_week",
    "time_available_per_week": "time_available_per_week",
    "hours per week": "time_available_per_week",
    "weekly time": "time_available_per_week",
    "target job title": "target_job_title",
    "target_job_title": "target_job_title",
}


DEFAULTS = {
    "experience_level": "beginner",
    "schedule_length": "8 weeks",
    "job_industry_focus": "general career relevance",
    "time_available_per_week": "6-8 hours",
}


DEFAULT_MODEL = "gpt-5.2"
DEFAULT_PROVIDER = "browse"
OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_COMPATIBLE_DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"

AI_PROVIDER_LABELS = {
    "ollama": "Local Ollama",
    "openai": "OpenAI",
    "openai_compatible": "OpenAI-Compatible",
    "gemini": "Google Gemini",
    "anthropic": "Anthropic Claude",
    "deepseek": "DeepSeek",
    "perplexity": "Perplexity",
}

AI_DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "openai": "gpt-5.2",
    "openai_compatible": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-3-5-sonnet-latest",
    "deepseek": "deepseek-chat",
    "perplexity": "sonar",
}


PROFILE_LIBRARY: List[Dict[str, Any]] = [
    {
        "id": "data_engineering",
        "keywords": [
            "data engineering",
            "data engineer",
            "analytics engineer",
            "etl",
            "elt",
            "airflow",
            "dbt",
            "warehouse",
            "spark",
            "snowflake",
            "bigquery",
        ],
        "skill_name": "data engineering",
        "trend_sentence": "This roadmap emphasizes modern data engineering skills because teams continue investing in reliable pipelines, governed analytics layers, and cloud-native data platforms.",
        "modules": [
            {
                "focus": "SQL fundamentals, relational modeling, and warehouse thinking",
                "resources": [
                    "PostgreSQL official documentation SELECT JOIN aggregate tutorial",
                    "freeCodeCamp SQL full course relational database design",
                    "GitHub data engineering SQL practice repository analytics engineering",
                ],
                "hands_on_project": "Model a small sales dataset in PostgreSQL and answer business questions with reusable SQL queries.",
            },
            {
                "focus": "Python for data processing, file formats, and testable transforms",
                "resources": [
                    "Python official documentation csv pathlib json tutorial",
                    "YouTube Python data engineering pandas ETL tutorial free",
                    "GitHub python ETL project unit tests repository",
                ],
                "hands_on_project": "Build a Python pipeline that ingests CSV files, validates records, and writes cleaned output tables.",
            },
            {
                "focus": "ELT workflows, dbt basics, and analytics engineering patterns",
                "resources": [
                    "dbt official docs getting started models tests sources",
                    "YouTube dbt fundamentals analytics engineering tutorial free",
                    "GitHub dbt starter project marts staging sources tests",
                ],
                "hands_on_project": "Create a staged-to-mart dbt project with source freshness checks and basic data tests.",
            },
            {
                "focus": "Orchestration with Airflow and production-minded scheduling",
                "resources": [
                    "Apache Airflow official documentation tutorial DAG task scheduling",
                    "YouTube Airflow beginner orchestration tutorial data engineering",
                    "GitHub airflow ETL example dag postgres repository",
                ],
                "hands_on_project": "Schedule your ingestion and dbt steps in a simple Airflow DAG with logging and retry behavior.",
            },
            {
                "focus": "Distributed data processing concepts with Spark",
                "resources": [
                    "Apache Spark official documentation SQL DataFrame getting started",
                    "freeCodeCamp Apache Spark data engineering full course",
                    "GitHub pyspark batch processing example parquet partitioning",
                ],
                "hands_on_project": "Process a larger synthetic event dataset with PySpark and compare results to a local pandas flow.",
            },
            {
                "focus": "Streaming, event pipelines, and data quality monitoring",
                "resources": [
                    "Kafka official documentation event streaming basics consumers producers",
                    "YouTube Kafka for data engineers tutorial free",
                    "GitHub data quality checks pipeline Great Expectations example repository",
                ],
                "hands_on_project": "Simulate a stream of app events and define quality checks for late, duplicated, and malformed records.",
            },
            {
                "focus": "Cloud warehouse architecture, cost awareness, and observability",
                "resources": [
                    "BigQuery documentation cost optimization partition clustering tutorial",
                    "Snowflake documentation warehouse sizing query performance guide",
                    "YouTube data observability monitoring lineage tutorial free",
                ],
                "hands_on_project": "Write an architecture note that compares one local pipeline to a cloud deployment and adds monitoring checkpoints.",
            },
            {
                "focus": "Portfolio capstone for a job-ready data platform story",
                "resources": [
                    "GitHub awesome data engineering portfolio project examples",
                    "YouTube data engineer portfolio project end to end tutorial",
                    "freeCodeCamp resume project storytelling data engineering portfolio",
                ],
                "hands_on_project": "Publish an end-to-end data platform project that shows ingestion, transformation, orchestration, documentation, and business-ready outputs.",
            },
        ],
    },
    {
        "id": "frontend",
        "keywords": [
            "frontend",
            "front end",
            "web development",
            "react",
            "next.js",
            "javascript",
            "typescript",
            "ui engineer",
        ],
        "skill_name": "frontend development",
        "trend_sentence": "This roadmap emphasizes frontend engineering because companies still need fast, accessible interfaces that tie product quality directly to revenue and retention.",
        "modules": [
            {
                "focus": "Semantic HTML, modern CSS, and responsive layout fundamentals",
                "resources": [
                    "MDN HTML CSS responsive design semantic elements guide",
                    "freeCodeCamp responsive web design full course latest",
                    "YouTube modern CSS layout flexbox grid tutorial free",
                ],
                "hands_on_project": "Build a polished landing page that works cleanly on mobile, tablet, and desktop screens.",
            },
            {
                "focus": "JavaScript and TypeScript foundations for interactive UIs",
                "resources": [
                    "TypeScript official documentation handbook everyday types",
                    "freeCodeCamp JavaScript algorithms DOM course latest",
                    "GitHub JavaScript frontend exercises TypeScript repository",
                ],
                "hands_on_project": "Create a typed interactive component set with form validation, filtering, and async state handling.",
            },
            {
                "focus": "React component architecture and state-driven thinking",
                "resources": [
                    "React official documentation learn components state effects",
                    "YouTube React project based course 2026 free",
                    "GitHub React dashboard starter TypeScript repository",
                ],
                "hands_on_project": "Build a small dashboard with reusable cards, tables, filters, and loading states.",
            },
            {
                "focus": "APIs, forms, routing, and authentication-aware flows",
                "resources": [
                    "Next.js official documentation routing data fetching forms",
                    "MDN fetch API forms authentication tutorial",
                    "YouTube React forms auth protected routes tutorial free",
                ],
                "hands_on_project": "Wire a multi-page app to a public API with search, save, and sign-in style navigation flows.",
            },
            {
                "focus": "Accessibility, design systems, and maintainable styling",
                "resources": [
                    "WAI WCAG accessibility tutorial forms navigation contrast",
                    "YouTube design system React accessibility tutorial free",
                    "GitHub accessible component library React TypeScript example",
                ],
                "hands_on_project": "Refactor your app into a small design system with accessible buttons, forms, and keyboard-friendly navigation.",
            },
            {
                "focus": "Testing, performance, and production debugging habits",
                "resources": [
                    "Playwright official documentation end to end testing getting started",
                    "web.dev performance optimization core web vitals guide",
                    "YouTube React testing Playwright Lighthouse tutorial free",
                ],
                "hands_on_project": "Add end-to-end tests, optimize bundle-heavy pages, and document measurable performance improvements.",
            },
            {
                "focus": "Framework deployment and role-specific specialization",
                "resources": [
                    "Vercel Next.js deployment documentation environment variables",
                    "GitHub frontend portfolio projects ecommerce dashboard marketing site",
                    "YouTube Next.js production app tutorial free",
                ],
                "hands_on_project": "Turn your strongest project into a production-style app tailored to your target product domain.",
            },
            {
                "focus": "Portfolio capstone with product thinking and polish",
                "resources": [
                    "GitHub awesome frontend portfolio examples React TypeScript",
                    "freeCodeCamp frontend developer portfolio project guide",
                    "YouTube frontend interview project walkthrough free",
                ],
                "hands_on_project": "Ship a portfolio-ready product that demonstrates architecture, accessibility, testing, and deployment in one story.",
            },
        ],
    },
    {
        "id": "backend",
        "keywords": [
            "backend",
            "back end",
            "api",
            "server",
            "fastapi",
            "node",
            "express",
            "spring",
            "golang",
            "microservices",
        ],
        "skill_name": "backend development",
        "trend_sentence": "This roadmap emphasizes backend engineering because organizations keep prioritizing dependable APIs, data integrity, and service reliability across their product stack.",
        "modules": [
            {
                "focus": "HTTP, REST design, and backend system fundamentals",
                "resources": [
                    "MDN HTTP overview methods status codes REST guide",
                    "YouTube backend development fundamentals APIs databases tutorial free",
                    "GitHub backend system design learning repository REST API examples",
                ],
                "hands_on_project": "Design an API contract for a small product service with clear endpoints, errors, and resource relationships.",
            },
            {
                "focus": "Service implementation, routing, validation, and request lifecycle",
                "resources": [
                    "FastAPI official documentation tutorial path operation body validation",
                    "Node.js Express REST API tutorial validation free",
                    "GitHub backend CRUD API starter tests repository",
                ],
                "hands_on_project": "Implement a CRUD API with validation, pagination, and structured error responses.",
            },
            {
                "focus": "Databases, schema design, and persistence patterns",
                "resources": [
                    "PostgreSQL official documentation data types indexes constraints",
                    "YouTube backend database schema design postgres tutorial free",
                    "GitHub backend postgres migration ORM example repository",
                ],
                "hands_on_project": "Model a relational schema for your API and add migrations plus seed data for local development.",
            },
            {
                "focus": "Authentication, authorization, and secure defaults",
                "resources": [
                    "OWASP authentication cheat sheet session management",
                    "FastAPI security OAuth2 JWT tutorial official docs",
                    "YouTube backend auth RBAC JWT tutorial free",
                ],
                "hands_on_project": "Add sign-in, role-based access control, and protected routes to your service.",
            },
            {
                "focus": "Testing, observability, and debugging in real services",
                "resources": [
                    "pytest official documentation fixtures API testing",
                    "OpenTelemetry documentation tracing logging metrics getting started",
                    "YouTube backend testing observability tutorial free",
                ],
                "hands_on_project": "Write integration tests and add logs plus traces for your highest-value endpoints.",
            },
            {
                "focus": "Caching, queues, and asynchronous work",
                "resources": [
                    "Redis official documentation data structures caching tutorial",
                    "RabbitMQ tutorials official work queues routing",
                    "YouTube background jobs queues backend architecture tutorial free",
                ],
                "hands_on_project": "Move one expensive workflow into an async job and cache one read-heavy endpoint.",
            },
            {
                "focus": "Deployment, containers, and runtime reliability",
                "resources": [
                    "Docker official documentation getting started containerize application",
                    "GitHub dockerized backend API production example repository",
                    "YouTube deploy backend API Docker cloud tutorial free",
                ],
                "hands_on_project": "Containerize the service, document runtime configuration, and publish a deployment checklist.",
            },
            {
                "focus": "Capstone service with job-ready engineering habits",
                "resources": [
                    "GitHub backend portfolio projects microservice API examples",
                    "YouTube backend project architecture code review tutorial free",
                    "freeCodeCamp backend developer portfolio guide API projects",
                ],
                "hands_on_project": "Ship a backend capstone that includes API design, security, tests, persistence, and deployment notes.",
            },
        ],
    },
    {
        "id": "machine_learning",
        "keywords": [
            "machine learning",
            "ml",
            "ml engineer",
            "artificial intelligence",
            "deep learning",
            "pytorch",
            "tensorflow",
            "mlops",
        ],
        "skill_name": "machine learning",
        "trend_sentence": "This roadmap emphasizes machine learning because companies want people who can move from experimentation to measurable product or automation outcomes.",
        "modules": [
            {
                "focus": "Python, NumPy, pandas, and data preparation foundations",
                "resources": [
                    "pandas official documentation getting started tutorial",
                    "freeCodeCamp machine learning with Python full course",
                    "Kaggle Python pandas exercises beginner dataset",
                ],
                "hands_on_project": "Clean and profile a messy dataset, then publish a short notebook explaining your preprocessing choices.",
            },
            {
                "focus": "Statistics, evaluation, and model selection habits",
                "resources": [
                    "Khan Academy statistics and probability course free",
                    "scikit-learn official documentation model evaluation guide",
                    "YouTube machine learning evaluation precision recall cross validation tutorial free",
                ],
                "hands_on_project": "Compare baseline models on a simple prediction task and justify your metric choice.",
            },
            {
                "focus": "Supervised learning with scikit-learn pipelines",
                "resources": [
                    "scikit-learn official documentation pipeline preprocessing tutorial",
                    "YouTube scikit-learn end to end machine learning project free",
                    "GitHub sklearn machine learning pipeline example repository",
                ],
                "hands_on_project": "Train and package a reproducible scikit-learn pipeline with feature preprocessing and error analysis.",
            },
            {
                "focus": "Feature engineering, experimentation, and iteration loops",
                "resources": [
                    "Google machine learning crash course feature engineering",
                    "fast.ai practical machine learning course lesson tabular",
                    "GitHub feature engineering examples machine learning repository",
                ],
                "hands_on_project": "Run a controlled feature experiment and document which changes actually improved results.",
            },
            {
                "focus": "Neural network fundamentals and modern deep learning tooling",
                "resources": [
                    "PyTorch official tutorials beginner deep learning",
                    "fast.ai deep learning for coders course free",
                    "YouTube PyTorch beginner tutorial project based free",
                ],
                "hands_on_project": "Train a small neural model on a focused dataset and compare it against a classical baseline.",
            },
            {
                "focus": "Model serving, reproducibility, and MLOps basics",
                "resources": [
                    "MLflow documentation tracking models tutorial",
                    "FastAPI model serving tutorial machine learning API free",
                    "GitHub MLOps starter project Docker MLflow FastAPI repository",
                ],
                "hands_on_project": "Serve a trained model behind an API and track experiments plus versioned artifacts.",
            },
            {
                "focus": "Monitoring, drift thinking, and domain specialization",
                "resources": [
                    "Evidently AI documentation data drift monitoring tutorial",
                    "YouTube machine learning monitoring drift tutorial free",
                    "GitHub production ML monitoring example repository",
                ],
                "hands_on_project": "Define a monitoring plan for one production-style model with drift and quality alerts.",
            },
            {
                "focus": "Capstone from notebook to deployable ML story",
                "resources": [
                    "GitHub machine learning portfolio projects end to end repository",
                    "YouTube ML engineer portfolio project tutorial free",
                    "freeCodeCamp machine learning portfolio advice projects",
                ],
                "hands_on_project": "Publish an end-to-end ML capstone with experimentation, deployment, and business-facing documentation.",
            },
        ],
    },
    {
        "id": "cybersecurity",
        "keywords": [
            "cybersecurity",
            "security",
            "soc analyst",
            "penetration testing",
            "blue team",
            "red team",
            "siem",
            "incident response",
        ],
        "skill_name": "cybersecurity",
        "trend_sentence": "This roadmap emphasizes cybersecurity because attack surfaces keep expanding across cloud systems, software supply chains, and identity workflows.",
        "modules": [
            {
                "focus": "Networking, Linux, and core security fundamentals",
                "resources": [
                    "Cisco networking basics free course TCP IP security fundamentals",
                    "Linux Journey command line permissions processes free",
                    "YouTube cybersecurity fundamentals networking Linux tutorial free",
                ],
                "hands_on_project": "Document a small home-lab network and explain basic ports, protocols, and hardening opportunities.",
            },
            {
                "focus": "Threat models, common vulnerabilities, and OWASP thinking",
                "resources": [
                    "OWASP Top 10 official project latest",
                    "PortSwigger Web Security Academy free labs beginner",
                    "YouTube threat modeling application security tutorial free",
                ],
                "hands_on_project": "Threat-model a small web application and list realistic abuse cases plus mitigations.",
            },
            {
                "focus": "Logs, detection ideas, and blue-team analysis habits",
                "resources": [
                    "Splunk fundamentals free training detection basics",
                    "Elastic security getting started SIEM tutorial",
                    "YouTube SOC analyst log analysis tutorial free",
                ],
                "hands_on_project": "Analyze sample logs and write simple detection logic for suspicious authentication behavior.",
            },
            {
                "focus": "Scripting and automation for repeatable security work",
                "resources": [
                    "Python official documentation subprocess pathlib csv tutorial",
                    "freeCodeCamp Python automation security scripting course",
                    "GitHub cybersecurity Python automation scripts repository",
                ],
                "hands_on_project": "Write a small script that parses system or web logs and summarizes suspicious patterns.",
            },
            {
                "focus": "Identity, access control, and cloud security basics",
                "resources": [
                    "AWS IAM documentation getting started permissions policies",
                    "Microsoft Learn identity access security fundamentals free",
                    "YouTube cloud security IAM least privilege tutorial free",
                ],
                "hands_on_project": "Design a least-privilege access plan for a small cloud-hosted application.",
            },
            {
                "focus": "Vulnerability management and incident response workflow",
                "resources": [
                    "NIST incident response guide free pdf",
                    "OpenVAS vulnerability scanning tutorial free",
                    "YouTube incident response tabletop exercise tutorial free",
                ],
                "hands_on_project": "Run a mini tabletop exercise and write an incident timeline plus remediation checklist.",
            },
            {
                "focus": "Specialization labs for blue team, red team, or appsec",
                "resources": [
                    "TryHackMe free rooms cybersecurity beginner intermediate",
                    "Hack The Box free labs cybersecurity practice",
                    "GitHub detection engineering appsec security lab repositories",
                ],
                "hands_on_project": "Complete and document one focused lab sequence aligned to your intended security path.",
            },
            {
                "focus": "Portfolio capstone with reports and defensive judgment",
                "resources": [
                    "GitHub cybersecurity portfolio report examples",
                    "YouTube SOC analyst portfolio project tutorial free",
                    "freeCodeCamp cybersecurity career project advice",
                ],
                "hands_on_project": "Publish a capstone report that shows lab evidence, findings, severity reasoning, and remediation guidance.",
            },
        ],
    },
    {
        "id": "data_analysis",
        "keywords": [
            "data analysis",
            "data analyst",
            "business intelligence",
            "analytics",
            "tableau",
            "power bi",
            "sql analyst",
        ],
        "skill_name": "data analytics",
        "trend_sentence": "This roadmap emphasizes data analytics because teams still rely on people who can turn raw business data into decisions, dashboards, and measurable experiments.",
        "modules": [
            {
                "focus": "SQL querying, spreadsheet fluency, and business metrics basics",
                "resources": [
                    "freeCodeCamp SQL full course data analyst",
                    "Google Sheets formulas pivot tables tutorial free",
                    "GitHub SQL analytics case study practice repository",
                ],
                "hands_on_project": "Analyze a sales or product dataset with spreadsheet pivots and SQL summaries for a short stakeholder memo.",
            },
            {
                "focus": "Data cleaning and analysis with pandas",
                "resources": [
                    "pandas official documentation user guide getting started",
                    "YouTube pandas data analysis project tutorial free",
                    "Kaggle data cleaning exercises beginner free",
                ],
                "hands_on_project": "Clean a messy dataset and produce a reproducible notebook with key findings and assumptions.",
            },
            {
                "focus": "Statistics for analysts and practical experiment thinking",
                "resources": [
                    "Khan Academy statistics and probability hypothesis testing",
                    "YouTube A/B testing statistics for analysts tutorial free",
                    "Coursera statistics for data analysis audit free search",
                ],
                "hands_on_project": "Evaluate a simple experiment scenario and explain metrics, bias risks, and decision criteria.",
            },
            {
                "focus": "Dashboard design and storytelling with BI tools",
                "resources": [
                    "Microsoft Learn Power BI dashboard tutorial free",
                    "Tableau public getting started dashboard tutorial free",
                    "YouTube dashboard design data storytelling tutorial free",
                ],
                "hands_on_project": "Build a dashboard that highlights trends, segments, and a clear recommended action.",
            },
            {
                "focus": "Business problem framing and stakeholder-ready analysis",
                "resources": [
                    "GitHub data analyst case study portfolio examples",
                    "YouTube data analyst business case study tutorial free",
                    "freeCodeCamp data analyst portfolio project ideas",
                ],
                "hands_on_project": "Translate one vague business question into metrics, a dataset plan, and a recommendation.",
            },
            {
                "focus": "Automation, reporting workflows, and reproducibility",
                "resources": [
                    "Python automation reporting pandas matplotlib tutorial",
                    "GitHub analytics reporting automation project repository",
                    "YouTube automate data analyst reports Python free",
                ],
                "hands_on_project": "Automate a recurring weekly report and save outputs in a consistent folder structure.",
            },
            {
                "focus": "Domain specialization for product, marketing, finance, or operations",
                "resources": [
                    "product analytics metrics north star retention tutorial free",
                    "marketing analytics attribution dashboard tutorial free",
                    "finance operations analytics case study tutorial free",
                ],
                "hands_on_project": "Tailor one dashboard and one written analysis to the domain you want to work in.",
            },
            {
                "focus": "Portfolio capstone with analysis, dashboard, and narrative",
                "resources": [
                    "GitHub data analyst portfolio examples dashboards SQL Python",
                    "YouTube data analyst portfolio review free",
                    "freeCodeCamp data analysis projects portfolio guide",
                ],
                "hands_on_project": "Publish a complete analytics case study with cleaned data, SQL work, dashboard screenshots, and a short business brief.",
            },
        ],
    },
    {
        "id": "devops_cloud",
        "keywords": [
            "devops",
            "cloud",
            "platform engineer",
            "site reliability",
            "sre",
            "kubernetes",
            "terraform",
            "aws",
            "azure",
            "gcp",
        ],
        "skill_name": "cloud and DevOps",
        "trend_sentence": "This roadmap emphasizes cloud and DevOps skills because teams keep standardizing delivery, infrastructure automation, and reliability practices across products.",
        "modules": [
            {
                "focus": "Linux, networking, and cloud delivery fundamentals",
                "resources": [
                    "Linux Journey networking processes permissions free",
                    "AWS cloud practitioner free training fundamentals",
                    "YouTube DevOps fundamentals Linux networking tutorial free",
                ],
                "hands_on_project": "Document the path from local code to a deployed service, including runtime, network, and environment assumptions.",
            },
            {
                "focus": "Git workflows, CI thinking, and pipeline basics",
                "resources": [
                    "GitHub Actions official documentation quickstart tutorial",
                    "YouTube CI CD pipeline beginner tutorial GitHub Actions free",
                    "GitHub sample CI pipeline test lint build repository",
                ],
                "hands_on_project": "Set up a CI pipeline that lints, tests, and packages a small sample app.",
            },
            {
                "focus": "Containers and immutable runtime habits",
                "resources": [
                    "Docker official documentation getting started containerize application",
                    "YouTube Docker full course DevOps beginners free",
                    "GitHub dockerized web app production example repository",
                ],
                "hands_on_project": "Containerize an app and create a repeatable local development plus runtime workflow.",
            },
            {
                "focus": "Infrastructure as code with Terraform",
                "resources": [
                    "Terraform official tutorials getting started providers modules",
                    "YouTube Terraform beginner infrastructure as code tutorial free",
                    "GitHub terraform starter aws azure gcp repository",
                ],
                "hands_on_project": "Write Terraform for a small cloud environment and organize it with variables and modules.",
            },
            {
                "focus": "Kubernetes concepts and service deployment",
                "resources": [
                    "Kubernetes official documentation concepts pods deployments services",
                    "YouTube Kubernetes beginner hands on tutorial free",
                    "GitHub kubernetes demo app manifests repository",
                ],
                "hands_on_project": "Deploy a simple containerized app with health checks, config, and scaling notes.",
            },
            {
                "focus": "Observability, alerting, and incident-friendly operations",
                "resources": [
                    "Prometheus documentation getting started metrics alerts",
                    "Grafana documentation dashboards alerting tutorial",
                    "YouTube observability DevOps Prometheus Grafana tutorial free",
                ],
                "hands_on_project": "Add metrics, dashboards, and a small alert plan for your deployed service.",
            },
            {
                "focus": "Security, secrets, and reliability engineering habits",
                "resources": [
                    "OWASP DevSecOps cheat sheet secrets management",
                    "Google SRE workbook free reliability chapters",
                    "YouTube platform engineering reliability tutorial free",
                ],
                "hands_on_project": "Create a runbook that covers secrets handling, rollback, and a simple incident response path.",
            },
            {
                "focus": "Capstone platform story for an employable DevOps portfolio",
                "resources": [
                    "GitHub DevOps portfolio projects terraform kubernetes cicd",
                    "YouTube DevOps portfolio project end to end free",
                    "freeCodeCamp DevOps roadmap project guide",
                ],
                "hands_on_project": "Publish a capstone that shows CI, containers, infrastructure as code, Kubernetes, and observability in one workflow.",
            },
        ],
    },
]


GENERIC_PROFILE = {
    "id": "generic",
    "keywords": [],
    "skill_name": "applied professional skill-building",
    "trend_sentence": "This roadmap emphasizes applied professional skill-building because employers consistently reward people who can turn fundamentals into visible projects and work samples.",
    "modules": [
        {
            "focus": "Foundations, vocabulary, and core concepts",
            "resources": [
                "{topic} official documentation getting started guide",
                "YouTube {topic} beginner playlist free",
                "GitHub awesome {topic} curated resources",
            ],
            "hands_on_project": "Create a compact notes document that defines the core terms and shows one small working example in {topic}.",
        },
        {
            "focus": "Essential tools and hands-on practice",
            "resources": [
                "{topic} official tutorial hands on",
                "free course {topic} practical tutorial",
                "GitHub {topic} beginner exercises repository",
            ],
            "hands_on_project": "Recreate one canonical beginner exercise in a way you can explain from memory.",
        },
        {
            "focus": "Structured practice and feedback loops",
            "resources": [
                "{topic} project based learning tutorial free",
                "YouTube {topic} intermediate practice tutorial",
                "GitHub {topic} sample projects repository",
            ],
            "hands_on_project": "Build a small practice project and keep a short change log of what improved between versions.",
        },
        {
            "focus": "Workflow, quality, and repeatability",
            "resources": [
                "{topic} best practices guide official",
                "GitHub {topic} style guide examples",
                "YouTube {topic} workflow tutorial free",
            ],
            "hands_on_project": "Turn your practice work into a repeatable workflow with clear steps and quality checks.",
        },
        {
            "focus": "Applied problem-solving in a realistic scenario",
            "resources": [
                "{topic} case study tutorial free",
                "GitHub {topic} case study examples",
                "YouTube {topic} real world project tutorial",
            ],
            "hands_on_project": "Solve one realistic problem in a domain you care about and explain the tradeoffs you made.",
        },
        {
            "focus": "Specialization and role alignment",
            "resources": [
                "{topic} {specialization} tutorial free",
                "GitHub {topic} {specialization} examples",
                "YouTube {topic} career project tutorial",
            ],
            "hands_on_project": "Adapt your work to match the specialization or job context you want most.",
        },
        {
            "focus": "Portfolio framing and evidence of skill",
            "resources": [
                "{topic} portfolio project examples",
                "YouTube {topic} portfolio review tutorial free",
                "GitHub {topic} portfolio repository examples",
            ],
            "hands_on_project": "Package your strongest work with screenshots, notes, and a plain-language explanation of value.",
        },
        {
            "focus": "Capstone and next-step planning",
            "resources": [
                "{topic} capstone project ideas free",
                "GitHub {topic} advanced project repository",
                "YouTube {topic} roadmap next steps tutorial",
            ],
            "hands_on_project": "Ship a capstone that proves competence and list the next two skills that would make you more employable.",
        },
    ],
}


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", " ").replace("_", " ")


def normalize_request(user_input: Any) -> Tuple[Dict[str, Any], List[str]]:
    if isinstance(user_input, str):
        normalized = {"topic": user_input.strip()}
    elif isinstance(user_input, dict):
        normalized = {}
        for raw_key, value in user_input.items():
            if value in (None, ""):
                continue
            key = FIELD_ALIASES.get(
                _normalize_key(str(raw_key)),
                _normalize_key(str(raw_key)).replace(" ", "_"),
            )
            normalized[key] = value
    else:
        raise TypeError("user_input must be a string or dictionary")

    assumptions: List[str] = []
    for field, default_value in DEFAULTS.items():
        if field not in normalized:
            normalized[field] = default_value
            assumptions.append(f"Inferred {field} as {default_value}.")

    if "topic" not in normalized:
        if "target_job_title" in normalized:
            normalized["topic"] = normalized["target_job_title"]
            assumptions.append("Inferred topic from target_job_title.")
        elif "domain_specialization" in normalized:
            normalized["topic"] = normalized["domain_specialization"]
            assumptions.append("Inferred topic from domain_specialization.")
        else:
            raise ValueError("A topic, target_job_title, or domain_specialization is required.")

    return normalized, assumptions


def build_user_brief(normalized_input: Dict[str, Any], assumptions: Iterable[str]) -> str:
    lines = ["Create a learning roadmap for this user request:"]
    ordered_fields = [
        "topic",
        "experience_level",
        "schedule_length",
        "time_available_per_week",
        "job_industry_focus",
        "target_job_title",
        "domain_specialization",
        "secondary_goal",
        "custom_modifications",
    ]

    for field in ordered_fields:
        if field in normalized_input:
            pretty = field.replace("_", " ").title()
            lines.append(f"- {pretty}: {normalized_input[field]}")

    assumption_list = list(assumptions)
    if assumption_list:
        lines.append("- Inferred defaults to note in adjustment_log:")
        for item in assumption_list:
            lines.append(f"  - {item}")

    return "\n".join(lines)


def build_request_payload(user_input: Any) -> Dict[str, Any]:
    normalized_input, assumptions = normalize_request(user_input)
    user_brief = build_user_brief(normalized_input, assumptions)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_brief},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "learning_roadmap",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
        "normalized_input": normalized_input,
        "inferred_defaults": assumptions,
    }


def _build_json_only_instruction() -> str:
    return (
        "Return only valid JSON that matches this schema exactly. "
        f"Schema: {json.dumps(RESPONSE_SCHEMA, separators=(',', ':'))}"
    )


def build_ai_messages(user_input: Any) -> Dict[str, Any]:
    normalized_input, assumptions = normalize_request(user_input)
    user_brief = build_user_brief(normalized_input, assumptions)
    return {
        "system": SYSTEM_PROMPT + "\n\n" + _build_json_only_instruction(),
        "user": user_brief,
        "normalized_input": normalized_input,
        "inferred_defaults": assumptions,
    }


def _http_json_request(
    url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int = 120,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI provider request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"AI provider request could not reach the network: {exc.reason}") from exc


def _extract_json_string(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise ValueError(f"Model returned non-JSON output: {text}")


def _extract_openai_compatible_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("AI provider response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts).strip()
    raise ValueError("AI provider response did not include text content.")


def _extract_anthropic_text(response_json: Dict[str, Any]) -> str:
    parts: List[str] = []
    for block in response_json.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    text = "".join(parts).strip()
    if not text:
        raise ValueError("Anthropic response did not include text content.")
    return text


def _extract_gemini_text(response_json: Dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response did not include candidates.")
    parts = candidates[0].get("content", {}).get("parts", []) or []
    texts = [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    text = "".join(texts).strip()
    if not text:
        raise ValueError("Gemini response did not include text content.")
    return text


def _default_model_for_provider(provider: str, requested_model: Optional[str]) -> str:
    if requested_model:
        return requested_model
    return AI_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)


def build_responses_api_request(
    user_input: Any,
    model: str = DEFAULT_MODEL,
    enable_web_search: bool = True,
) -> Dict[str, Any]:
    prompt_payload = build_request_payload(user_input)
    api_request: Dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": prompt_payload["messages"][1]["content"],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "learning_roadmap",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    }

    if enable_web_search:
        api_request["tools"] = [
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "America/New_York",
                },
            }
        ]
        api_request["tool_choice"] = "auto"
        api_request["include"] = ["web_search_call.action.sources"]

    return api_request


def _extract_output_text(response_json: Dict[str, Any]) -> str:
    output = response_json.get("output", [])
    parts: List[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    text = "".join(parts).strip()
    if not text:
        raise ValueError("Model response did not include output text.")
    return text


def extract_sources(response_json: Dict[str, Any]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for item in response_json.get("output", []):
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action", {})
        for source in action.get("sources", []) or []:
            if isinstance(source, dict) and source.get("url"):
                sources.append(
                    {
                        "title": str(source.get("title") or source["url"]),
                        "url": str(source["url"]),
                    }
                )
    return sources


def _extract_week_count(schedule_length: str) -> int:
    match = re.search(r"(\d+)", str(schedule_length))
    if not match:
        return 8
    value = max(1, int(match.group(1)))
    if "month" in str(schedule_length).lower():
        value *= 4
    return min(max(value, 1), 24)


def _average_hours_per_week(time_available_per_week: str) -> int:
    values = [int(item) for item in re.findall(r"\d+", str(time_available_per_week))]
    if not values:
        return 7
    return round(sum(values) / len(values))


def _combined_text(normalized_input: Dict[str, Any]) -> str:
    parts = [
        normalized_input.get("topic", ""),
        normalized_input.get("target_job_title", ""),
        normalized_input.get("domain_specialization", ""),
        normalized_input.get("secondary_goal", ""),
        normalized_input.get("job_industry_focus", ""),
        normalized_input.get("custom_modifications", ""),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def _compact_query_phrase(value: str, max_terms: int = 8) -> str:
    tokens = re.findall(r"[a-zA-Z0-9\+\#\.-]+", str(value or "").lower())
    stop = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "for",
        "with",
        "toward",
        "into",
        "from",
        "that",
        "this",
        "week",
        "project",
        "practical",
        "focused",
        "module",
    }
    filtered = [token for token in tokens if token not in stop and len(token) > 1]
    compact = filtered[:max_terms] if filtered else tokens[:max_terms]
    return " ".join(compact).strip()


def _choose_profile(normalized_input: Dict[str, Any]) -> Dict[str, Any]:
    haystack = _combined_text(normalized_input)
    best_profile = GENERIC_PROFILE
    best_score = 0
    for profile in PROFILE_LIBRARY:
        score = sum(1 for keyword in profile["keywords"] if keyword in haystack)
        if score > best_score:
            best_profile = profile
            best_score = score
    return copy.deepcopy(best_profile)


def _format_string_template(value: str, normalized_input: Dict[str, Any]) -> str:
    topic = normalized_input.get("topic", "the topic")
    specialization = (
        normalized_input.get("domain_specialization")
        or normalized_input.get("secondary_goal")
        or normalized_input.get("target_job_title")
        or "specialization"
    )
    return value.format(topic=topic, specialization=specialization)


def _resource_source_type(query: str) -> str:
    lowered = query.lower()
    if "official" in lowered or "documentation" in lowered or " docs" in lowered:
        return "Official documentation"
    if "github" in lowered:
        return "GitHub repository"
    if "youtube" in lowered:
        return "YouTube tutorial"
    if "freecodecamp" in lowered:
        return "freeCodeCamp"
    if "khan academy" in lowered:
        return "Khan Academy"
    if "coursera" in lowered:
        return "Coursera free tier"
    if "mit opencourseware" in lowered or "mit ocw" in lowered:
        return "MIT OpenCourseWare"
    if "fast.ai" in lowered:
        return "fast.ai"
    return "Free web resource"


def _source_default_domain(source_type: str, query: str) -> str:
    by_source = {
        "Official documentation": "docs.python.org",
        "GitHub repository": "github.com",
        "YouTube tutorial": "youtube.com",
        "freeCodeCamp": "freecodecamp.org",
        "Khan Academy": "khanacademy.org",
        "Coursera free tier": "coursera.org",
        "MIT OpenCourseWare": "ocw.mit.edu",
        "fast.ai": "fast.ai",
        "Free web resource": "github.com",
    }
    # Preserve explicit source intent (GitHub/YouTube/etc.) so links stay aligned
    # with the recommended resource type.
    if source_type and source_type != "Official documentation":
        return by_source.get(source_type, "github.com")

    lowered = query.lower()
    keyword_domains = {
        "postgresql": "postgresql.org",
        "python": "docs.python.org",
        "dbt": "docs.getdbt.com",
        "airflow": "airflow.apache.org",
        "spark": "spark.apache.org",
        "kafka": "kafka.apache.org",
        "react": "react.dev",
        "next.js": "nextjs.org",
        "typescript": "typescriptlang.org",
        "node": "nodejs.org",
        "docker": "docs.docker.com",
        "kubernetes": "kubernetes.io",
        "terraform": "developer.hashicorp.com",
        "prometheus": "prometheus.io",
        "grafana": "grafana.com",
        "pytorch": "pytorch.org",
        "scikit-learn": "scikit-learn.org",
        "fastapi": "fastapi.tiangolo.com",
    }
    for keyword, domain in keyword_domains.items():
        if keyword in lowered:
            return domain

    return by_source.get(source_type, "github.com")


def _ensure_direct_resource_url(resource: Dict[str, Any]) -> Dict[str, Any]:
    current_url = str(resource.get("url") or "").strip()
    if current_url.startswith("http://") or current_url.startswith("https://"):
        return resource

    source_type = str(resource.get("source_type") or "Free web resource")
    query = str(resource.get("search_query") or resource.get("title") or "")
    curated = find_curated_exact_resource(query, source_type)
    if curated:
        resource["url"] = str(curated.get("url") or "")
        resource["title"] = str(resource.get("title") or curated.get("title") or query)
        resource["source_type"] = str(curated.get("source_label") or source_type)
        return resource
    domain = _source_default_domain(source_type, query)
    resource["url"] = build_direct_topic_url(domain, query)
    return resource


PAID_RESOURCE_TOKENS = {
    "pricing",
    "plans",
    "subscription",
    "subscribe",
    "trial",
    "checkout",
    "buy",
    "purchase",
    "udemy",
    "skillshare",
    "pluralsight",
}


def _has_paid_signal(*values: str) -> bool:
    combined = " ".join(values).lower()
    return any(token in combined for token in PAID_RESOURCE_TOKENS)


def _trusted_free_resource_template(week_focus: str, resource_index: int) -> Dict[str, str]:
    focus_phrase = _compact_query_phrase(week_focus, max_terms=8) or "core concepts"
    role = _resource_role(resource_index)
    if role == "foundation":
        query = f"Official documentation {focus_phrase} getting started quickstart"
        source_type = "Official documentation"
    elif role == "guided practice":
        query = f"freeCodeCamp {focus_phrase} project based tutorial free"
        source_type = "freeCodeCamp"
    else:
        query = f"GitHub {focus_phrase} practical project example"
        source_type = "GitHub repository"
    return {
        "search_query": query,
        "title": query,
        "source_type": source_type,
        "access_note": "Upgraded to a higher-confidence free source to keep this path quality-first and budget-safe.",
    }


def _resource_source_rank(source_type: str) -> int:
    order = {
        "Official documentation": 0,
        "freeCodeCamp": 1,
        "Khan Academy": 1,
        "MIT OpenCourseWare": 1,
        "fast.ai": 1,
        "GitHub repository": 2,
        "YouTube tutorial": 3,
        "Coursera free tier": 4,
        "Free web resource": 5,
    }
    return order.get(source_type, 5)


def _resource_cap_minutes(hours_per_week: int, resource_index: int) -> int:
    if hours_per_week <= 4:
        caps = [60, 50, 45]
    elif hours_per_week <= 7:
        caps = [95, 80, 70]
    elif hours_per_week <= 10:
        caps = [130, 110, 95]
    else:
        caps = [170, 145, 130]
    return caps[min(resource_index, len(caps) - 1)]


def _estimate_resource_minutes(resource: Dict[str, Any]) -> int:
    source_type = str(resource.get("source_type") or "Free web resource")
    query_text = f"{resource.get('search_query', '')} {resource.get('title', '')}".lower()
    base_minutes = {
        "Official documentation": 95,
        "freeCodeCamp": 145,
        "Khan Academy": 120,
        "MIT OpenCourseWare": 180,
        "Coursera free tier": 170,
        "fast.ai": 165,
        "YouTube tutorial": 115,
        "GitHub repository": 80,
        "Free web resource": 95,
    }.get(source_type, 95)

    if any(token in query_text for token in ["full course", "complete", "bootcamp", "specialization", "semester"]):
        base_minutes += 75
    if "playlist" in query_text:
        base_minutes += 45
    if any(token in query_text for token in ["getting started", "quickstart", "fundamentals", "intro", "basics"]):
        base_minutes -= 25
    return max(35, min(base_minutes, 300))


def _resource_quality_score(resource: Dict[str, Any]) -> int:
    source_type = str(resource.get("source_type") or "Free web resource")
    query_text = f"{resource.get('search_query', '')} {resource.get('title', '')}".lower()
    url = str(resource.get("url") or "").lower()

    score = {
        "Official documentation": 4,
        "freeCodeCamp": 4,
        "Khan Academy": 4,
        "MIT OpenCourseWare": 4,
        "Coursera free tier": 2,
        "fast.ai": 4,
        "YouTube tutorial": 2,
        "GitHub repository": 3,
        "Free web resource": 1,
    }.get(source_type, 1)

    if any(token in query_text for token in ["official", "documentation", "docs"]):
        score += 1
    if any(token in query_text for token in ["free", "audit", "open course", "open source", "quickstart", "getting started"]):
        score += 1
    if _has_paid_signal(query_text, url):
        score -= 2
    if any(token in url for token in [".edu", "docs.", "developer.", "github.com", "freecodecamp.org", "ocw.mit.edu"]):
        score += 1
    if source_type == "Free web resource" and not url:
        score -= 1
    return max(1, min(score, 5))


def _quality_label(score: int) -> str:
    if score >= 4:
        return "high"
    if score >= 2:
        return "solid"
    return "exploratory"


def _reviewed_time_fit(
    estimated_minutes: int,
    cap_minutes: int,
    source_type: str,
) -> str:
    if estimated_minutes <= cap_minutes:
        return (
            f"Estimated effort is about {estimated_minutes} minutes, which fits within the weekly cap of about {cap_minutes} minutes. "
            f"Stay on the sections tied to this week's deliverable."
        )
    if source_type == "GitHub repository":
        return (
            f"Estimated effort is about {estimated_minutes} minutes, which is likely too long for one week. "
            f"Cap this at about {cap_minutes} minutes by reading the README, setup, and one relevant implementation slice."
        )
    return (
        f"Estimated effort is about {estimated_minutes} minutes, which is likely longer than this week's allowance. "
        f"Cap this resource to about {cap_minutes} minutes and complete only the modules directly needed for the weekly project."
    )


def _reviewed_use_strategy(
    source_type: str,
    focus: str,
    cap_minutes: int,
) -> str:
    if source_type == "Official documentation":
        return (
            f"Time-box to about {cap_minutes} minutes: skim the overview first, then focus on the exact APIs and patterns needed for {focus.lower()}."
        )
    if source_type == "GitHub repository":
        return (
            f"Time-box to about {cap_minutes} minutes: read the README, run or inspect the project quickly, and borrow one implementation pattern for your own work."
        )
    return (
        f"Time-box to about {cap_minutes} minutes: complete only the lesson segments that unlock this week's project and skip remaining optional content."
    )


def _review_resource(
    resource: Dict[str, Any],
    resource_index: int,
    week_focus: str,
    hours_per_week: int,
) -> Dict[str, Any]:
    query = str(resource.get("search_query") or resource.get("title") or "").strip()
    if not query:
        query = f"{_compact_query_phrase(week_focus, max_terms=8) or 'core concepts'} practical tutorial free"
    resource["search_query"] = query
    resource["title"] = str(resource.get("title") or query)
    resource["url"] = str(resource.get("url") or "")
    resource["source_type"] = str(resource.get("source_type") or _resource_source_type(query))
    if resource["source_type"] == "Web":
        resource["source_type"] = "Free web resource"

    # Enforce a free-first quality floor before final review metadata is generated.
    if resource["source_type"] == "Free web resource" or _has_paid_signal(
        query,
        str(resource.get("title") or ""),
        str(resource.get("url") or ""),
    ):
        resource.update(_trusted_free_resource_template(week_focus, resource_index))
        resource["url"] = ""

    resource = _ensure_direct_resource_url(resource)

    base_access_note = str(resource.get("access_note") or "").split(" Quality review:")[0].strip()
    cap_minutes = _resource_cap_minutes(hours_per_week, resource_index)
    estimated_minutes = _estimate_resource_minutes(resource)

    if estimated_minutes > int(cap_minutes * 1.5):
        quickstart = _trusted_free_resource_template(week_focus, resource_index)
        quickstart["search_query"] = f"{quickstart['search_query']} focused module"
        quickstart["title"] = quickstart["search_query"]
        quickstart["access_note"] = (
            "Adjusted to a shorter high-quality free option so it can realistically fit this week's time cap."
        )
        resource.update(quickstart)
        resource["url"] = ""
        resource = _ensure_direct_resource_url(resource)
        estimated_minutes = _estimate_resource_minutes(resource)

    quality_score = _resource_quality_score(resource)
    if quality_score < 3:
        resource.update(_trusted_free_resource_template(week_focus, resource_index))
        resource["url"] = ""
        resource = _ensure_direct_resource_url(resource)
        estimated_minutes = _estimate_resource_minutes(resource)
        quality_score = _resource_quality_score(resource)
    quality_label = _quality_label(quality_score)

    if not base_access_note:
        base_access_note = "Resource selected for this week's skill objective."
    resource["access_note"] = (
        f"{base_access_note} Quality review: {quality_label} confidence source; "
        f"timeline cap is about {cap_minutes} minutes this week."
    )

    why_base = str(resource.get("why_this_resource") or "").split(" Quality signal:")[0].strip()
    if why_base:
        resource["why_this_resource"] = f"{why_base} Quality signal: {quality_label} confidence for this stage of the roadmap."
    else:
        resource["why_this_resource"] = (
            f"This resource supports {week_focus.lower()} and was selected after a quality/timeline review. "
            f"Quality signal: {quality_label} confidence for this stage of the roadmap."
        )

    if not str(resource.get("contribution_to_path") or "").strip():
        resource["contribution_to_path"] = (
            "This contributes directly to this week's project output and reinforces the progression into later weeks."
        )
    if not str(resource.get("primary_focus") or "").strip():
        resource["primary_focus"] = _resource_primary_focus(resource["source_type"], week_focus, query)

    resource["time_fit"] = _reviewed_time_fit(estimated_minutes, cap_minutes, resource["source_type"])
    resource["use_strategy"] = _reviewed_use_strategy(resource["source_type"], week_focus, cap_minutes)
    return resource


def _review_roadmap_for_quality_and_timeline(
    roadmap: Dict[str, Any],
    normalized_input: Dict[str, Any],
    provider: str,
    allow_reorder: bool = True,
) -> Dict[str, Any]:
    reviewed = copy.deepcopy(roadmap)
    hours_per_week = _average_hours_per_week(str(normalized_input.get("time_available_per_week", "6-8 hours")))
    weekly_breakdown = reviewed.get("weekly_breakdown", [])

    for week in weekly_breakdown:
        resources = list(week.get("resources", []))
        if allow_reorder:
            resources = sorted(resources, key=lambda item: _resource_source_rank(str(item.get("source_type") or "")))
        week["resources"] = [
            _review_resource(resource, index, str(week.get("focus", "this week")), hours_per_week)
            for index, resource in enumerate(resources)
        ]
        week["execution_plan"] = (
            "Follow this order: resource 1 for fundamentals, resource 2 for guided practice, and resource 3 for transfer into your project. "
            "Priority is high-quality free sources first. Respect each resource's time cap and move remaining time to the hands-on implementation."
        )
        week["time_budget"] = _time_budget_copy(hours_per_week)

    base_adjustment = str(reviewed.get("adjustment_log", "")).split(" Resource review pass:")[0].strip()
    reviewed["adjustment_log"] = (
        f"{base_adjustment} Resource review pass: prioritized high-quality free sources, replaced weaker options with trusted free alternatives when needed, reviewed organization, and enforced timeline caps for about {hours_per_week} hours/week in {provider} mode."
    ).strip()
    return reviewed


def _resource_role(resource_index: int) -> str:
    if resource_index == 0:
        return "foundation"
    if resource_index == 1:
        return "guided practice"
    return "project transfer"


def _time_budget_copy(hours_per_week: int) -> str:
    if hours_per_week <= 4:
        return "About 3-4 hours total: 45 minutes on orientation, 60-75 minutes on the clearest teaching resource, 60-75 minutes on the project, and 20 minutes to capture notes and blockers."
    if hours_per_week <= 7:
        return "About 5-7 hours total: 1.5 hours on fundamentals, 1.5-2 hours on guided learning, 2-2.5 hours on the project, and 30 minutes for review and cleanup."
    if hours_per_week <= 10:
        return "About 8-10 hours total: 2 hours on fundamentals, 2-3 hours on guided practice, 3-4 hours on the project, and about 1 hour on notes, review, and polish."
    return "10+ hours total: use 2-3 hours on fundamentals, 3 hours on guided practice, 4+ hours on implementation, and keep at least 1 hour for reflection, debugging, and documentation."


def _resource_time_fit(source_type: str, hours_per_week: int) -> str:
    if "Official documentation" == source_type:
        if hours_per_week <= 4:
            return "This fits the week if you read only the getting-started, tutorial, or one targeted concept section rather than trying to finish the full documentation set."
        return "This fits because official docs work well as a selective reference: focus on the sections tied to this week's deliverable instead of reading the entire manual."
    if source_type in {"YouTube tutorial", "freeCodeCamp", "Coursera free tier", "MIT OpenCourseWare", "Khan Academy", "fast.ai"}:
        if hours_per_week <= 4:
            return "This is only realistic if you use it as a partial module: watch or complete the segment that matches this week's focus and stop once you can reproduce the key workflow."
        if hours_per_week <= 7:
            return "This is doable if you treat it as a curated slice, usually the first 60-120 minutes or one module that supports the weekly project."
        return "This fits well if you still stay selective and prioritize the lessons that unlock the week's project fastest."
    if source_type == "GitHub repository":
        return "This is doable because you are not meant to study every file. Read the README, run the project, inspect the key folders, and borrow one implementation pattern."
    return "This should fit if you use it as a focused reference rather than trying to consume everything on the page."


def _resource_primary_focus(source_type: str, focus: str, query: str) -> str:
    if source_type == "Official documentation":
        return f"Focus on the core concepts and commands that directly support {focus.lower()}."
    if source_type == "GitHub repository":
        return f"Focus on project structure, setup steps, and one pattern you can adapt into your own work related to {focus.lower()}."
    if source_type in {"YouTube tutorial", "freeCodeCamp", "Coursera free tier", "MIT OpenCourseWare", "Khan Academy", "fast.ai"}:
        return f"Focus on reproducing the workflow shown in the resource, not passively consuming the whole lesson sequence, so it directly strengthens {focus.lower()}."
    return f"Focus on the sections most closely tied to {focus.lower()} and ignore tangential material."


def _resource_use_strategy(source_type: str, hours_per_week: int, focus: str) -> str:
    if source_type == "Official documentation":
        return "Use it first for terminology and exact behavior, then come back while building when you need confirmation on syntax, APIs, or best practices."
    if source_type == "GitHub repository":
        return "Skim the README first, run or inspect the project quickly, then borrow one small structure, test pattern, or implementation idea for your own artifact."
    if hours_per_week <= 4:
        return f"Use it as a narrowly scoped lesson for this week only: stop after the portion that lets you explain and apply {focus.lower()} in your project."
    return f"Use it as guided practice after the docs, then translate the key steps into your own project so the learning sticks."


def _resource_why_and_contribution(
    source_type: str,
    role: str,
    focus: str,
    hands_on_project: str,
) -> Tuple[str, str]:
    if role == "foundation":
        why = f"This resource was chosen because it is the fastest reliable anchor for learning {focus.lower()} without building on shaky assumptions."
        contribution = f"It gives you the baseline understanding needed to complete the week's project: {hands_on_project}"
    elif role == "guided practice":
        why = f"This resource was chosen to turn the concepts in {focus.lower()} into a step-by-step example you can follow with less ambiguity."
        contribution = "It helps bridge the gap between theory and implementation so you can move into hands-on work faster."
    else:
        why = f"This resource was chosen so you can see how the week's ideas appear in realistic code, project structure, or portfolio-style work."
        contribution = "It contributes to the path by helping you transfer the week's learning into a job-relevant artifact instead of stopping at tutorials."

    if source_type == "GitHub repository":
        why += " GitHub examples are especially useful here because they expose file layout, naming, and execution flow."
    elif source_type == "Official documentation":
        why += " Official docs are especially valuable here because they are less likely to teach outdated patterns."
    return why, contribution


def _build_resource_entry(
    query: str,
    resource_index: int,
    focus: str,
    hands_on_project: str,
    hours_per_week: int,
) -> Dict[str, str]:
    source_type = _resource_source_type(query)
    role = _resource_role(resource_index)
    why_this_resource, contribution_to_path = _resource_why_and_contribution(
        source_type,
        role,
        focus,
        hands_on_project,
    )
    return {
        "search_query": query,
        "title": query,
        "url": "",
        "source_type": source_type,
        "access_note": "Planned search query. Browse mode will try to replace this with a direct free resource when possible.",
        "why_this_resource": why_this_resource,
        "contribution_to_path": contribution_to_path,
        "primary_focus": _resource_primary_focus(source_type, focus, query),
        "time_fit": _resource_time_fit(source_type, hours_per_week),
        "use_strategy": _resource_use_strategy(source_type, hours_per_week, focus),
    }


def _build_week_guidance(
    week: Dict[str, Any],
    week_number: int,
    total_weeks: int,
    normalized_input: Dict[str, Any],
    hours_per_week: int,
) -> Dict[str, Any]:
    specialization = normalized_input.get("domain_specialization") or normalized_input.get("secondary_goal")
    target_job = normalized_input.get("target_job_title")
    if week_number == 1:
        why_this_week = f"This opening week establishes the vocabulary and working habits needed to make the rest of the {total_weeks}-week plan feel manageable rather than fragmented."
    elif week_number == total_weeks:
        why_this_week = "This final week is about converting your progress into something demonstrable, reviewable, and easier to talk about in interviews or portfolio conversations."
    elif specialization and week_number >= max(2, (total_weeks + 1) // 2):
        why_this_week = f"This week pushes the roadmap toward {specialization}, so the second half of the plan feels closer to your actual end goal instead of staying generic."
    elif target_job and week_number >= max(2, (total_weeks + 1) // 2):
        why_this_week = f"This week leans more explicitly toward {target_job} expectations so your practice starts sounding and looking like the target role."
    else:
        why_this_week = "This week builds on the earlier foundation and adds a slightly more applied layer so the roadmap progresses in a steady, realistic sequence."

    execution_plan = (
        "Start with the first resource to get the terminology right, move to the second resource for guided repetition, "
        "then use the third resource while building the hands-on project so the learning turns into output."
    )
    week["why_this_week"] = why_this_week
    week["priority_focus"] = f"Focus most on being able to explain and apply {week['focus'].lower()} in the hands-on project, even if you skip lower-value optional material from longer resources."
    week["time_budget"] = _time_budget_copy(hours_per_week)
    week["execution_plan"] = execution_plan
    return week


def _merge_module_chunk(
    chunk: List[Dict[str, Any]],
    week_number: int,
    normalized_input: Dict[str, Any],
    hours_per_week: int,
) -> Dict[str, Any]:
    focus = " and ".join(module["focus"] for module in chunk)
    resource_queries: List[str] = []
    for module in chunk:
        for resource in module["resources"]:
            formatted = _format_string_template(resource, normalized_input)
            if formatted not in resource_queries:
                resource_queries.append(formatted)
            if len(resource_queries) == 3:
                break
        if len(resource_queries) == 3:
            break

    project_parts = [_format_string_template(module["hands_on_project"], normalized_input) for module in chunk]
    if len(project_parts) == 1:
        project = project_parts[0]
    else:
        project = "Combine these outcomes into one scoped artifact: " + " Then ".join(project_parts[:2])

    return {
        "week": week_number,
        "focus": focus,
        "resources": [
            _build_resource_entry(query, index, focus, project, hours_per_week)
            for index, query in enumerate(resource_queries[:3])
        ],
        "hands_on_project": project,
    }


def _chunk_modules(modules: List[Dict[str, Any]], week_count: int) -> List[List[Dict[str, Any]]]:
    if week_count <= 0:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    total = len(modules)
    for index in range(week_count):
        start = round(index * total / week_count)
        end = round((index + 1) * total / week_count)
        chunk = modules[start:end]
        if not chunk:
            chunk = [modules[min(start, total - 1)]]
        chunks.append(chunk)
    return chunks


def _specialize_week(
    week: Dict[str, Any],
    week_number: int,
    total_weeks: int,
    normalized_input: Dict[str, Any],
) -> Dict[str, Any]:
    specialization = normalized_input.get("domain_specialization") or normalized_input.get("secondary_goal")
    target_job = normalized_input.get("target_job_title")
    if week_number < max(2, (total_weeks + 1) // 2):
        return week
    if not specialization and not target_job:
        return week

    pivot = specialization or target_job
    week["focus"] = f"{week['focus']} tailored toward {pivot}"
    week["resources"][2]["search_query"] = f"GitHub {normalized_input['topic']} {pivot} portfolio project example"
    week["resources"][2]["title"] = week["resources"][2]["search_query"]
    week["resources"][2]["source_type"] = "GitHub repository"
    week["resources"][2]["why_this_resource"] = (
        f"This resource was chosen to make the specialization toward {pivot} concrete through a portfolio-style example rather than leaving it as a theoretical pivot."
    )
    week["resources"][2]["contribution_to_path"] = (
        f"It contributes to the path by showing how the week's concepts can be reframed in language and artifacts that better match {pivot} work."
    )
    week["resources"][2]["primary_focus"] = (
        f"Focus on the project decisions, naming, and deliverables that make the work look relevant to {pivot}."
    )
    week["resources"][2]["time_fit"] = (
        "This stays realistic if you study the README, repo structure, and one or two relevant implementation areas instead of trying to reverse-engineer the full project."
    )
    week["resources"][2]["use_strategy"] = (
        f"Use it near the end of the week to adapt your own project wording, scope, and implementation choices toward {pivot} expectations."
    )
    week["hands_on_project"] = f"{week['hands_on_project']} Tie the result to {pivot} requirements and vocabulary."
    return week


def _apply_schedule_intensity(week: Dict[str, Any], hours_per_week: int) -> Dict[str, Any]:
    if hours_per_week <= 4:
        week["hands_on_project"] = f"Keep scope intentionally small: {week['hands_on_project']}"
        week["priority_focus"] = f"{week['priority_focus']} Keep the scope narrow and prioritize one clear win over completeness."
    elif hours_per_week >= 10:
        week["hands_on_project"] = f"Stretch goal: {week['hands_on_project']}"
        week["priority_focus"] = f"{week['priority_focus']} Use the extra time for cleanup, testing, or portfolio polish after the main outcome is working."
    return week


def _build_extension_modules(profile: Dict[str, Any], normalized_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    topic = normalized_input["topic"]
    specialization = (
        normalized_input.get("domain_specialization")
        or normalized_input.get("secondary_goal")
        or normalized_input.get("target_job_title")
        or "career-aligned specialization"
    )
    return [
        {
            "focus": "Specialization sprint and domain-specific implementation",
            "resources": [
                f"YouTube {topic} {specialization} project tutorial free",
                f"GitHub {topic} {specialization} case study repository",
                f"{topic} official documentation advanced tutorial {specialization}",
            ],
            "hands_on_project": f"Extend your strongest project with a specialization-focused feature set for {specialization}.",
        },
        {
            "focus": "Portfolio refinement, documentation, and presentation quality",
            "resources": [
                f"GitHub {topic} portfolio examples documentation README",
                f"YouTube {topic} portfolio review tutorial free",
                f"freeCodeCamp {topic} portfolio project storytelling guide",
            ],
            "hands_on_project": "Refine documentation, screenshots, architecture notes, and the project narrative so a hiring manager can scan it quickly.",
        },
        {
            "focus": "Interview readiness, troubleshooting practice, and gap closing",
            "resources": [
                f"GitHub {topic} interview questions repository",
                f"YouTube {topic} code review troubleshooting tutorial free",
                f"{topic} best practices checklist official guide",
            ],
            "hands_on_project": "Run a self-review on the capstone, fix the weakest area, and write down the tradeoffs you would discuss in an interview.",
        },
    ]


def _validate_roadmap(roadmap: Dict[str, Any]) -> None:
    required_keys = {"industry_insight", "weekly_breakdown", "adjustment_log"}
    if set(roadmap.keys()) != required_keys:
        raise ValueError("Roadmap does not match the required top-level schema.")
    if not isinstance(roadmap["industry_insight"], str):
        raise ValueError("industry_insight must be a string.")
    if roadmap["industry_insight"].count(".") < 2:
        raise ValueError("industry_insight must contain two sentences.")
    if not isinstance(roadmap["weekly_breakdown"], list) or not roadmap["weekly_breakdown"]:
        raise ValueError("weekly_breakdown must be a non-empty list.")
    for week in roadmap["weekly_breakdown"]:
        if set(week.keys()) != {
            "week",
            "focus",
            "why_this_week",
            "priority_focus",
            "time_budget",
            "execution_plan",
            "resources",
            "hands_on_project",
        }:
            raise ValueError("A weekly item is missing required keys.")
        if len(week["resources"]) != 3:
            raise ValueError("Each week must contain exactly three resources.")
        for resource in week["resources"]:
            if set(resource.keys()) != {
                "search_query",
                "title",
                "url",
                "source_type",
                "access_note",
                "why_this_resource",
                "contribution_to_path",
                "primary_focus",
                "time_fit",
                "use_strategy",
            }:
                raise ValueError("A resource item is missing required keys.")


def generate_roadmap_offline(user_input: Any) -> Dict[str, Any]:
    normalized_input, assumptions = normalize_request(user_input)
    profile = _choose_profile(normalized_input)
    week_count = _extract_week_count(str(normalized_input["schedule_length"]))
    hours_per_week = _average_hours_per_week(str(normalized_input["time_available_per_week"]))

    modules = copy.deepcopy(profile["modules"])
    extension_modules = _build_extension_modules(profile, normalized_input)
    while len(modules) < week_count:
        next_extension = extension_modules[(len(modules) - len(profile["modules"])) % len(extension_modules)]
        modules.append(copy.deepcopy(next_extension))

    weekly_breakdown: List[Dict[str, Any]] = []
    for week_number, chunk in enumerate(_chunk_modules(modules, week_count), start=1):
        week = _merge_module_chunk(chunk, week_number, normalized_input, hours_per_week)
        week = _specialize_week(week, week_number, week_count, normalized_input)
        week = _build_week_guidance(week, week_number, week_count, normalized_input, hours_per_week)
        week = _apply_schedule_intensity(week, hours_per_week)
        weekly_breakdown.append(week)

    industry_insight = (
        f"{profile['trend_sentence']} "
        "The tool choices and sequence reflect a best-effort offline inference of current hiring patterns rather than live-verified job-market data."
    )

    adjustment_parts = [
        f"Built a local-first roadmap using the {profile['skill_name']} profile.",
        f"Matched the plan to {week_count} weeks at about {hours_per_week} hours per week.",
    ]
    if normalized_input.get("target_job_title"):
        adjustment_parts.append(f"Vocabulary and projects were tuned toward the target role {normalized_input['target_job_title']}.")
    if normalized_input.get("domain_specialization"):
        adjustment_parts.append(f"The second half pivots toward the specialization {normalized_input['domain_specialization']}.")
    if normalized_input.get("secondary_goal"):
        adjustment_parts.append(f"The later weeks also support the secondary goal {normalized_input['secondary_goal']}.")
    if normalized_input.get("custom_modifications"):
        adjustment_parts.append(f"Custom modifications were considered: {normalized_input['custom_modifications']}.")
    if assumptions:
        adjustment_parts.append("Assumptions: " + " ".join(assumptions))
    adjustment_parts.append("No live market browsing was used, so demand alignment is an informed offline estimate.")

    roadmap = {
        "industry_insight": industry_insight,
        "weekly_breakdown": weekly_breakdown,
        "adjustment_log": " ".join(adjustment_parts),
    }
    roadmap = _review_roadmap_for_quality_and_timeline(
        roadmap=roadmap,
        normalized_input=normalized_input,
        provider="offline",
        allow_reorder=True,
    )
    _validate_roadmap(roadmap)
    return {
        "roadmap": roadmap,
        "sources": [],
        "provider": "offline",
        "normalized_input": normalized_input,
        "resource_details_by_week": {},
        "profile_trend_sentence": profile["trend_sentence"],
    }


def _split_title_url(value: str) -> Optional[Dict[str, str]]:
    if " | " not in value:
        return None
    title, url = value.split(" | ", 1)
    return {"title": title.strip(), "url": url.strip()}


def generate_roadmap_browse(
    user_input: Any,
    timeout: int = 12,
    max_workers: int = 6,
) -> Dict[str, Any]:
    base = generate_roadmap_offline(user_input)
    roadmap = copy.deepcopy(base["roadmap"])
    weekly_breakdown = roadmap["weekly_breakdown"]
    raw_queries = [resource["search_query"] for week in weekly_breakdown for resource in week["resources"]]
    discovered = discover_best_resources(raw_queries, timeout=timeout, max_workers=max_workers)

    live_hits = 0
    fallback_hits = 0
    resource_details_by_week: Dict[str, List[Dict[str, Any]]] = {}
    sources: List[Dict[str, str]] = []
    seen_urls = set()

    for week in weekly_breakdown:
        week_details: List[Dict[str, Any]] = []
        for resource in week["resources"]:
            resource_query = resource["search_query"]
            detail = discovered.get(
                resource_query,
                {"display": resource_query, "live": False, "query": resource_query},
            )
            if detail.get("live"):
                live_hits += 1
                resource["title"] = str(detail.get("title") or resource_query)
                resource["url"] = str(detail.get("url") or "")
                resource["source_type"] = str(detail.get("source_label") or resource["source_type"])
                resource["access_note"] = "Direct free resource found from current public web search results."
            else:
                fallback_hits += 1
                resource["title"] = str(detail.get("title") or resource_query)
                resource["url"] = str(detail.get("url") or "")
                if detail.get("source_label"):
                    resource["source_type"] = str(detail.get("source_label"))
                resource["access_note"] = (
                    "Validated exact fallback resource used because a direct current page for the original suggestion could not be confirmed."
                )
            week_details.append(detail)
            if detail.get("live") and resource["url"] and resource["url"] not in seen_urls:
                sources.append({"title": resource["title"], "url": resource["url"]})
                seen_urls.add(resource["url"])
        resource_details_by_week[str(week["week"])] = week_details

    first_sentence = base["profile_trend_sentence"]
    if live_hits:
        second_sentence = "Live, non-AI web discovery was used to surface current free resources from public search results and trusted learning domains."
    else:
        second_sentence = "Live, non-AI web discovery could not fetch results in this run, so the roadmap fell back to direct-site links and offline planning logic."
    roadmap["industry_insight"] = f"{first_sentence} {second_sentence}"
    roadmap["adjustment_log"] = roadmap["adjustment_log"].replace(
        "No live market browsing was used, so demand alignment is an informed offline estimate.",
        "Demand alignment used local planning plus live public-web discovery when available.",
    )
    roadmap["adjustment_log"] = (
        f"{roadmap['adjustment_log']} "
        f"Live resource discovery refreshed {live_hits} resource slots and used {fallback_hits} direct-site fallback links when needed."
    )
    roadmap = _review_roadmap_for_quality_and_timeline(
        roadmap=roadmap,
        normalized_input=base["normalized_input"],
        provider="browse",
        allow_reorder=False,
    )
    _validate_roadmap(roadmap)

    return {
        "roadmap": roadmap,
        "sources": sources,
        "provider": "browse",
        "normalized_input": base["normalized_input"],
        "resource_details_by_week": resource_details_by_week,
        "live_resource_hits": live_hits,
        "fallback_resource_hits": fallback_hits,
    }


def _generate_with_openai(
    user_input: Any,
    model: Optional[str],
    enable_web_search: bool,
    timeout: int,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    resolved_model = _default_model_for_provider("openai", model)
    request_body = build_responses_api_request(
        user_input=user_input,
        model=resolved_model,
        enable_web_search=enable_web_search,
    )
    response_json = _http_json_request(
        OPENAI_URL,
        request_body,
        {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    roadmap = _extract_json_string(_extract_output_text(response_json))
    _validate_roadmap(roadmap)
    return {
        "roadmap": roadmap,
        "sources": extract_sources(response_json),
        "raw_response": response_json,
        "request_body": request_body,
        "provider": "openai",
        "ai_provider_label": AI_PROVIDER_LABELS["openai"],
        "ai_model": resolved_model,
    }


def _generate_with_openai_compatible(
    user_input: Any,
    provider_key: str,
    model: Optional[str],
    timeout: int,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    messages = build_ai_messages(user_input)
    env_prefix = provider_key.upper().replace("_", "_")
    resolved_model = _default_model_for_provider(provider_key, model or os.getenv(f"{env_prefix}_MODEL"))
    if provider_key == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        request_body = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]},
            ],
            "stream": False,
            "format": "json",
        }
        response_json = _http_json_request(
            base_url.rstrip("/") + "/api/chat",
            request_body,
            {"Content-Type": "application/json"},
            timeout=timeout,
        )
        response_text = str((response_json.get("message") or {}).get("content") or "").strip()
    else:
        if provider_key == "deepseek":
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            endpoint = base_url.rstrip("/") + "/chat/completions"
        elif provider_key == "perplexity":
            base_url = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
            resolved_api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
            endpoint = base_url.rstrip("/") + "/chat/completions"
        else:
            base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", OPENAI_COMPATIBLE_DEFAULT_URL)
            resolved_api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
            endpoint = base_url

        is_local_endpoint = endpoint.startswith("http://127.0.0.1") or endpoint.startswith("http://localhost")
        if not resolved_api_key and not is_local_endpoint:
            raise ValueError(f"{env_prefix}_API_KEY is not set.")

        request_body = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        response_json = _http_json_request(
            endpoint,
            request_body,
            {
                **({"Authorization": f"Bearer {resolved_api_key}"} if resolved_api_key else {}),
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        response_text = _extract_openai_compatible_text(response_json)

    roadmap = _extract_json_string(response_text)
    _validate_roadmap(roadmap)
    sources = response_json.get("search_results") if provider_key == "perplexity" else []
    normalized_sources = []
    for item in sources or []:
        if isinstance(item, dict) and item.get("url"):
            normalized_sources.append({"title": str(item.get("title") or item["url"]), "url": str(item["url"])})
    return {
        "roadmap": roadmap,
        "sources": normalized_sources,
        "raw_response": response_json,
        "provider": provider_key,
        "ai_provider_label": AI_PROVIDER_LABELS[provider_key],
        "ai_model": resolved_model,
    }


def _generate_with_gemini(
    user_input: Any,
    model: Optional[str],
    timeout: int,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    resolved_model = _default_model_for_provider("gemini", model or os.getenv("GEMINI_MODEL"))
    messages = build_ai_messages(user_input)
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(resolved_model, safe="")
        + ":generateContent"
    )
    request_body = {
        "system_instruction": {"parts": [{"text": messages["system"]}]},
        "contents": [{"role": "user", "parts": [{"text": messages["user"]}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": RESPONSE_SCHEMA,
        },
    }
    response_json = _http_json_request(
        endpoint,
        request_body,
        {
            "x-goog-api-key": resolved_api_key,
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    roadmap = _extract_json_string(_extract_gemini_text(response_json))
    _validate_roadmap(roadmap)
    return {
        "roadmap": roadmap,
        "sources": [],
        "raw_response": response_json,
        "provider": "gemini",
        "ai_provider_label": AI_PROVIDER_LABELS["gemini"],
        "ai_model": resolved_model,
    }


def _generate_with_anthropic(
    user_input: Any,
    model: Optional[str],
    timeout: int,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    resolved_model = _default_model_for_provider("anthropic", model or os.getenv("ANTHROPIC_MODEL"))
    messages = build_ai_messages(user_input)
    request_body = {
        "model": resolved_model,
        "max_tokens": 4000,
        "system": messages["system"],
        "messages": [{"role": "user", "content": messages["user"]}],
    }
    response_json = _http_json_request(
        "https://api.anthropic.com/v1/messages",
        request_body,
        {
            "x-api-key": resolved_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    roadmap = _extract_json_string(_extract_anthropic_text(response_json))
    _validate_roadmap(roadmap)
    return {
        "roadmap": roadmap,
        "sources": [],
        "raw_response": response_json,
        "provider": "anthropic",
        "ai_provider_label": AI_PROVIDER_LABELS["anthropic"],
        "ai_model": resolved_model,
    }


def generate_roadmap_with_ai(
    user_input: Any,
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_web_search: bool = True,
    timeout: int = 120,
) -> Dict[str, Any]:
    if provider == "openai":
        return _generate_with_openai(
            user_input=user_input,
            model=model,
            enable_web_search=enable_web_search,
            timeout=timeout,
            api_key=api_key,
        )
    if provider in {"ollama", "openai_compatible", "deepseek", "perplexity"}:
        return _generate_with_openai_compatible(
            user_input=user_input,
            provider_key=provider,
            model=model,
            timeout=timeout,
            api_key=api_key,
        )
    if provider == "gemini":
        return _generate_with_gemini(
            user_input=user_input,
            model=model,
            timeout=timeout,
            api_key=api_key,
        )
    if provider == "anthropic":
        return _generate_with_anthropic(
            user_input=user_input,
            model=model,
            timeout=timeout,
            api_key=api_key,
        )
    raise ValueError("Unknown AI provider.")


def generate_roadmap(
    user_input: Any,
    provider: str = DEFAULT_PROVIDER,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_web_search: bool = True,
    timeout: int = 120,
) -> Dict[str, Any]:
    if provider == "browse":
        return generate_roadmap_browse(user_input)
    if provider == "offline":
        return generate_roadmap_offline(user_input)
    if provider in {"ollama", "openai", "openai_compatible", "gemini", "anthropic", "deepseek", "perplexity"}:
        result = generate_roadmap_with_ai(
            user_input=user_input,
            provider=provider,
            api_key=api_key,
            model=model,
            enable_web_search=enable_web_search,
            timeout=timeout,
        )
        normalized_input, _ = normalize_request(user_input)
        result["roadmap"] = _review_roadmap_for_quality_and_timeline(
            roadmap=result["roadmap"],
            normalized_input=normalized_input,
            provider=provider,
            allow_reorder=True,
        )
        _validate_roadmap(result["roadmap"])
        result.setdefault("normalized_input", normalized_input)
        return result
    raise ValueError("provider must be 'browse', 'offline', 'ollama', 'openai', 'openai_compatible', 'gemini', 'anthropic', 'deepseek', or 'perplexity'.")


def _load_input(path: Optional[str]) -> Any:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    raw = sys.stdin.read()
    if not raw:
        raise ValueError("Provide JSON via stdin or pass a path to a JSON file.")

    raw = raw.strip()
    if not raw:
        raise ValueError("Input was empty.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learning roadmaps with browsing, offline logic, or optional AI providers.")
    parser.add_argument("input", nargs="?", help="Path to a JSON file. If omitted, stdin is used.")
    parser.add_argument("--model", default=None, help="Model override for AI providers.")
    parser.add_argument(
        "--provider",
        choices=["browse", "offline", "ollama", "openai", "openai_compatible", "gemini", "anthropic", "deepseek", "perplexity"],
        default=DEFAULT_PROVIDER,
        help="Roadmap provider. Default: browse",
    )
    parser.add_argument(
        "--preview-openai-payload",
        action="store_true",
        help="Print the OpenAI request payload instead of generating a roadmap.",
    )
    parser.add_argument(
        "--preview-ai-messages",
        action="store_true",
        help="Print the normalized AI system and user messages instead of generating a roadmap.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Backward-compatible alias for --provider openai.",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable the OpenAI web_search tool when using provider=openai.",
    )
    args = parser.parse_args()

    user_input = _load_input(args.input)
    provider = "openai" if args.execute else args.provider

    if args.preview_openai_payload:
        print(
            json.dumps(
                build_responses_api_request(
                    user_input,
                    model=_default_model_for_provider("openai", args.model),
                    enable_web_search=not args.no_web_search,
                ),
                indent=2,
            )
        )
        return
    if args.preview_ai_messages:
        print(json.dumps(build_ai_messages(user_input), indent=2))
        return

    result = generate_roadmap(
        user_input=user_input,
        provider=provider,
        model=args.model,
        enable_web_search=not args.no_web_search,
    )
    print(json.dumps(result["roadmap"], indent=2))


if __name__ == "__main__":
    main()
