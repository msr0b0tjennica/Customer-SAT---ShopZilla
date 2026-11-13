**KLA CORPORATION  |  ERC / CEG — DT / IDEAS PLATFORM** 

# **KEDS AI Submission — Test Automation Architecture & Execution Plan (v2.0)** 

Azure DevOps (ADO) Test Plans Integration, CI/CD Workflows & Framework Blueprint 

|**Plan Attribute**|**Specification**|
|---|---|
|**Target Application**|KEDS AI Submission Chatbot (v0.1) —<br>https://test-dt-infra.dev.dias.kla.com/hosted/eng_reqs|
|**Test Basis & Scope**|KEDS User Stories & Test Cases — Submission (v2.0 / v2.1), BRD Sections 2.4–2.5|
|**Primary Automation**|Playwright for Python + Pytest (Browser UI End-to-End)|
|**CI/CD Orchestrator**|Azure DevOps YAML Pipelines (Self-Hosted KLA Intranet Agents)|
|**ADO Integration**|pytest-azurepipelines mapped directly to ADO Work Items / Test Cases via markers|
|**Prepared By / Date**|Jennica Bhaskaran (QA & Test Automation, ERC/CEG Team)  |  November 13, 2026|



### **v2.0 Update: Transitioning to ADO Continuous Testing** 

This version of the test plan formally transitions the architecture from localized standalone scripts into a fully integrated Azure DevOps (ADO) Continuous Testing Quality Gate. 

Instead of manual execution, tests are directly mapped to ADO Work Items. Test status, failure screenshots, and Playwright traces are automatically synced to Sasha’s ADO Test Plans dashboard. 

## **1.  Test Automation Strategy & ADO Integration** 

The KEDS AI Submission automation suite follows a clean separation of concerns, heavily augmented by native Azure DevOps integrations to ensure manual test cases (which currently display as 'Not Run' in ADO) are programmatically executed and tracked. 

- **Tier 1 — Declarative Test Configuration & ADO Input Repository:** Test datasets (TD-01, TD-09) and the golden Master Excel baselines are now committed directly to the ADO Git Repository. When the pipeline triggers, it pulls these assets down to ephemeral agents so data is universally synchronized. 

- **Tier 2 & 3 — Page Object Model & Validation Engine:** Reusable Playwright interactions and OpenPyXL workbook validation logic run efficiently in headless mode. 

- **Tier 4 — Pytest Runner & ADO Test Case Mapping (pytest-azurepipelines):** Code is directly linked to ADO Work Item IDs via custom pytest markers. When the suite executes, it utilizes an ADO Access Token to update the manual test cases dynamically. 

KEDS AI Submission (v0.1) — ADO Integrated Test Architecture v2.0 

## **2.  Tooling Stack & CI/CD Enhancements** 

|**Tool / Library**|**Role in Framework**|**Technical Justification**|**ADO Context**|
|---|---|---|---|
|**Playwright & Pytest**|Primary UI|Built-in auto-waiting, browser storage|Captures DOM traces and|
|**(pytest-playwright)**|Automation &<br>Runner|state for KLA SSO, and robust<br>parametrization.|failure PNGs natively, which are<br>piped directly to ADO.|
|**pytest-azurepipelines**|ADO Test Plans|Creates a direct bridge between|Automatically associates the|
||Integrator|Python test executions and the Azure<br>DevOps REST API.|Python test_ function with the<br>ADO Test Case Work Item.|
|**OpenPyXL / python-docx**|Output Validation<br>Engine|Inspects exact Excel cell outputs, row<br>counts, and downloaded Word/PDF|Operates on ephemeral pipeline<br>copies of the baseline file,|
|||reports.|leaving production clean.|



## **3.  Code Level: Mapping Python to ADO Test Cases** 

To bridge the gap between code and management tracking, each test function is explicitly decorated with its corresponding ADO Work Item ID. 

**Listing 1: Using pytest-azurepipelines to map Python functions to ADO Test Cases** 

import pytest from pages.submission_page import KedsSubmissionPage # Assuming KEDS-TC-015 is mapped to ADO Work Item #45102 @pytest.mark.ado_test_case("45102") def test_keds_tc_015_type_and_paste_requirement(page): submission_page = KedsSubmissionPage(page) submission_page.open_and_start_submission() # 1. Enter requirement text submission_page.enter_requirement_text( title="TD-01 Title", detail_description="TD-02 Detail" ) # (Assertions follow...) 

## **4.  Azure DevOps Pipeline & Execution Workflow** 

The execution of tests shifts entirely away from local environments to three standard pipeline triggers: 

- **1. Pull Request (CI Gate):** Runs automatically when developers push new code to KEDS repo. Blocks merges if a test fails. 

- **2. Nightly Regression Schedule:** Executes the full 83-case suite at 2:00 AM on test-dt-infra, giving Sasha a clean status pie chart each morning. 

KLA Corporation | ERC / CEG Quality Assurance & Test Automation   |   Page 2 

KEDS AI Submission (v0.1) — ADO Integrated Test Architecture v2.0 

- **3. Manual Trigger from Test Plans:** QA can right-click a suite in ADO Test Plans and select 'Run Test' to dispatch a targeted job to the KLA self-hosted agents. 

**Listing 2: Updated Azure DevOps YAML Pipeline featuring ADO API Token & Attachment Publishing** 

# azure-pipelines.yml trigger: - main schedules: - cron: "0 2 * * 1-5" displayName: "Nightly KEDS Regression" always: true pool: name: 'KLA-Intranet-QA-Agents' # Must run on-premise to access test-dt-infra steps: - task: UsePythonVersion@0 inputs: versionSpec: '3.11' - script: | pip install pytest pytest-playwright pytest-azurepipelines openpyxl playwright install chromium displayName: 'Install Dependencies' # The --test-run-title flag natively integrates with ADO dashboards - script: | pytest tests/ --junitxml=TEST-keds-results.xml --tracing=retain-on-failure displayName: 'Run Pytest & Generate ADO Report' env: SYSTEM_ACCESSTOKEN: $(System.AccessToken) # Critical for ADO Test Plans API auth - task: PublishTestResults@2 inputs: testResultsFormat: 'JUnit' testResultsFiles: 'TEST-keds-results.xml' testRunTitle: 'KEDS Automation Run - $(Build.BuildId)' publishRunAttachments: true # Attaches Playwright failure traces to the ADO Work Item condition: succeededOrFailed() 

## **5.  ADO Configuration Checklist** 

To fully activate this integration, the following one-time setup steps are required within the ADO UI: 

- **Associate Automation:** Open each KEDS Test Case Work Item (e.g., KEDS-TC-001) in ADO, navigate to the 'Associated Automation' tab, and paste the exact name of the Python function (e.g., test_keds_tc_001_login). 

- **Pipeline Permissions:** Ensure the Build Service Account running the YAML pipeline has the 'Edit work items in this node' and 'Manage test plans' permissions so it can programmatically pass/fail the cases. 

KLA Corporation | ERC / CEG Quality Assurance & Test Automation   |   Page 3 

KEDS AI Submission (v0.1) — ADO Integrated Test Architecture v2.0 

- **Secret Variable Groups:** Configure the ADO Library Variable Group 'KEDS-QA-Secrets' to store the TD-09 test user credentials securely. 

KLA Corporation | ERC / CEG Quality Assurance & Test Automation   |   Page 4 

