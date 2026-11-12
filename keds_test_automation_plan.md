# KEDS AI Submission — Test Automation Architecture & Execution Plan

**KLA CORPORATION | ERC / CEG — DT / IDEAS PLATFORM**
*Hybrid Playwright UI End-to-End & Black-Box API Automation Blueprint Integrated with Azure DevOps*

| Plan Attribute | Specification |
| :--- | :--- |
| **Target Application** | KEDS AI Submission Chatbot (v0.1) — `https://test-dt-infra.dev.dias.kla.com/hosted/eng_reqs` |
| **Test Basis & Scope** | KEDS User Stories & Test Cases — Submission (v2.0 / v2.1), BRD Sections 2.4–2.5, Master Excel Output |
| **Primary Automation** | Playwright for Python + Pytest (Browser UI End-to-End & File Download Verification) |
| **Secondary / Backup** | Black-Box REST API & Contract Testing via Python `requests` / `httpx` + Pytest + Pydantic |
| **CI/CD & Test Tracking** | Azure DevOps (ADO) Git Repos, YAML Pipelines (Self-Hosted KLA Agent), and JUnit Test Plans |
| **Prepared By / Date** | Jennica Bhaskaran (QA & Test Automation, ERC/CEG Team) \| November 12, 2026 |

---

### Architectural Premise Confirmation: UI Automation vs. API Requirements

1. **Primary Layer (Playwright UI):** You do NOT need backend API endpoints to automate browser workflows in Playwright. Playwright drives a real Chromium/Edge browser instance using only the hosted KEDS application URL, KLA intranet/VPN connectivity, and valid KLA Single Sign-On (SSO) credentials.
2. **Backup Layer (Black-Box API + Pytest):** Retaining a lightweight black-box API test layer provides high-speed regression coverage for network-boundary assertions (e.g., `KEDS-TC-002` off-VPN checks), endpoint health verification when UI selectors evolve in KEDS AI v0.1, and schema validation.

---

## 1. Test Automation Strategy & Four-Tier Framework Architecture

The KEDS AI Submission application automates requirement capture through a conversational web interface and persists validated records into a structured Master Excel workbook (`Rakesh KEDS Requirements Master Excel.xlsx`) as well as downloadable submission reports. Because user acceptance criteria span browser behavior (SSO authentication, non-editable HR fields, scroll/focus stability, dropdown filtering) and backend file generation (exact Excel cell population, row heights, freeze panes, report formatting), an effective automation suite must validate both layers.

### 1.1 Four-Tier Modular Python Framework

Rather than writing monolithic scripts where selectors, test data, and assertions are mixed together, the automation codebase follows a clean four-tier separation of concerns:

*   **Tier 1 — Declarative Test Configuration & Input Data Repository (YAML / JSON / Golden Excel):** Stores environment URLs, timeout thresholds, test datasets (`TD-01`, `TD-02`, `TD-03`, `TD-09`), sample customer check sheets/PDFs, and expected outputs outside Python source code.
*   **Tier 2 — Page Object Model (POM) & Black-Box API Client Library:** Encapsulates reusable Playwright UI interactions (e.g., `open_chatbot()`, `start_new_submission()`, `fill_requirement_text()`, `select_responsible_org()`, `submit_and_download()`) alongside fallback REST API client wrappers.
*   **Tier 3 — Artifact & Output Validation Engine:** Dedicated verification modules using OpenPyXL (Excel diffing, row height, freeze panes), `python-docx` / `pypdf` (downloaded report parsing), and DOM assertion helpers.
*   **Tier 4 — Pytest Execution Runner & Azure DevOps CI/CD Pipeline:** Orchestrates test discovery, parameterization, SSO session reuse, JUnit XML reporting, and failure trace/screenshot uploads inside Azure DevOps.

---

## 2. Tooling Stack Declaration, Justification & Operational Usage

Every tool selected for this test plan is Python-native, enterprise-compatible within KLA's network environment, and directly mapped to specific verification requirements in the KEDS test suite.

| Tool / Library | Role in Framework | Technical Justification | Operational Usage in KEDS |
| :--- | :--- | :--- | :--- |
| **Playwright for Python**<br>`playwright`, `pytest-playwright` | Primary UI & E2E Browser Automation | Built-in auto-waiting eliminates flaky sleeps during AI chatbot streaming responses; native browser `storage_state` preserves KLA SSO logins; built-in download interception and Trace Viewer. | Navigates to KEDS URL, verifies SSO auto-login, enters TD-01/TD-02 text, tests long-edit focus/scroll retention, checks dropdown values, and captures downloaded reports/Excel files. |
| **Pytest**<br>`pytest`, `pytest-xdist`, `pytest-timeout` | Test Runner, Fixtures & Parameterization | Industry-standard Python runner supporting data-driven parameterization (1 function running 12 mandatory-field variants), clean setup/teardown fixtures, and native JUnit XML export. | Discovers `test_keds_tc_*.py` suites, injects isolated Master Excel copies per test, executes parameterized negative tests (`KEDS-TC-027`), and emits `junit-results.xml` for ADO. |
| **Requests / HTTPX + Pydantic** | Secondary Black-Box API & Schema Backup | Fast, headless HTTP execution without browser overhead; Pydantic enforces strict data-contract validation on extracted requirement payloads. | Verifies network isolation outside KLA VPN (`KEDS-TC-002`), runs rapid endpoint smoke checks when UI locators change, and validates structured JSON/Excel schema compliance. |
| **OpenPyXL & Pandas** | Master Excel Workbook Validation Engine | Playwright cannot inspect internal Excel formatting. OpenPyXL programmatically reads cell values, row counts, `row_dimensions.height`, and `freeze_panes`. | Compares pre-run vs. post-run Master Excel copies: verifies Columns C, D, E, F, G, M; confirms rows 9, 623, 710 are untouched (`TC-066`); checks row height & freeze panes (`TC-065`). |
| **python-docx & pypdf** | Document Fixture & Report Parser | Enables automated text, paragraph, and line-break extraction from downloaded submission reports and multi-format input test files. | Parses downloaded reports to confirm full non-truncated email addresses (`TC-005`) and line-by-line item numbers 1.3 / 1.3.1 / 1.3.1.1 (`TC-064`); stages Word/PDF files (`TC-067`). |
| **PyYAML & python-dotenv** | Configuration & Secret Management | Separates test logic from test data (`TD-01`..`TD-09`) in human-readable YAML files while keeping credentials out of Git source control. | Maps KEDS-TC IDs to test inputs and expected results in YAML; loads KLA test account credentials securely from `.env` locally or ADO Secret Variable Groups in CI. |
| **Azure DevOps**<br>Repos, Pipelines, Test Plans | CI/CD Orchestration & Reporting | Matches KLA's existing ADO project management for KEDS user stories and work items; native `PublishTestResults@2` renders pass/fail dashboards without external servers. | Hosts Git repo, runs `azure-pipelines.yml` on a KLA self-hosted agent inside corporate VPN, publishes JUnit XML results, and attaches failure screenshots/traces. |

---

## 3. Test Execution Plan Across 18 KEDS Test Cases

To demonstrate full end-to-end coverage, the execution plan below maps 18 representative test cases directly from *KEDS User Stories & Test Cases (v2.0 / v2.1)* across four functional modules.

| Test Case ID & Title | Layer | Precond. & Data | Automated Execution Steps & Verification Logic |
| :--- | :--- | :--- | :--- |
| **`KEDS-TC-001`**<br>Open chatbot on KLA network & sign in | Playwright UI | `PRE-2`<br>Tester KLA SSO account | 1. Launch Chromium with KLA network/VPN active and navigate to hosted KEDS URL.<br>2. Assert zero login prompt appears (SSO pass-through).<br>3. Click 'Start new submission' and assert Originator & Originator Email auto-populate with KLA HR name and email. |
| **`KEDS-TC-002`**<br>Chatbot unreachable outside KLA network | Black-Box API + UI | Non-KLA network / VPN off | 1. Execute `httpx.get(KEDS_URL, timeout=10)` from off-VPN runner; assert `ConnectTimeout` / `ConnectError`.<br>2. Connect KLA VPN, reload page in Playwright, and assert HTTP 200 + tester identity banner. |
| **`KEDS-TC-003`**<br>Originator details come from KLA account | Playwright UI + OpenPyXL | `PRE-1`, `PRE-2`<br>`TD-09`, `TD-01` | 1. Authenticate with `TD-09` account (HR display name differs from email).<br>2. Assert UI Originator shows HR name and Originator Email shows exact KLA email.<br>3. Submit `TD-01` and inspect Master Excel: assert Column D == Originator Email and Column E == Originator. |
| **`KEDS-TC-004`**<br>Originator details can't be edited | Playwright UI | `PRE-2`<br>`TD-09` strings | 1. Attempt to locate editable input or fill 'John Smith' into Originator and 'happy.user@klatoday.com' into Originator Email; assert fields are read-only / non-editable.<br>2. Navigate to Field Review and assert Created Date, Originator, and Originator Email lack a 'Confirm' button. |
| **`KEDS-TC-005`**<br>E-mail address shown in full everywhere | Playwright UI + OpenPyXL + Report | `PRE-1`, `PRE-2`<br>`TD-09` (long email), `TD-01` | 1. Complete submission with long KLA email and open Review; assert full email string has no ellipsis ('...').<br>2. Submit and assert full email appears in UI 'Submission Report and Workbook Updated', 'Open report', downloaded report file, and Master Excel Column D. |
| **`KEDS-TC-006`**<br>Created Date is the submission time | Playwright UI + OpenPyXL | `PRE-1`, `PRE-2`<br>`TD-01` | 1. Start submission and record timestamp `T1`.<br>2. Enter `TD-01`, wait >= 5 min, click 'Confirm all', record `T2`.<br>3. Wait >= 2 min, click 'Submit', record `T3`.<br>4. Assert Created Date in Submission Report and Master Excel Column C matches `T3` to the minute (not `T1` or `T2`). |
| **`KEDS-TC-013`**<br>Submission page contains only KEDS fields | Playwright UI | `PRE-2`<br>None | 1. Open new submission and assert count == 0 for legacy elements: '1 Basic request', 'Shared Excel...', 'Requester Name', 'Requestor email', and 'Additional field hints / context'.<br>2. Proceed to next step and assert only 'Continue' button exists (no 'Continue and update workbook'). |
| **`KEDS-TC-014`**<br>Titles are capitalized like the Excel | Playwright UI + OpenPyXL | `PRE-2`, `PRE-3`<br>Master Excel headers | 1. Traverse all submission steps and scrape all visible section and field labels.<br>2. Assert Title-Case formatting (every word capitalized) and exact match against Master Excel header titles (e.g., 'Source Document Location', 'Engineering Discipline Subcategory'). |
| **`KEDS-TC-015`**<br>Type and paste the requirement text | Playwright UI + OpenPyXL | `PRE-1`, `PRE-2`<br>`TD-01` Title, `TD-02` Detail | 1. Type `TD-01` Title (with line break) into Title / Requirement; assert exact preservation.<br>2. Paste `TD-02` Detail Description; assert line breaks and numbering (1.3, 1.3.1, 1.3.1.1) remain intact.<br>3. Select 'Use original text', complete `TD-01` attributes, submit, and assert Master Excel Columns F and G match input text verbatim. |
| **`KEDS-TC-016`**<br>Text fields keep focus & position on long edits | Playwright UI | `PRE-2`<br>`TD-02` | 1. Type into Detail Description with simulated 60-second pauses between words; assert active element stays focused and zero characters are dropped.<br>2. Select 'Use edited custom text', type spaces in Final Requirement Text, and assert focus does not jump to Created Date.<br>3. Paste `TD-02`, scroll to midpoint, wait 30s, and assert `scrollTop` delta == 0. |
| **`KEDS-TC-023`**<br>Template changes picked up dynamically | OpenPyXL + Playwright UI | `OP-06` (test template copy) | 1. Remove 'STY- Ergonomics & Human Factors' from test template Selection List and save.<br>2. Start new submission with Engineering Discipline = 'Safety'; assert removed value is absent while other 10 'STY-' values appear.<br>3. Restore value in template, start new submission, and assert value reappears plus updated Responsible Organization title reflects template. |
| **`KEDS-TC-027`**<br>Mandatory attributes can't be skipped | Playwright UI (Parameterized) | `PRE-2`<br>`TD-01` (12 omission variants) | 1. Run `@pytest.mark.parametrize` across all 12 mandatory fields (Title, Detail Description, Customer Name, Source Doc Version#, Location, Responsible Org, Product Family, Source of Req, DFX Category, Subsystem, Eng Discipline, Subcategory when Safety).<br>2. Omit target field and click Continue; assert chatbot names missing attribute, explains requirement, blocks Review, and advances once filled. |
| **`KEDS-TC-028`**<br>Change an earlier answer before submitting | Playwright UI + OpenPyXL | `PRE-1`<br>`TD-01` | 1. Fill `TD-01` attributes up to Engineering Discipline Subcategory.<br>2. Navigate back, change Responsible Organization from 'DE PLC' to 'DE Compliance Tool', and assert later answers remain intact.<br>3. Submit and assert Master Excel Column M == 'DE Compliance Tool'. |
| **`KEDS-TC-029`**<br>Responsible Organization: current list | Playwright UI + OpenPyXL | `PRE-1`, `PRE-3`<br>`TD-01` | 1. Open Responsible Organization dropdown and assert exact set equality with the 11 approved values.<br>2. Click 'DE PLC', then click 'DE Training'; assert single-selection behavior.<br>3. Submit with `TD-01` and assert Master Excel Column M == 'DE Training'. |
| **`KEDS-TC-030`**<br>Older Responsible Organization values excluded | Playwright UI | `PRE-3`<br>None | 1. Open Responsible Organization dropdown.<br>2. Assert legacy values 'Installation & Acceptance', 'Manufacturing', and 'Service' are NOT present in the option list. |
| **`KEDS-TC-064`**<br>Downloaded report keeps one item number per line | Playwright UI + File Parser | `PRE-1`<br>`TD-02` Detail Description | 1. Submit requirement using `TD-02` and click 'Open report'; assert each item number (1.3, 1.3.1, 1.3.1.1) starts on a new line.<br>2. Click 'Download report', capture file via `page.expect_download()`, parse paragraphs, and assert identical line breaks. |
| **`KEDS-TC-065`**<br>Output workbook is readable (height & freeze) | Playwright UI + OpenPyXL | `PRE-1`<br>`TD-02` Detail Description | 1. Submit requirement using `TD-02` and load updated Master Excel via OpenPyXL.<br>2. Assert `row_dimensions[new_row].height` accommodates full multi-line Detail Description.<br>3. Assert `ws.freeze_panes` is configured on header boundary so headers stay pinned during vertical scroll. |
| **`KEDS-TC-066` & `TC-067`**<br>Master Excel append integrity & File Imports | Playwright UI + OpenPyXL | `PRE-1`, `PRE-2`<br>`TD-01`, Samsung Check Sheet, Word/PPT/PDF | 1. Snapshot pre-run Master Excel row count and SHA-256 hashes of rows 9, 623, and 710.<br>2. Submit `TD-01`; assert `new_row_count == old_row_count + 1` and rows 9, 623, 710 are 100% unchanged.<br>3. Import Samsung Commonspec Check Sheet (v31.xlsx), Word, PowerPoint, and PDF files; assert draft requirements generate (including row 9 match) and require explicit Submit. |

### 3.1 Reference Implementation: Page Object & Pytest Execution Code

*Listing 1: Reusable Playwright Page Object encapsulating KEDS chatbot actions (`pages/submission_page.py`)*

```python
# pages/submission_page.py
from playwright.sync_api import Page, expect

class KedsSubmissionPage:
    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url

    def open_and_start_submission(self):
        self.page.goto(self.base_url, wait_until="networkidle")
        self.page.get_by_role("button", name="Start a new submission").click()

    def enter_requirement_text(self, title: str, detail_description: str, mode: str = "Use original text"):
        self.page.get_by_label("Title / Requirement").fill(title)
        self.page.get_by_label("Detail Description").fill(detail_description)
        self.page.get_by_role("button", name=mode).click()

    def submit_and_download_report(self, download_dir: str) -> str:
        self.page.get_by_role("button", name="Submit").click()
        expect(self.page.get_by_text("Submission Report and Workbook Updated")).to_be_visible(timeout=30000)
        
        with self.page.expect_download() as download_info:
            self.page.get_by_role("button", name="Download report").click()
        
        download = download_info.value
        saved_path = f"{download_dir}/{download.suggested_filename}"
        download.save_as(saved_path)
        return saved_path
```

---

## 4. Input Data Repository & Expected Results Architecture

### 4.1 Where the Input Data Repository Lives & Governance Rules

The Input Data Repository is hosted inside the dedicated Azure DevOps Git repository (`keds-qa-automation`) under a structured `test-data/` directory, version-controlled alongside the test code. Keeping test inputs in Git guarantees that every test run is reproducible, auditable, and synchronized with specific versions of the KEDS User Stories & Test Cases document (e.g., v2.0 vs. v2.1).

1.  **Versioned Text & YAML Datasets (`test-data/datasets/`):** Stores structured YAML representations of BRD Section 2.4 test datasets (`TD-01` standard submission payload, `TD-02` multi-line hierarchical requirement text with 1.3 / 1.3.1 / 1.3.1.1 numbering, `TD-03` dictation test strings).
2.  **Read-Only Golden Excel Baselines (`test-data/golden_baselines/`):** Stores an immutable, sanitized reference copy of `Rakesh KEDS Requirements Master Excel.xlsx` (containing baseline rows 9, 623, and 710) and the test Selection List template (`OP-06`). Before each test run, a Pytest fixture clones this golden file into an isolated run workspace (`artifacts/sandbox_<run_id>/`) so live production Excel files are never modified.
3.  **Multi-Format Document Fixtures (`test-data/import_documents/`):** Stores the exact binary files required by `KEDS-TC-067`: Samsung Commonspec Check Sheet(ESH,8C,ELE) v31.xlsx, sample KLA BKM Word (.docx), PowerPoint (.pptx), and SEMI S2 PDF (.pdf) documents.
4.  **Secure Credential Vault (Azure DevOps Variable Groups & Local `.env`):** KLA test account identities (`TD-09` accounts where HR display name differs from email, plus standard and privileged accounts for `PRE-2` / `PRE-3`) are NEVER committed to Git. They are injected at runtime via Azure DevOps Secret Variable Groups or a gitignored `.env` file.

*Listing 2: Physical directory layout of the KEDS Input Data & Automation Repository in Azure DevOps Git*

```text
keds-qa-automation/
├── azure-pipelines.yml                  # Azure DevOps CI/CD pipeline definition
├── pytest.ini                           # Pytest markers, timeouts, and JUnit XML defaults
├── config/
│   └── environments.yaml                # KEDS URLs, timeouts, and sandbox workbook paths
├── test-data/                           # <--- INPUT DATA & EXPECTED RESULTS REPOSITORY
│   ├── datasets/
│   │   ├── td01_standard_submission.yaml
│   │   ├── td02_multiline_detail.yaml
│   │   └── td09_account_profiles.yaml   # References secret env vars (no raw passwords)
│   ├── expected_results/
│   │   ├── dropdown_lists.yaml          # Exact 11 Responsible Org values & 3 excluded legacy values
│   │   ├── excel_column_mapping.yaml    # Maps TD-01 fields -> Columns C, D, E, F, G, M
│   │   └── mandatory_attributes.yaml    # 12 attribute variants for KEDS-TC-027
│   ├── golden_baselines/
│   │   ├── Rakesh_KEDS_Requirements_Master_Excel_BASELINE.xlsx
│   │   └── KEDS_Template_Selection_List_TEST.xlsx
│   └── import_documents/
│       ├── Samsung_Commonspec_Check_Sheet_v31.xlsx
│       ├── sample_kla_bkm_dfx.docx
│       ├── sample_kla_dfx_deck.pptx
│       └── sample_semi_s2_standard.pdf
├── pages/                               # Reusable Playwright Page Objects
├── clients/                             # Black-box REST API backup clients
├── validators/                          # OpenPyXL, report parser, and DOM diff engines
└── tests/                               # Pytest test suites (test_keds_tc_001.py ... tc_068.py)
```

### 4.2 Expected Results Specification Format

Expected results are externalized in `test-data/expected_results/` as machine-readable YAML assertions so QA engineers can update expected lists without modifying Python test logic:

*Listing 3: Machine-readable Expected Results specification for KEDS-TC-029, TC-030, and Master Excel columns*

```yaml
# test-data/expected_results/dropdown_lists.yaml
responsible_organization:
  expected_count: 11
  allowed_values:
    - "DE PLC"
    - "DE Compliance Tool"
    - "DE Training"
    - "Engineering Services (Enovia)"
    - "Manufacturing (eQuality)"
    - "Packaging & Shipping"
    - "Installation (iDeliver,eQuality)"
    - "Qualification (iDeliver)"
    - "Service (CRM/SAP)"
    - "Supply Chain"
    - "Other"
  forbidden_legacy_values:
    - "Installation & Acceptance"
    - "Manufacturing"
    - "Service"

# test-data/expected_results/excel_column_mapping.yaml
master_excel_columns:
  created_date: "C"
  originator_email: "D"
  originator_name: "E"
  title_requirement: "F"
  detail_description: "G"
  responsible_organization: "M"
```

---

## 5. Test Outputs & Automated Comparison Methodology

Executing the KEDS automation suite produces four distinct output artifacts. The validation layer inspects each output programmatically and compares it against the expected baseline to emit deterministic PASS / FAIL verdicts.

| Output Stream | Generated Artifact / Target | Comparison Algorithm & Assertion Logic | Covered Test Cases |
| :--- | :--- | :--- | :--- |
| **1. Mutated Master Excel Workbook** | Sandbox copy of `Rakesh KEDS Requirements Master Excel.xlsx` | **Pre/Post Snapshot Diff via OpenPyXL:**<br>• Row Count Delta: `assert after_rows == before_rows + 1`<br>• Cell Exact Match: assert Columns C, D, E, F, G, M match TD-01 & SSO profile<br>• Immutability Hash: SHA-256 hash of rows 9, 623, 710 before == after<br>• Layout Check: `assert ws.freeze_panes is not None` and `row_dimensions[last].height >= expected_min` | KEDS-TC-003, TC-005, TC-006, TC-015, TC-028, TC-029, TC-065, TC-066 |
| **2. Downloaded Submission Report** | Exported report file (`.docx` / `.pdf`) saved via Playwright download listener | **Document Structure & Line-Break Diff:**<br>• Parses paragraphs/lines via `python-docx` / `pypdf`<br>• Asserts full KLA email appears without truncation ('...')<br>• Asserts hierarchical item numbers (1.3, 1.3.1, 1.3.1.1) start on separate lines (`\n` preserved) | KEDS-TC-005, TC-006, TC-064 |
| **3. Live Chatbot DOM & Network State** | Browser DOM elements, CSS properties, scroll metrics, HTTP responses | **Web-First Playwright & Network Assertions:**<br>• Set Equality: `set(ui_options) == set(expected_11_values)`<br>• Negative Set Check: `set(ui_options).isdisjoint(forbidden_legacy_values)`<br>• Temporal Minute Match: `created_date_str == T3.strftime('%Y-%m-%d %H:%M')`<br>• Scroll/Focus Lock: `page.evaluate('el.scrollTop')` unchanged after 30s | KEDS-TC-001, TC-002, TC-004, TC-013, TC-014, TC-016, TC-023, TC-027, TC-030, TC-067 |
| **4. CI/CD Evidence & Execution Reports** | `junit-results.xml`, Playwright `trace.zip`, failure PNGs, Section 7 summary | **Automated Pipeline Aggregation:**<br>• Azure DevOps `PublishTestResults@2` parses JUnit XML into Test Plans dashboard<br>• Failed tests automatically attach full-page PNG screenshot + interactive Playwright `trace.zip` | All Test Cases (TC-001 through TC-068) |

### 5.1 Reference Implementation: OpenPyXL Workbook Diff & Immutability Validator

*Listing 4: Automated OpenPyXL comparison validator enforcing KEDS-TC-015, TC-065, and TC-066*

```python
# validators/excel_validator.py
import hashlib
import openpyxl

def snapshot_row_hashes(xlsx_path: str, sheet_name: str, row_indices: list[int]) -> dict[int, str]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    hashes = {}
    for r in row_indices:
        row_vals = [str(cell.value) for cell in ws[r]]
        hashes[r] = hashlib.sha256("|".join(row_vals).encode("utf-8")).hexdigest()
    return hashes

def verify_keds_submission_row(
    before_xlsx: str, 
    after_xlsx: str, 
    sheet_name: str, 
    expected_row: dict, 
    protected_rows: list[int] = [9, 623, 710]
):
    before_hashes = snapshot_row_hashes(before_xlsx, sheet_name, protected_rows)
    after_hashes = snapshot_row_hashes(after_xlsx, sheet_name, protected_rows)
    assert before_hashes == after_hashes, f"Protected rows mutated! Diff: {before_hashes} vs {after_hashes}"

    wb_before = openpyxl.load_workbook(before_xlsx, data_only=True)[sheet_name]
    wb_after = openpyxl.load_workbook(after_xlsx, data_only=True)[sheet_name]
    
    assert wb_after.max_row == wb_before.max_row + 1, "Expected exactly 1 appended requirement row"
    new_row_idx = wb_after.max_row
    
    for col_letter, expected_val in expected_row.items():
        actual_val = wb_after[f"{col_letter}{new_row_idx}"].value
        assert actual_val == expected_val, f"Column {col_letter} mismatch: expected {expected_val!r}, got {actual_val!r}"

    # KEDS-TC-065: Verify freeze panes and readable row height
    assert wb_after.freeze_panes is not None, "Header freeze panes missing in output workbook"
    assert (wb_after.row_dimensions[new_row_idx].height or 0) >= 30, "Row height too small for multi-line text"
```

---

## 6. Azure DevOps CI/CD Pipeline & Implementation Roadmap

Because the KEDS test environment (`https://test-dt-infra.dev.dias.kla.com/hosted/eng_reqs`) is restricted to the KLA corporate network (`KEDS-TC-001` / `TC-002`), the Azure DevOps pipeline must execute on a KLA self-hosted agent pool inside the intranet/VPN boundary rather than a public Microsoft-hosted runner.

*Listing 5: Azure DevOps YAML Pipeline configured for KLA intranet execution and native JUnit reporting*

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main

schedules:
  - cron: "0 2 * * 1-5"
    displayName: "Nightly KEDS Regression Run"
    branches:
      include:
        - main
    always: true

pool:
  name: 'KLA-Intranet-QA-Agents'   # Self-hosted agent inside KLA corporate network

variables:
  - group: 'KEDS-QA-Secrets'        # Stores KEDS_BASE_URL, TD09_TEST_USER, TD09_TEST_PASS

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'
    displayName: 'Configure Python 3.11'

  - script: |
      python -m pip install --upgrade pip
      pip install pytest pytest-playwright pytest-xdist openpyxl python-docx pypdf pyyaml requests pydantic python-dotenv
      playwright install chromium
    displayName: 'Install Automation Dependencies & Chromium'

  - script: |
      pytest tests/ --junitxml=artifacts/junit-results.xml --tracing=retain-on-failure --screenshot=only-on-failure
    env:
      KEDS_BASE_URL: $(KEDS_BASE_URL)
      TD09_TEST_USER: $(TD09_TEST_USER)
      TD09_TEST_PASS: $(TD09_TEST_PASS)
    displayName: 'Execute KEDS Automated Test Suite'

  - task: PublishTestResults@2
    inputs:
      testResultsFormat: 'JUnit'
      testResultsFiles: 'artifacts/junit-results.xml'
      testRunTitle: 'KEDS AI v0.1 Automated Regression'
      failTaskOnFailedTests: true
    condition: succeededOrFailed()
    displayName: 'Publish Test Results to Azure DevOps'

  - task: PublishBuildArtifacts@1
    inputs:
      PathtoPublish: 'artifacts'
      ArtifactName: 'KEDS-Test-Evidence-And-Workbooks'
    condition: failed()
    displayName: 'Archive Failure Screenshots, Traces & Output Excel'
```

### 6.1 Four-Phase Rollout Schedule

| Phase | Deliverable Scope | Target Test Cases | Exit Criteria |
| :--- | :--- | :--- | :--- |
| **Phase 1**<br>(Days 1–3) | Repository setup in Azure DevOps Git, Playwright + Pytest installation, SSO `storage_state` authentication helper, and golden Excel baseline isolation. | `KEDS-TC-001`, `TC-002` | Playwright authenticates to KEDS URL via KLA SSO and clones sandbox Master Excel copy cleanly. |
| **Phase 2**<br>(Days 4–8) | Build `KedsSubmissionPage` Page Object, implement UI identity/field validation tests, and parameterize the 12 mandatory attribute variants. | `KEDS-TC-003`, `TC-004`, `TC-013`, `TC-014`, `TC-015`, `TC-016`, `TC-027` | All UI form, focus/scroll retention, and 12 mandatory attribute checks pass locally in headless mode. |
| **Phase 3**<br>(Days 9–13) | Implement OpenPyXL workbook diff engine, report line-break parser, dropdown list validators, and multi-format file import tests. | `KEDS-TC-005`, `TC-006`, `TC-023`, `TC-028`, `TC-029`, `TC-030`, `TC-064`, `TC-065`, `TC-066`, `TC-067` | Automated Excel row/hash diffing and downloaded report parsing pass with zero false positives. |
| **Phase 4**<br>(Days 14–15) | Commit `azure-pipelines.yml`, configure KLA self-hosted agent & ADO Variable Group, and link automated runs to ADO Test Plans. | Full Regression Suite + Backup API Smoke Checks | Nightly pipeline publishes JUnit XML results and failure traces directly in Azure DevOps. |