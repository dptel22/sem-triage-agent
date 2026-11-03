"""Automated Browser Verification Suite: SEM Metrology & Triage Command Center.

Validates the overhauled frontend across all 5 core user journeys against app.py:
1. Journey 1: Initial Page Load & Industrial Layout Hygiene (strict 100vh lock, 0 scrollbar, 14 dies).
2. Journey 2: Sample Queue Selection & C1 Perturbation (die switching, empty scan handling, C1 toggles).
3. Journey 3: Vector Overlay Toggles & Canvas Compositing (6 layer toggles, HUD coordinate tracking).
4. Journey 4: Inspection Execution & Dual-Model Arena (POST /api/inspect < 1.0s, dual card metrics, concordance).
5. Journey 5: Benchmark Matrix Dynamic Data Retrieval (GET /api/benchmark-summary, full telemetry matrix, 0 console/network errors).

Outputs:
  - Screenshots to outputs/browser_verification/*.png
  - Exit code 0 if all assertions and SLA gates pass cleanly.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "browser_verification"
BASE_URL = "http://127.0.0.1:8000/"


def is_server_responsive(url: str = BASE_URL, timeout: float = 1.0) -> bool:
    """Checks whether the application server is responding with HTTP 200."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server_if_needed() -> Optional[subprocess.Popen]:
    """Launches uvicorn in a subprocess if the server is not already listening."""
    if is_server_responsive():
        print(f"[SERVER] Server at {BASE_URL} is already active and responsive.")
        return None

    print(f"[SERVER] Server at {BASE_URL} not responsive. Spawning uvicorn app:app --port 8000...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    deadline = time.time() + 20.0
    while time.time() < deadline:
        if is_server_responsive(timeout=1.0):
            print(f"[SERVER] Server successfully spawned and responding at {BASE_URL}.")
            return proc
        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError(f"Server at {BASE_URL} failed to respond within 20 seconds.")


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    """Terminates the spawned server process cleanly."""
    if proc is None:
        return
    print("[SERVER] Shutting down spawned uvicorn server...")
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait(timeout=2)
    print("[SERVER] Server process terminated cleanly.")


class BrowserVerificationSuite:
    def __init__(self, base_url: str = BASE_URL, outputs_dir: Path = OUTPUTS_DIR):
        self.base_url = base_url
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.console_errors: List[str] = []
        self.page_errors: List[str] = []
        self.failed_requests: List[str] = []

    def _setup_listeners(self, page: Page) -> None:
        """Hooks listeners for console errors, unhandled exceptions, and failed HTTP requests."""
        def on_console(msg):
            if msg.type == "error":
                text = msg.text
                # Filter benign or known noise if any
                print(f"  [CONSOLE ERROR] {text}")
                self.console_errors.append(text)

        def on_page_error(err):
            err_str = str(err)
            print(f"  [PAGE EXCEPTION] {err_str}")
            self.page_errors.append(err_str)

        def on_response(resp):
            if resp.status >= 400 and not resp.url.endswith("favicon.ico"):
                entry = f"{resp.status} {resp.request.method} {resp.url}"
                print(f"  [NETWORK ERROR] {entry}")
                self.failed_requests.append(entry)

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("response", on_response)

    def run_journey_1(self, page: Page) -> None:
        """Journey 1: Initial Page Load & Industrial Layout Hygiene."""
        print("\n=== Journey 1: Initial Page Load & Industrial Layout Hygiene ===")
        t0 = time.perf_counter()

        response = page.goto(self.base_url, wait_until="domcontentloaded")
        assert response is not None, "Failed to navigate to base URL"
        assert response.status == 200, f"Expected HTTP 200 on GET /, got {response.status}"

        # Wait for queue cards to render
        page.wait_for_selector("#queueList .queue-card", state="attached", timeout=10000)

        # Assert Layout Geometry: Strict 100vh lock with zero outer scrollbar
        scroll_height = page.evaluate("() => document.documentElement.scrollHeight")
        inner_height = page.evaluate("() => window.innerHeight")
        assert (
            scroll_height <= inner_height
        ), f"Outer window scrollbar detected: scrollHeight={scroll_height} > innerHeight={inner_height}"

        # Assert 14 sample cards in queue
        cards = page.locator("#queueList .queue-card")
        card_count = cards.count()
        assert card_count == 14, f"Expected exactly 14 sample cards in queue, got {card_count}"

        # Assert active die defaults to d1147ea9
        active_id = page.locator("#currentSampleIdText").text_content().strip()
        assert active_id == "d1147ea9", f"Expected initial active die d1147ea9, got {active_id}"

        # Assert 480x480 Wafer Canvas
        canvas = page.locator("#semCanvas")
        assert canvas.is_visible(), "Wafer canvas #semCanvas is not visible"
        c_width = page.evaluate("() => document.getElementById('semCanvas').width")
        c_height = page.evaluate("() => document.getElementById('semCanvas').height")
        assert c_width == 480 and c_height == 480, f"Canvas dimension mismatch: {c_width}x{c_height}"

        # Assert zero console errors & zero failed requests during initial load
        assert len(self.console_errors) == 0, f"Console errors on initial load: {self.console_errors}"
        assert len(self.page_errors) == 0, f"Page exceptions on initial load: {self.page_errors}"
        assert len(self.failed_requests) == 0, f"Failed requests on initial load: {self.failed_requests}"

        # Save screenshot
        ss_path = self.outputs_dir / "01_initial_page_load.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Journey 1 completed in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def run_journey_2(self, page: Page) -> None:
        """Journey 2: Sample Queue Selection & C1 Perturbation Handling."""
        print("\n=== Journey 2: Sample Queue Selection & C1 Perturbation Handling ===")
        t0 = time.perf_counter()

        # Step 1: Click 75e64833 (rare class 5 micro-bridge defect)
        with page.expect_response(
            lambda r: "/api/inspect" in r.url and r.request.method == "POST",
            timeout=5000,
        ):
            page.locator("#card-75e64833").click()

        page.wait_for_selector("#currentSampleIdText:has-text('75e64833')", timeout=5000)
        role = page.locator("#currentSampleRoleBadge").text_content().strip()
        assert "rare_c5_unseen" in role, f"Expected role rare_c5_unseen for 75e64833, got {role}"

        # Step 2: Click 404e7770 (class 6 empty scan)
        with page.expect_response(
            lambda r: "/api/inspect" in r.url and r.request.method == "POST",
            timeout=5000,
        ):
            page.locator("#card-404e7770").click()

        page.wait_for_selector("#currentSampleIdText:has-text('404e7770')", timeout=5000)
        verdict = page.locator("#verdictBadge").text_content().strip()
        decision = page.locator("#decisionBadge").text_content().strip()
        gate1_details = page.locator("#gate1Details").text_content().strip()
        assert (
            verdict == "REVISE" or decision == "NO_DEFECT_REIMAGE" or "Empty scan" in gate1_details
        ), f"Empty scan 404e7770 not handled correctly: verdict={verdict}, decision={decision}, gate1={gate1_details}"

        # Step 3: Click 52b03c7a_perturbed (C1 perturbed sample)
        with page.expect_response(
            lambda r: "/api/inspect" in r.url and r.request.method == "POST",
            timeout=5000,
        ):
            page.locator("#card-52b03c7a_perturbed").click()

        page.wait_for_selector("#currentSampleIdText:has-text('52b03c7a_perturbed')", timeout=5000)

        c1_container = page.locator("#c1ToggleContainer")
        perturbed_badge = page.locator("#currentPerturbedBadge")
        assert c1_container.is_visible(), "C1 perturbation toggle container #c1ToggleContainer is not visible"
        assert perturbed_badge.is_visible(), "C1 perturbation badge #currentPerturbedBadge is not visible"

        # Toggle perturbation view
        c1_btn = page.locator("#c1ToggleBtn")
        coords_initial = page.locator("#activeBoxCoordsText").text_content().strip()
        c1_btn.click()
        page.wait_for_timeout(200)

        coords_toggled = page.locator("#activeBoxCoordsText").text_content().strip()
        assert (
            "PERTURBED" in coords_toggled or coords_initial != coords_toggled
        ), f"C1 Toggle did not alter active box readout: initial='{coords_initial}', toggled='{coords_toggled}'"

        # Revert toggle back
        c1_btn.click()
        page.wait_for_timeout(200)

        # Save screenshot
        ss_path = self.outputs_dir / "02_sample_selection_and_perturbed.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Journey 2 completed in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def run_journey_3(self, page: Page) -> None:
        """Journey 3: Vector Overlay Toggles & Canvas Compositing."""
        print("\n=== Journey 3: Vector Overlay Toggles & Canvas Compositing ===")
        t0 = time.perf_counter()

        layers = [
            "#layerSem",
            "#layerPromptBox",
            "#layerUnet",
            "#layerSam",
            "#layerGt",
            "#layerExclusion",
        ]

        # Toggle each overlay off and on
        for lid in layers:
            chk = page.locator(lid)
            assert chk.is_visible(), f"Layer checkbox {lid} not visible"
            chk.uncheck()
            page.wait_for_timeout(60)
            chk.check()
            page.wait_for_timeout(60)

        # Assert no exceptions fired during layer toggling
        assert len(self.page_errors) == 0, f"Page exceptions occurred during layer toggle: {self.page_errors}"

        # Interactive Coordinate HUD test: move mouse over canvas
        canvas = page.locator("#semCanvas")
        box = canvas.bounding_box()
        assert box is not None, "Bounding box for #semCanvas could not be computed"

        # Move cursor to coordinate (240, 240) in the wafer canvas
        page.mouse.move(box["x"] + 240, box["y"] + 240)
        page.wait_for_timeout(200)

        hud_text = page.locator("#canvasCoordsText").text_content().strip()
        assert "X:" in hud_text and "Y:" in hud_text, f"Canvas HUD did not update on mousemove: {hud_text}"

        match = re.search(r"X:\s*(\d+).*Y:\s*(\d+)", hud_text)
        assert match is not None, f"HUD coordinate numbers not parseable in '{hud_text}'"
        x_val, y_val = int(match.group(1)), int(match.group(2))
        assert 0 <= x_val < 480 and 0 <= y_val < 480, f"HUD coordinates out of bounds: ({x_val}, {y_val})"

        # Save screenshot
        ss_path = self.outputs_dir / "03_overlay_toggles.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Journey 3 completed in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def run_journey_4(self, page: Page) -> None:
        """Journey 4: Inspection Execution & Dual-Model Arena."""
        print("\n=== Journey 4: Inspection Execution & Dual-Model Arena ===")
        t0 = time.perf_counter()

        # Switch model selector to arena_both
        model_select = page.locator("#modelSelect")
        model_select.select_option("arena_both")
        page.wait_for_timeout(100)

        # Trigger inspection on d1147ea9
        req_start = time.perf_counter()
        with page.expect_response(
            lambda r: "/api/inspect" in r.url and r.request.method == "POST",
            timeout=5000,
        ) as resp_info:
            page.locator("#card-d1147ea9").click()

        inspect_resp = resp_info.value
        elapsed = time.perf_counter() - req_start

        assert inspect_resp.status == 200, f"POST /api/inspect failed with HTTP {inspect_resp.status}"
        assert elapsed < 1.0, f"POST /api/inspect SLA violation: took {elapsed:.3f}s (required < 1.0s)"

        # Assert Arena Comparison Card displays both models
        page.wait_for_selector("#arenaComparisonCard", state="visible", timeout=5000)
        meta_verdict = page.locator("#arenaMetaVerdict").text_content().strip()
        meta_iou = page.locator("#arenaMetaIoU").text_content().strip()
        baseten_verdict = page.locator("#arenaBasetenVerdict").text_content().strip()
        baseten_iou = page.locator("#arenaBasetenIoU").text_content().strip()
        concordance = page.locator("#arenaAgreementBadge").text_content().strip()

        assert meta_verdict in ("PASS", "FLAG", "REVISE"), f"Unexpected Meta verdict: {meta_verdict}"
        assert "IoU:" in meta_iou, f"Meta IoU readout missing: {meta_iou}"
        assert baseten_verdict in ("PASS", "FLAG", "REVISE"), f"Unexpected Baseten verdict: {baseten_verdict}"
        assert "IoU:" in baseten_iou, f"Baseten IoU readout missing: {baseten_iou}"
        assert (
            "CONCORDAN" in concordance or "DISCORDANT" in concordance
        ), f"Concordance badge text unexpected: {concordance}"

        # Assert Gate 1-4 statuses and verbatim rationale
        g1 = page.locator("#gate1Badge").text_content().strip()
        g2 = page.locator("#gate2Badge").text_content().strip()
        g3 = page.locator("#gate3Badge").text_content().strip()
        g4 = page.locator("#gate4Badge").text_content().strip()
        rationale = page.locator("#rationaleText").text_content().strip()

        assert g1 in ("PASS", "REVISE"), f"Gate 1 badge invalid: {g1}"
        assert g2 in ("INTACT", "REFINED"), f"Gate 2 badge invalid: {g2}"
        assert g3 in ("PASS", "FLAG"), f"Gate 3 badge invalid: {g3}"
        assert g4 in ("PASS", "FLAG", "REVISE"), f"Gate 4 badge invalid: {g4}"
        assert len(rationale) > 10, f"Verbatim rationale missing or truncated: {rationale}"

        # Save screenshot
        ss_path = self.outputs_dir / "04_arena_inspection_execution.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Journey 4 completed in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def run_journey_5(self, page: Page) -> None:
        """Journey 5: Benchmark Matrix Dynamic Data Retrieval."""
        print("\n=== Journey 5: Benchmark Matrix Dynamic Data Retrieval ===")
        t0 = time.perf_counter()

        # Open Benchmark Drawer & wait for /api/benchmark-summary
        with page.expect_response(
            lambda r: "/api/benchmark-summary" in r.url and r.request.method == "GET",
            timeout=5000,
        ) as resp_info:
            page.locator("#benchmarkBtn").click()

        bench_resp = resp_info.value
        assert bench_resp.status == 200, f"GET /api/benchmark-summary failed with HTTP {bench_resp.status}"

        # Assert drawer panel is visible
        page.wait_for_selector("#benchmarkModal .drawer-panel", state="visible", timeout=5000)

        # Assert dynamic table values
        table_text = page.locator("#benchmarkTableBody").text_content()
        assert "muse-spark-1.3" in table_text, "Meta model row missing from benchmark table"
        assert "meta-models/muse-spark-open-1026" in table_text, "Baseten model row missing from benchmark table"
        assert "Baseline 1" in table_text, "Baseline 1 missing from benchmark table"
        assert "Baseline 2" in table_text, "Baseline 2 missing from benchmark table"
        assert "Baseline 3" in table_text, "Baseline 3 missing from benchmark table"

        # Assert Gates 0-3 in matrix
        gates_text = page.locator("#benchmarkGatesStatus").text_content()
        assert "Gate 0" in gates_text, "Gate 0 missing from gates status matrix"
        assert "Gate 1" in gates_text, "Gate 1 missing from gates status matrix"
        assert "Gate 2" in gates_text, "Gate 2 missing from gates status matrix"
        assert "Gate 3" in gates_text, "Gate 3 missing from gates status matrix"

        # Verify overall error hygiene across all journeys
        assert len(self.console_errors) == 0, f"Total console errors encountered: {self.console_errors}"
        assert len(self.page_errors) == 0, f"Total page exceptions encountered: {self.page_errors}"
        assert len(self.failed_requests) == 0, f"Total failed requests encountered: {self.failed_requests}"

        # Save screenshot
        ss_path = self.outputs_dir / "05_benchmark_matrix_summary.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Journey 5 completed in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")


def run_full_suite() -> int:
    """Entry point executing all 5 user journeys with server management and assertions."""
    print("=" * 70)
    print("STARTING AUTOMATED BROWSER VERIFICATION SUITE")
    print("=" * 70)

    server_proc = None
    try:
        server_proc = start_server_if_needed()

        suite = BrowserVerificationSuite()

        with sync_playwright() as p:
            browser: Browser = p.chromium.launch(headless=True)
            context: BrowserContext = browser.new_context(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=1.0,
            )
            page: Page = context.new_page()
            suite._setup_listeners(page)

            suite.run_journey_1(page)
            suite.run_journey_2(page)
            suite.run_journey_3(page)
            suite.run_journey_4(page)
            suite.run_journey_5(page)

            context.close()
            browser.close()

        print("\n" + "=" * 70)
        print("ALL 5 BROWSER USER JOURNEYS PASSED CLEANLY (EXIT CODE 0)")
        print("Console errors: 0 | Page exceptions: 0 | Failed requests: 0")
        print("Screenshots captured in outputs/browser_verification/:")
        for f in sorted(OUTPUTS_DIR.glob("*.png")):
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        print("=" * 70)
        return 0

    except Exception as exc:
        print(f"\n[FATAL VERIFICATION FAILURE] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        stop_server(server_proc)


def test_ui_browser():
    """Pytest hook allowing verify_ui_browser.py to run under pytest."""
    exit_code = run_full_suite()
    assert exit_code == 0, "Automated browser verification suite failed"


if __name__ == "__main__":
    sys.exit(run_full_suite())
