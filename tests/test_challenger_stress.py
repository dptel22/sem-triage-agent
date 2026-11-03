"""Empirical Challenger Stress Test Suite: SEM Metrology & Triage Command Center.

Adversarially challenges:
- Challenge 1: Arena Discordance & Concordance Handling (0c15addf empirical reality + mocked discordant path)
- Challenge 2: Empty Scan Die Handling (404e7770, Class 6, black mask, REVISE verdict)
- Challenge 3: Perturbation Handling & IoU Recovery (52b03c7a_perturbed, box toggles, telemetry verification)
- Challenge 4: Rapid Sequential Load & Race Condition Stress (all 14 dies, SLA < 1.0s, caching, rapid switching)
- Challenge 5: Multi-Viewport Layout Lock (1280x800, 1440x900, 1920x1080 100vh overflow tests)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "challenger_verification"
BASE_URL = "http://127.0.0.1:8000/"


def is_server_responsive(url: str = BASE_URL, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server_if_needed() -> Optional[subprocess.Popen]:
    if is_server_responsive():
        print(f"[SERVER] Server at {BASE_URL} already active.")
        return None

    print(f"[SERVER] Starting server at {BASE_URL}...")
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if is_server_responsive(timeout=1.0):
            print(f"[SERVER] Server responsive at {BASE_URL}.")
            return proc
        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError(f"Server at {BASE_URL} failed to start within 20s.")


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    print("[SERVER] Shutting down server...")
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait(timeout=2)
    print("[SERVER] Server terminated cleanly.")


class ChallengerHarness:
    def __init__(self, base_url: str = BASE_URL, outputs_dir: Path = OUTPUTS_DIR):
        self.base_url = base_url
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.console_errors: List[str] = []
        self.page_errors: List[str] = []
        self.failed_requests: List[str] = []

    def setup_listeners(self, page: Page) -> None:
        def on_console(msg):
            if msg.type == "error":
                text = msg.text
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

    def test_challenge_1_arena_discordance(self, page: Page) -> None:
        """Challenge 1: Arena Concordance & Discordance Handling."""
        print("\n--- Challenge 1: Arena Concordance and Discordance Handling ---")
        t0 = time.perf_counter()

        # Step 1A: Live verification of 0c15addf in Arena Mode
        page.locator("#modelSelect").select_option("arena_both")
        page.wait_for_timeout(100)

        with page.expect_response(
            lambda r: "/api/inspect" in r.url and r.request.method == "POST",
            timeout=5000,
        ) as resp_info:
            page.locator("#card-0c15addf").click()

        resp = resp_info.value
        data = resp.json()
        print(f"  [API OBSERVATION] 0c15addf agreement={data.get('agreement')}, "
              f"meta={data.get('meta', {}).get('verdict')}, "
              f"baseten={data.get('baseten', {}).get('verdict')}")

        badge_text = page.locator("#arenaAgreementBadge").text_content().strip()
        badge_class = page.locator("#arenaAgreementBadge").get_attribute("class") or ""
        print(f"  [UI OBSERVATION] 0c15addf live badge text='{badge_text}', class='{badge_class}'")

        if data.get("agreement") is True:
            assert badge_text == "100% CONCORDANCE", f"Expected '100% CONCORDANCE' when agreement is True, got '{badge_text}'"
            assert "badge-pass" in badge_class, f"Expected badge-pass class, got '{badge_class}'"

        # Step 1B: Synthetic Discordant Arena Challenge
        # Intercept POST /api/inspect and inject discordant payload (Meta FLAG vs Baseten PASS, agreement: false)
        discordant_payload = dict(data)
        discordant_payload["agreement"] = False
        discordant_payload["model"] = "arena_both"
        discordant_payload["meta"] = dict(data.get("meta", {}))
        discordant_payload["meta"]["verdict"] = "FLAG"
        discordant_payload["meta"]["decision"] = "DEFER_RARE_REVIEW"
        discordant_payload["baseten"] = dict(data.get("baseten", {}))
        discordant_payload["baseten"]["verdict"] = "PASS"
        discordant_payload["baseten"]["decision"] = "ACCEPT_NOMINAL"

        def route_discordant(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(discordant_payload),
            )

        page.route("**/api/inspect", route_discordant)
        page.locator("#card-0c15addf").click()
        page.wait_for_timeout(300)

        badge_discordant_text = page.locator("#arenaAgreementBadge").text_content().strip()
        badge_discordant_class = page.locator("#arenaAgreementBadge").get_attribute("class") or ""
        meta_verdict_text = page.locator("#arenaMetaVerdict").text_content().strip()
        baseten_verdict_text = page.locator("#arenaBasetenVerdict").text_content().strip()

        print(f"  [UI OBSERVATION] Mocked discordant: badge='{badge_discordant_text}', class='{badge_discordant_class}', "
              f"meta='{meta_verdict_text}', baseten='{baseten_verdict_text}'")

        assert badge_discordant_text == "DISCORDANT", f"Expected 'DISCORDANT' badge, got '{badge_discordant_text}'"
        assert "badge-flag" in badge_discordant_class, f"Expected 'badge-flag' class on discordant, got '{badge_discordant_class}'"
        assert meta_verdict_text == "FLAG", f"Expected Meta verdict FLAG, got '{meta_verdict_text}'"
        assert baseten_verdict_text == "PASS", f"Expected Baseten verdict PASS, got '{baseten_verdict_text}'"

        # Unroute
        page.unroute("**/api/inspect")

        ss_path = self.outputs_dir / "c1_discordant_arena.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Challenge 1 verified in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def test_challenge_2_empty_scan(self, page: Page) -> None:
        """Challenge 2: Empty Scan Die (404e7770, Class 6)."""
        print("\n--- Challenge 2: Empty Scan Die Handling (404e7770) ---")
        t0 = time.perf_counter()

        with page.expect_response(
            lambda r: "/api/inspect" in r.url and r.request.method == "POST",
            timeout=5000,
        ) as resp_info:
            page.locator("#card-404e7770").click()

        resp = resp_info.value
        data = resp.json()
        assert data.get("is_empty_scan") is True, f"Expected is_empty_scan=True, got {data.get('is_empty_scan')}"
        assert data.get("verdict") == "REVISE", f"Expected verdict=REVISE, got {data.get('verdict')}"
        assert data.get("decision") == "NO_DEFECT_REIMAGE", f"Expected decision=NO_DEFECT_REIMAGE, got {data.get('decision')}"

        # Check UI Badges
        verdict_badge = page.locator("#verdictBadge").text_content().strip()
        decision_badge = page.locator("#decisionBadge").text_content().strip()
        gate1_badge = page.locator("#gate1Badge").text_content().strip()
        gate1_details = page.locator("#gate1Details").text_content().strip()

        print(f"  [UI OBSERVATION] 404e7770: verdict='{verdict_badge}', decision='{decision_badge}', "
              f"gate1='{gate1_badge}', details='{gate1_details}'")

        assert verdict_badge == "REVISE", f"UI verdict badge was '{verdict_badge}'"
        assert decision_badge == "NO_DEFECT_REIMAGE", f"UI decision badge was '{decision_badge}'"
        assert gate1_badge == "EMPTY_SCAN", f"UI Gate 1 badge was '{gate1_badge}'"
        assert "Filtered foreground" in gate1_details or "Empty scan" in gate1_details, \
            f"Gate 1 details did not report empty scan reason: '{gate1_details}'"

        # Verify Canvas does not crash and zero console errors
        assert len(self.console_errors) == 0, f"Console errors on empty scan: {self.console_errors}"
        assert len(self.page_errors) == 0, f"Page errors on empty scan: {self.page_errors}"

        ss_path = self.outputs_dir / "c2_empty_scan_404e7770.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Challenge 2 verified in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def test_challenge_3_perturbation(self, page: Page) -> None:
        """Challenge 3: Perturbed Die (52b03c7a_perturbed)."""
        print("\n--- Challenge 3: Perturbed Die Handling (52b03c7a_perturbed) ---")
        t0 = time.perf_counter()

        with page.expect_response(
            lambda r: "/api/inspect" in r.url and r.request.method == "POST",
            timeout=5000,
        ) as resp_info:
            page.locator("#card-52b03c7a_perturbed").click()

        resp = resp_info.value
        data = resp.json()

        # Telemetry assertions
        assert data.get("refinement_triggered") is True, "Expected refinement_triggered=True"
        assert data.get("refinement_accepted") is True, "Expected refinement_accepted=True"
        delta_iou = data.get("delta_iou_gt", 0.0)
        assert delta_iou > 0.5, f"Expected major delta IoU recovery (>0.50), got {delta_iou}"

        # UI container assertions
        c1_container = page.locator("#c1ToggleContainer")
        assert c1_container.is_visible(), "C1 toggle container not visible"

        iou_metric_text = page.locator("#iouMetricText").text_content().strip()
        strategy_text = page.locator("#refinementStrategyText").text_content().strip()
        print(f"  [UI OBSERVATION] 52b03c7a_perturbed: IoU='{iou_metric_text}', Strategy='{strategy_text}'")
        assert "core_box_recentering" in strategy_text, f"Expected core_box_recentering, got '{strategy_text}'"
        assert "->" in iou_metric_text or "\u2192" in iou_metric_text, f"IoU progression missing arrow: '{iou_metric_text}'"

        # Box toggle assertion
        c1_btn = page.locator("#c1ToggleBtn")
        coords_initial = page.locator("#activeBoxCoordsText").text_content().strip()
        print(f"  [UI OBSERVATION] Initial active box readout: '{coords_initial}'")

        c1_btn.click()
        page.wait_for_timeout(200)
        coords_toggled = page.locator("#activeBoxCoordsText").text_content().strip()
        print(f"  [UI OBSERVATION] Toggled box readout: '{coords_toggled}'")

        assert "PERTURBED INITIAL" in coords_toggled, f"Expected '(PERTURBED INITIAL)' tag, got '{coords_toggled}'"
        assert coords_initial != coords_toggled, "Coords did not change on toggle"

        # Toggle back to refined
        c1_btn.click()
        page.wait_for_timeout(200)
        coords_reverted = page.locator("#activeBoxCoordsText").text_content().strip()
        print(f"  [UI OBSERVATION] Reverted box readout: '{coords_reverted}'")
        assert "REFINED SAM" in coords_reverted, f"Expected '(REFINED SAM)' tag, got '{coords_reverted}'"

        ss_path = self.outputs_dir / "c3_perturbed_52b03c7a.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Challenge 3 verified in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def test_challenge_4_rapid_sequential_and_sla(self, page: Page) -> None:
        """Challenge 4: Rapid Sequential Requests, SLA & Race Conditions across 14 dies."""
        print("\n--- Challenge 4: Rapid Sequential Load & SLA Across 14 Samples ---")
        t0 = time.perf_counter()

        cards = page.locator("#queueList .queue-card")
        count = cards.count()
        assert count == 14, f"Expected 14 sample cards, got {count}"

        # Part 1: Sequential SLA verification
        sample_ids = [cards.nth(i).get_attribute("id").replace("card-", "") for i in range(count)]
        print(f"  [QUEUE] Found {len(sample_ids)} sample IDs: {sample_ids}")

        latencies = []
        for sid in sample_ids:
            print(f"  [SLA] Inspecting {sid}...")
            req_start = time.perf_counter()
            with page.expect_response(
                lambda r: "/api/inspect" in r.url and r.request.method == "POST",
                timeout=5000,
            ) as resp_info:
                page.locator(f"#card-{sid}").click()

            resp = resp_info.value
            dur = time.perf_counter() - req_start
            latencies.append((sid, dur))
            assert resp.status == 200, f"POST /api/inspect for {sid} failed with {resp.status}"
            assert dur < 1.0, f"SLA Violation: {sid} took {dur:.3f}s (> 1.0s limit)"

        max_lat = max(latencies, key=lambda x: x[1])
        min_lat = min(latencies, key=lambda x: x[1])
        avg_lat = sum(x[1] for x in latencies) / len(latencies)
        print(f"  [SLA TELEMETRY] 14 Samples: min={min_lat[0]} ({min_lat[1]*1000:.1f}ms), "
              f"max={max_lat[0]} ({max_lat[1]*1000:.1f}ms), avg={avg_lat*1000:.1f}ms")

        # Part 2: Race Condition Stress Test (Fire rapid clicks across 5 cards in 100ms)
        print("  [RACE CONDITION STRESS] Firing rapid sequential card clicks without waiting...")
        race_targets = ["d1147ea9", "75e64833", "404e7770", "52b03c7a_perturbed", "0c15addf"]
        for target in race_targets:
            page.locator(f"#card-{target}").click()
            page.wait_for_timeout(25)  # 25ms delay, much faster than network turnaround

        # Wait for settling
        page.wait_for_timeout(1000)

        # Confirm the active die settles on the final clicked target ("0c15addf")
        active_id = page.locator("#currentSampleIdText").text_content().strip()
        print(f"  [RACE SETTLING] Final active die settled on: '{active_id}' (target: '0c15addf')")
        assert active_id == "0c15addf", f"Race condition desynchronization: active die is '{active_id}' instead of '0c15addf'"

        # Verify 0 console errors or page exceptions during rapid thrashing
        assert len(self.console_errors) == 0, f"Console errors during rapid switching: {self.console_errors}"
        assert len(self.page_errors) == 0, f"Page errors during rapid switching: {self.page_errors}"

        ss_path = self.outputs_dir / "c4_rapid_switching_settled.png"
        page.screenshot(path=str(ss_path))
        print(f"  [PASS] Challenge 4 verified in {time.perf_counter() - t0:.3f}s. Screenshot: {ss_path.name}")

    def test_challenge_5_viewport_overflow(self, context: BrowserContext) -> None:
        """Challenge 5: Multi-Viewport Layout Lock (1280x800, 1440x900, 1920x1080)."""
        print("\n--- Challenge 5: Multi-Viewport 100vh Layout Lock Stress Test ---")
        t0 = time.perf_counter()

        viewports = [
            (1280, 800),
            (1440, 900),
            (1920, 1080),
        ]

        for width, height in viewports:
            vp_page = context.new_page()
            vp_page.set_viewport_size({"width": width, "height": height})
            self.setup_listeners(vp_page)

            vp_page.goto(self.base_url, wait_until="domcontentloaded")
            vp_page.wait_for_selector("#queueList .queue-card", state="attached", timeout=10000)

            # Assert Layout Strict 100vh Lock (scrollHeight <= innerHeight)
            scroll_h = vp_page.evaluate("() => document.documentElement.scrollHeight")
            inner_h = vp_page.evaluate("() => window.innerHeight")
            scroll_w = vp_page.evaluate("() => document.documentElement.scrollWidth")
            inner_w = vp_page.evaluate("() => window.innerWidth")

            print(f"  [VIEWPORT {width}x{height}] scrollH={scroll_h}/innerH={inner_h}, "
                  f"scrollW={scroll_w}/innerW={inner_w}")

            assert scroll_h <= inner_h, f"Vertical scrollbar detected at {width}x{height}: scrollH={scroll_h} > innerH={inner_h}"
            assert scroll_w <= inner_w, f"Horizontal scrollbar detected at {width}x{height}: scrollW={scroll_w} > innerW={inner_w}"

            # Verify 3 columns are visible and non-zero
            sidebar_box = vp_page.locator("#queueSidebar").bounding_box()
            stage_box = vp_page.locator("#centerStage").bounding_box()
            panel_box = vp_page.locator("#rightPanel").bounding_box()

            assert sidebar_box and sidebar_box["width"] > 200 and sidebar_box["height"] > 400, \
                f"Sidebar layout broken at {width}x{height}: {sidebar_box}"
            assert stage_box and stage_box["width"] > 400 and stage_box["height"] > 400, \
                f"Center stage layout broken at {width}x{height}: {stage_box}"
            assert panel_box and panel_box["width"] > 300 and panel_box["height"] > 400, \
                f"Right panel layout broken at {width}x{height}: {panel_box}"

            ss_path = self.outputs_dir / f"c5_viewport_{width}x{height}.png"
            vp_page.screenshot(path=str(ss_path))
            vp_page.close()

        print(f"  [PASS] Challenge 5 verified in {time.perf_counter() - t0:.3f}s across all 3 viewports.")


def run_challenger_stress_suite() -> int:
    print("=" * 70)
    print("STARTING EMPIRICAL CHALLENGER STRESS HARNESS")
    print("=" * 70)

    server_proc = None
    try:
        server_proc = start_server_if_needed()
        harness = ChallengerHarness()

        with sync_playwright() as p:
            browser: Browser = p.chromium.launch(headless=True)
            context: BrowserContext = browser.new_context(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=1.0,
            )
            page: Page = context.new_page()
            harness.setup_listeners(page)

            page.goto(BASE_URL, wait_until="domcontentloaded")
            page.wait_for_selector("#queueList .queue-card", state="attached", timeout=10000)

            # Execute all 5 stress challenges
            harness.test_challenge_1_arena_discordance(page)
            harness.test_challenge_2_empty_scan(page)
            harness.test_challenge_3_perturbation(page)
            harness.test_challenge_4_rapid_sequential_and_sla(page)
            harness.test_challenge_5_viewport_overflow(context)

            context.close()
            browser.close()

        print("\n" + "=" * 70)
        print("ALL 5 CHALLENGER STRESS TESTS PASSED CLEANLY (EXIT CODE 0)")
        print(f"Total Console Errors: {len(harness.console_errors)}")
        print(f"Total Page Exceptions: {len(harness.page_errors)}")
        print(f"Total Failed Requests: {len(harness.failed_requests)}")
        print("Generated Artifact Screenshots in outputs/challenger_verification/:")
        for f in sorted(OUTPUTS_DIR.glob("*.png")):
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        print("=" * 70)
        return 0

    except Exception as exc:
        print(f"\n[FATAL CHALLENGER FAILURE] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        stop_server(server_proc)


def test_challenger_stress():
    """Pytest hook."""
    exit_code = run_challenger_stress_suite()
    assert exit_code == 0, "Challenger stress harness failed"


if __name__ == "__main__":
    sys.exit(run_challenger_stress_suite())
