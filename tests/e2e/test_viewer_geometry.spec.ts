import { test, expect } from "@playwright/test";
import fs from "fs";
import path from "path";

test.describe("Real-Browser Citation Highlight & Geometry Verification", () => {
  const API_URL = "http://127.0.0.1:8000";
  const WEB_URL = "http://127.0.0.1:3000";

  let projectId: string;
  let paperId: string;

  test.beforeAll(async ({ request }) => {
    // 1. Create a dedicated test project via API
    const projRes = await request.post(`${API_URL}/api/v1/projects`, {
      data: {
        name: `E2E Geometry ${Date.now()}`,
        description: "Browser Geometry Acceptance Test",
      },
    });
    expect(projRes.ok()).toBeTruthy();
    const projData = await projRes.json();
    projectId = projData.id;

    // 2. Upload real PDF fixture (vaswani2017_attention.pdf)
    const pdfPath = path.resolve(
      __dirname,
      "../retrieval_eval/fixtures/vaswani2017_attention.pdf"
    );
    expect(fs.existsSync(pdfPath)).toBeTruthy();
    const pdfBuffer = fs.readFileSync(pdfPath);

    const uploadRes = await request.post(
      `${API_URL}/api/v1/projects/${projectId}/papers`,
      {
        multipart: {
          file: {
            name: "vaswani2017_attention.pdf",
            mimeType: "application/pdf",
            buffer: pdfBuffer,
          },
        },
      }
    );
    expect(uploadRes.ok()).toBeTruthy();
    const uploadData = await uploadRes.json();
    paperId = uploadData.paper_id;
    const jobId = uploadData.job_id;

    // 3. Poll until ingestion job is COMPLETED (or timeout after 60s)
    let completed = false;
    for (let i = 0; i < 30; i++) {
      const jobRes = await request.get(`${API_URL}/api/v1/jobs/${jobId}`);
      if (jobRes.ok()) {
        const job = await jobRes.json();
        if (job.status === "COMPLETED") {
          completed = true;
          break;
        }
        if (job.status === "FAILED") {
          throw new Error(`Ingestion failed: ${job.error_message}`);
        }
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
    expect(completed).toBeTruthy();
  });

  test("Renders real PDF text layer and verifies exact character-span bounding box geometry", async ({
    page,
    request,
  }) => {
    // 1. Navigate to Web UI
    page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));
    page.on("pageerror", (err) => console.log("PAGE ERROR:", err));
    page.on("requestfailed", (req) =>
      console.log("REQ FAILED:", req.url(), req.failure()?.errorText)
    );
    page.on("response", async (res) => {
      console.log("RESP:", res.status(), res.url());
      if (
        res.url().includes("/messages") &&
        res.request().method() === "POST"
      ) {
        try {
          console.log("CHAT_RESPONSE_BODY:", await res.text());
        } catch {
          // Ignore
        }
      }
    });
    await page.goto(WEB_URL);


    // 2. Verify API status connects
    await expect(page.getByText(/Connected/i)).toBeVisible({ timeout: 10000 });

    // 3. Select project
    const projectSelect = page.locator("select").first();
    await projectSelect.selectOption(projectId);

    // 4. Verify paper appears in list and is selected
    const paperButton = page.getByRole("button", { name: "vaswani2017_attention.pdf" });
    await expect(paperButton).toBeVisible({ timeout: 10000 });
    await paperButton.click();


    // 5. Verify PDF renders: Canvas and Text Layer
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible({ timeout: 15000 });

    const textLayer = page.locator('[data-testid="pdf-text-layer"]');
    await expect(textLayer).toBeVisible({ timeout: 10000 });

    // Wait for text spans to populate in the text layer
    await page.waitForFunction(() => {
      const el = document.querySelector('[data-testid="pdf-text-layer"]');
      return el && el.textContent && el.textContent.length > 50;
    }, { timeout: 15000 });

    // Verify initial text layer bounds
    const textLayerBox = await textLayer.boundingBox();
    expect(textLayerBox).not.toBeNull();
    expect(textLayerBox!.width).toBeGreaterThan(200);
    expect(textLayerBox!.height).toBeGreaterThan(200);

    // 6. Ask Question via Chat to generate grounded citation
    const chatInput = page.getByPlaceholder(/Ask a grounded research question/i);
    await expect(chatInput).toBeVisible({ timeout: 10000 });

    await chatInput.fill(
      "What BLEU score did Transformer achieve on WMT 2014 English-to-German?"
    );
    await page.keyboard.press("Enter");


    // 7. Wait for assistant message and citation [1]
    const citationBadge = page.locator("button", { hasText: "[1]" }).first();
    await expect(citationBadge).toBeVisible({ timeout: 30000 });

    // 8. Click citation [1] to activate highlight and verify navigation to page 6
    await citationBadge.click();

    // Verify Page 6 is displayed
    await expect(page.getByText(/Page 6 of/i)).toBeVisible({ timeout: 10000 });

    // Verify active citation callout shows verbatim quote
    await expect(page.getByText(/Cited on Page 6/i)).toBeVisible({ timeout: 10000 });


    // 9. Strict Geometry Assertion on Evidence Highlights
    const highlights = page.locator('[data-testid="evidence-highlight"]');
    await expect(highlights.first()).toBeVisible({ timeout: 10000 });

    const count = await highlights.count();
    expect(count).toBeGreaterThanOrEqual(1);

    const page6TextLayerBox = await textLayer.boundingBox();
    expect(page6TextLayerBox).not.toBeNull();

    for (let i = 0; i < count; i++) {
      const hBox = await highlights.nth(i).boundingBox();
      expect(hBox).not.toBeNull();
      // Ensure positive dimensions
      expect(hBox!.width).toBeGreaterThan(10);
      expect(hBox!.height).toBeGreaterThan(5);

      // Must be atomic single-line or wrapped line, NOT a full-page 500px union box!
      expect(hBox!.height).toBeLessThan(60);

      // Must be strictly positioned within current page text layer bounds
      expect(hBox!.x).toBeGreaterThanOrEqual(page6TextLayerBox!.x - 5);
      expect(hBox!.y).toBeGreaterThanOrEqual(page6TextLayerBox!.y - 5);
      expect(hBox!.x + hBox!.width).toBeLessThanOrEqual(
        page6TextLayerBox!.x + page6TextLayerBox!.width + 5
      );
      expect(hBox!.y + hBox!.height).toBeLessThanOrEqual(
        page6TextLayerBox!.y + page6TextLayerBox!.height + 5
      );
    }

    // 10. Multi-scale zoom verification
    const zoomInBtn = page.getByTitle("Zoom In");
    const initialWidth = (await highlights.first().boundingBox())!.width;

    await zoomInBtn.click();
    await expect
      .poll(
        async () => {
          const box = await highlights.first().boundingBox();
          return box?.width ?? 0;
        },
        { timeout: 10000 }
      )
      .toBeGreaterThan(initialWidth);

    // 11. Disambiguation & False-Box Prevention:
    // Navigate away to Page 3
    const prevBtn = page.getByRole("button", { name: "Previous", exact: true });
    for (let p = 0; p < 3; p++) {
      await prevBtn.click();
      await page.waitForTimeout(300);
    }

    // On Page 3, highlight must NOT be drawn (0 highlights on wrong page)
    await expect(page.getByText(/Page 3 of/i)).toBeVisible({ timeout: 5000 });
    await expect
      .poll(async () => highlights.count(), { timeout: 5000 })
      .toBe(0);

    // Navigate back to Page 6
    const nextBtn = page.getByRole("button", { name: "Next", exact: true });
    for (let p = 0; p < 3; p++) {
      await nextBtn.click();
      await page.waitForTimeout(300);
    }

    await expect(page.getByText(/Page 6 of/i)).toBeVisible({ timeout: 5000 });
    await expect(highlights.first()).toBeVisible({ timeout: 10000 });
  });
});
