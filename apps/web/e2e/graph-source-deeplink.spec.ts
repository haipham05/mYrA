import {
  test,
  expect,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";

const WEB_ORIGIN = "http://127.0.0.1:3000";
const apiDir = path.resolve(__dirname, "../../api");
const fixturePath = path.resolve(
  __dirname,
  "../../../tests/retrieval_eval/fixtures/devlin2018_bert.pdf",
);
const python = path.join(apiDir, ".venv/bin/python");
const projectName = "Isolated graph source link";
const factId = "isolated-fact-bert-glue";
const quote = "On the GLUE benchmark, BERT achieves an overall score of 80.5%,";

let tempDirectory: string | null = null;
let apiProcess: ChildProcess | null = null;
let workerProcess: ChildProcess | null = null;
let apiOrigin = "";
let projectId = "";
let paperId = "";
let documentHash = "";

function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("Could not allocate a local test port"));
        return;
      }
      server.close(() => resolve(address.port));
    });
  });
}

async function stopProcess(child: ChildProcess | null): Promise<void> {
  if (!child || child.exitCode !== null) return;
  child.kill("SIGTERM");
  await Promise.race([
    new Promise<void>((resolve) => child.once("exit", () => resolve())),
    new Promise<void>((resolve) => setTimeout(resolve, 2000)),
  ]);
}

async function cleanup(): Promise<void> {
  await Promise.all([stopProcess(apiProcess), stopProcess(workerProcess)]);
  apiProcess = null;
  workerProcess = null;
  if (tempDirectory) {
    if (!path.basename(tempDirectory).startsWith("myra-graph-e2e-")) {
      throw new Error("Refusing to remove a non-test directory");
    }
    fs.rmSync(tempDirectory, { recursive: true, force: true });
    tempDirectory = null;
  }
}

async function startIsolatedApi(request: APIRequestContext): Promise<void> {
  if (!fs.existsSync(python) || !fs.existsSync(fixturePath)) {
    throw new Error(
      "The existing API venv and checked-in PDF fixture are required",
    );
  }
  tempDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "myra-graph-e2e-"));
  const port = await findFreePort();
  apiOrigin = `http://127.0.0.1:${port}`;
  const env = {
    ...process.env,
    DATABASE_URL: `sqlite:///${path.join(tempDirectory, "test.db")}`,
    GCS_BUCKET_NAME: "",
    DEEPSEEK_API_KEY: "",
    MYRA_LLM_MODE: "test",
    MYRA_USE_DOCLING: "false",
    MYRA_EMBEDDING_PROVIDER: "deterministic",
    MYRA_RERANKER_PROVIDER: "simple-lexical",
    MYRA_CHECK_MIGRATIONS: "false",
    MYRA_GRAPHRAG_ENABLED: "false",
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    PYTHONPATH: apiDir,
    PYTHONDONTWRITEBYTECODE: "1",
  };
  const init = spawnSync(
    python,
    ["-c", "from app.db.session import create_tables; create_tables()"],
    { cwd: tempDirectory, env, stdio: "ignore" },
  );
  if (init.status !== 0)
    throw new Error("Could not initialize isolated graph test database");
  apiProcess = spawn(
    python,
    [
      "-m",
      "uvicorn",
      "app.main:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
    ],
    { cwd: tempDirectory, env, stdio: "ignore" },
  );
  workerProcess = spawn(
    python,
    ["-m", "app.worker", "--poll-interval", "0.1"],
    { cwd: tempDirectory, env, stdio: "ignore" },
  );
  for (let attempt = 0; attempt < 50; attempt++) {
    if (apiProcess.exitCode !== null || workerProcess.exitCode !== null) {
      throw new Error("Isolated API or worker exited during startup");
    }
    const health = await request.get(`${apiOrigin}/health`).catch(() => null);
    if (health?.ok()) return;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("Isolated API did not become healthy");
}

async function routeToIsolatedApiWithGraphFixture(page: Page): Promise<void> {
  const fact = {
    id: factId,
    project_id: projectId,
    paper_id: paperId,
    paper_title: "devlin2018_bert.pdf",
    generation_id: "isolated-generation",
    predicate: "REPORTS_RESULT",
    subject_key: "bert",
    subject_name: "BERT",
    subject_type: "Model",
    object_key: "glue-score",
    object_name: "GLUE score",
    object_type: "Metric",
    qualifiers: { result_value: 80.5, unit: "%" },
    exact_quote: quote,
    page_number: 5,
    char_start: 0,
    char_end: quote.length,
    document_sha256: documentHash,
    citation: {
      citation_index: 1,
      evidence_id: "isolated-evidence",
      paper_id: paperId,
      page_number: 5,
      bounding_boxes: [],
      quote,
      document_sha256: documentHash,
      parser_version: null,
      anchor_status: "verified",
      anchors: [
        {
          id: "isolated-anchor",
          page_number: 5,
          exact_quote: quote,
          source_element_id: "isolated-element",
          source_char_start: 0,
          source_char_end: quote.length,
          document_sha256: documentHash,
          parser_version: null,
          anchor_status: "verified",
          bounding_boxes: [],
        },
      ],
    },
    anchor_status: "verified",
  };

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const pathname = url.pathname;
    if (pathname.endsWith("/graph/status")) {
      await route.fulfill({
        status: 200,
        json: {
          project_id: projectId,
          graphrag_enabled: true,
          neo4j_available: true,
          node_count: 2,
          fact_count: 1,
          pending_events_count: 0,
          completed_events_count: 1,
          failed_events_count: 0,
        },
      });
      return;
    }
    if (pathname.endsWith("/graph/nodes")) {
      await route.fulfill({
        status: 200,
        json: { items: [], total: 0, limit: 10, skip: 0 },
      });
      return;
    }
    if (pathname.endsWith(`/graph/facts/${factId}`)) {
      await route.fulfill({ status: 200, json: fact });
      return;
    }
    const response = await route.fetch({
      url: `${apiOrigin}${pathname}${url.search}`,
    });
    await route.fulfill({ response });
  });
}

async function assertRealPdfHighlight(page: Page): Promise<void> {
  await expect(page.getByText("Verified M1 Citation")).toBeVisible();
  await expect(page.getByText("Verbatim match")).toBeVisible();
  await expect(page.getByTestId("evidence-highlight").first()).toBeVisible();
  const observed = await page.evaluate((targetQuote) => {
    const layer = document.querySelector<HTMLElement>(
      "[data-testid=pdf-text-layer]",
    );
    if (!layer) throw new Error("PDF text layer is missing");
    const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    const chunks: string[] = [];
    let node: Node | null;
    while ((node = walker.nextNode())) chunks.push(node.textContent ?? "");
    const text = chunks.join(" ").replace(/\s+/g, " ");
    if (!text.includes(targetQuote)) {
      throw new Error(
        "Expected graph quote is absent from the actual PDF layer",
      );
    }
    const highlights = Array.from(
      document.querySelectorAll<HTMLElement>(
        "[data-testid=evidence-highlight]",
      ),
    );
    return {
      quotePresent: text.includes(targetQuote),
      highlightCount: highlights.length,
    };
  }, quote);
  expect(observed.quotePresent).toBe(true);
  expect(observed.highlightCount).toBeGreaterThan(0);
}

test.describe("isolated graph fact source deep link", () => {
  test.beforeAll(async ({ request }) => {
    try {
      await startIsolatedApi(request);
      const projectResponse = await request.post(
        `${apiOrigin}/api/v1/projects`,
        { data: { name: projectName } },
      );
      expect(projectResponse.ok()).toBeTruthy();
      projectId = (await projectResponse.json()).id as string;
      const uploadResponse = await request.post(
        `${apiOrigin}/api/v1/projects/${projectId}/papers`,
        {
          multipart: {
            file: {
              name: "devlin2018_bert.pdf",
              mimeType: "application/pdf",
              buffer: fs.readFileSync(fixturePath),
            },
          },
        },
      );
      expect(uploadResponse.ok()).toBeTruthy();
      const upload = await uploadResponse.json();
      paperId = upload.paper_id as string;
      let ready = false;
      for (let attempt = 0; attempt < 100; attempt++) {
        const response = await request.get(
          `${apiOrigin}/api/v1/jobs/${upload.job_id}`,
        );
        const job = await response.json();
        if (job.status === "COMPLETED") {
          ready = true;
          break;
        }
        if (job.status === "FAILED")
          throw new Error("Isolated PDF ingestion failed");
        await new Promise((resolve) => setTimeout(resolve, 200));
      }
      expect(
        ready,
        "isolated worker should ingest the actual PDF",
      ).toBeTruthy();
      const paperResponse = await request.get(
        `${apiOrigin}/api/v1/papers/${paperId}`,
      );
      expect(paperResponse.ok()).toBeTruthy();
      const paper = await paperResponse.json();
      documentHash = paper.document_sha256 as string;
      expect(documentHash).toMatch(/^[a-f0-9]{64}$/);
    } catch (error) {
      await cleanup();
      throw error;
    }
  });

  test.afterAll(async () => cleanup());

  test("opens and reloads a mocked graph fact against the actual isolated PDF", async ({
    page,
  }) => {
    await routeToIsolatedApiWithGraphFixture(page);
    const deepLink = `${WEB_ORIGIN}/projects/${projectId}/graph?fact=${factId}`;
    await page.goto(deepLink);
    await assertRealPdfHighlight(page);

    await page.reload();
    await assertRealPdfHighlight(page);
  });
});
