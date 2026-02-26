import { describe, it, expect, vi, beforeEach } from "vitest";
import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";

// Keys from AnthropicProviderSettings that we forward from the Anthropic SDK client.
type ForwardedAnthropicKeys = "baseURL" | "apiKey" | "headers" | "fetch";

// We don't set `name` or `generateId` — they're provider-internal concerns.
type ControlledAnthropicKeys = "name" | "generateId";

// Compile-time exhaustiveness check: every key in AnthropicProviderSettings
// must be accounted for. If this line errors, a new key was added.
type _exhaustive =
  Exclude<
    keyof AnthropicProviderSettings,
    ForwardedAnthropicKeys | ControlledAnthropicKeys
  > extends never
    ? true
    : {
        error: "AnthropicProviderSettings has unhandled keys";
        unhandled: Exclude<
          keyof AnthropicProviderSettings,
          ForwardedAnthropicKeys | ControlledAnthropicKeys
        >;
      };
const _check: _exhaustive = true;

const mockProviderFn = vi.fn().mockReturnValue({ modelId: "test-model" });
const mockCreateAnthropic = vi.fn().mockReturnValue(mockProviderFn);

vi.mock("@ai-sdk/anthropic", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@ai-sdk/anthropic")>();
  return { ...actual, createAnthropic: mockCreateAnthropic };
});

vi.mock("@anthropic-ai/sdk", () => {
  return {
    default: class MockAnthropic {
      baseURL: string;
      apiKey: string;
      _options: Record<string, any>;
      messages = { create: vi.fn() };

      constructor(opts: any = {}) {
        this.baseURL = opts.baseURL ?? "https://api.anthropic.com/v1";
        this.apiKey = opts.apiKey ?? "default-key";
        this._options = {
          defaultHeaders: opts.defaultHeaders,
          fetch: opts.fetch,
          ...opts,
        };
      }
    },
  };
});

describe("AnthropicAdapter.getLanguageModel()", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("forwards all provider-relevant options from the Anthropic SDK client", async () => {
    const { AnthropicAdapter } =
      await import("../../../src/service-adapters/anthropic/anthropic-adapter");
    const Anthropic = (await import("@anthropic-ai/sdk")).default;

    const customFetch = vi.fn();
    const anthropic = new Anthropic({
      apiKey: "sk-ant-test",
      baseURL: "https://proxy.example.com/v1",
      defaultHeaders: { "x-custom": "value" },
      fetch: customFetch,
    });

    const adapter = new AnthropicAdapter({
      anthropic: anthropic as any,
      model: "claude-3-5-sonnet-latest",
    });
    adapter.getLanguageModel();

    expect(mockCreateAnthropic).toHaveBeenCalledOnce();
    const settings = mockCreateAnthropic.mock.calls[0][0];

    expect(settings.baseURL).toBe("https://proxy.example.com/v1");
    expect(settings.apiKey).toBe("sk-ant-test");
    expect(settings.headers).toEqual({ "x-custom": "value" });
    expect(settings.fetch).toBe(customFetch);

    expect(mockProviderFn).toHaveBeenCalledWith("claude-3-5-sonnet-latest");
  });

  it("works with default Anthropic config (no custom options)", async () => {
    const { AnthropicAdapter } =
      await import("../../../src/service-adapters/anthropic/anthropic-adapter");
    const Anthropic = (await import("@anthropic-ai/sdk")).default;

    const anthropic = new Anthropic({ apiKey: "sk-ant-default" });
    const adapter = new AnthropicAdapter({ anthropic: anthropic as any });
    adapter.getLanguageModel();

    const settings = mockCreateAnthropic.mock.calls[0][0];
    expect(settings.baseURL).toBe("https://api.anthropic.com/v1");
    expect(settings.apiKey).toBe("sk-ant-default");
  });
});
