// tsup.config.ts
import { defineConfig } from "tsup";
import { generate } from "@graphql-codegen/cli";

// codegen.ts
import path from "node:path";
var __injected_dirname__ = "/Users/arielweinberger/Development/copilotkit/CopilotKit/CopilotKit/packages/runtime-client-gql";
var schema = path.resolve(__injected_dirname__, "../runtime/__snapshots__/schema/schema.graphql");
var config = {
  schema,
  documents: ["src/graphql/definitions/**/*.{ts,tsx}"],
  generates: {
    "./src/graphql/@generated/": {
      preset: "client",
      config: {
        useTypeImports: true,
        withHooks: false
      },
      plugins: []
    }
  },
  hooks: {}
};
var codegen_default = config;

// tsup.config.ts
var runBeforeBuildPlugin = {
  name: "run-before-build",
  setup(build) {
    const prefix = build.initialOptions.format;
    build.onStart(async () => {
      console.log(`[${prefix}] Running graphql-codegen`);
      await generate(codegen_default);
      console.log(`[${prefix}] graphql-codegen completed successfully`);
    });
  }
};
var tsup_config_default = defineConfig((options) => ({
  entry: ["src/**/*.{ts,tsx}"],
  format: ["esm", "cjs"],
  dts: true,
  minify: false,
  external: ["react"],
  sourcemap: true,
  exclude: [
    "**/*.test.ts",
    // Exclude TypeScript test files
    "**/*.test.tsx",
    // Exclude TypeScript React test files
    "**/__tests__/*"
    // Exclude any files inside a __tests__ directory
  ],
  esbuildPlugins: [runBeforeBuildPlugin],
  ...options
}));
export {
  tsup_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidHN1cC5jb25maWcudHMiLCAiY29kZWdlbi50cyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiY29uc3QgX19pbmplY3RlZF9maWxlbmFtZV9fID0gXCIvVXNlcnMvYXJpZWx3ZWluYmVyZ2VyL0RldmVsb3BtZW50L2NvcGlsb3RraXQvQ29waWxvdEtpdC9Db3BpbG90S2l0L3BhY2thZ2VzL3J1bnRpbWUtY2xpZW50LWdxbC90c3VwLmNvbmZpZy50c1wiO2NvbnN0IF9faW5qZWN0ZWRfZGlybmFtZV9fID0gXCIvVXNlcnMvYXJpZWx3ZWluYmVyZ2VyL0RldmVsb3BtZW50L2NvcGlsb3RraXQvQ29waWxvdEtpdC9Db3BpbG90S2l0L3BhY2thZ2VzL3J1bnRpbWUtY2xpZW50LWdxbFwiO2NvbnN0IF9faW5qZWN0ZWRfaW1wb3J0X21ldGFfdXJsX18gPSBcImZpbGU6Ly8vVXNlcnMvYXJpZWx3ZWluYmVyZ2VyL0RldmVsb3BtZW50L2NvcGlsb3RraXQvQ29waWxvdEtpdC9Db3BpbG90S2l0L3BhY2thZ2VzL3J1bnRpbWUtY2xpZW50LWdxbC90c3VwLmNvbmZpZy50c1wiO2ltcG9ydCB7IGRlZmluZUNvbmZpZywgT3B0aW9ucyB9IGZyb20gXCJ0c3VwXCI7XG5pbXBvcnQgeyBQbHVnaW4gfSBmcm9tIFwiZXNidWlsZFwiO1xuaW1wb3J0IHsgZ2VuZXJhdGUgfSBmcm9tIFwiQGdyYXBocWwtY29kZWdlbi9jbGlcIjtcbmltcG9ydCBjb2RlZ2VuQ29uZmlnIGZyb20gXCIuL2NvZGVnZW5cIjtcblxuY29uc3QgcnVuQmVmb3JlQnVpbGRQbHVnaW46IFBsdWdpbiA9IHtcbiAgbmFtZTogXCJydW4tYmVmb3JlLWJ1aWxkXCIsXG4gIHNldHVwKGJ1aWxkKSB7XG4gICAgY29uc3QgcHJlZml4ID0gYnVpbGQuaW5pdGlhbE9wdGlvbnMuZm9ybWF0O1xuXG4gICAgYnVpbGQub25TdGFydChhc3luYyAoKSA9PiB7XG4gICAgICBjb25zb2xlLmxvZyhgWyR7cHJlZml4fV0gUnVubmluZyBncmFwaHFsLWNvZGVnZW5gKTtcbiAgICAgIGF3YWl0IGdlbmVyYXRlKGNvZGVnZW5Db25maWcpO1xuICAgICAgY29uc29sZS5sb2coYFske3ByZWZpeH1dIGdyYXBocWwtY29kZWdlbiBjb21wbGV0ZWQgc3VjY2Vzc2Z1bGx5YCk7XG4gICAgfSk7XG4gIH0sXG59O1xuXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVDb25maWcoKG9wdGlvbnM6IE9wdGlvbnMpID0+ICh7XG4gIGVudHJ5OiBbXCJzcmMvKiovKi57dHMsdHN4fVwiXSxcbiAgZm9ybWF0OiBbXCJlc21cIiwgXCJjanNcIl0sXG4gIGR0czogdHJ1ZSxcbiAgbWluaWZ5OiBmYWxzZSxcbiAgZXh0ZXJuYWw6IFtcInJlYWN0XCJdLFxuICBzb3VyY2VtYXA6IHRydWUsXG4gIGV4Y2x1ZGU6IFtcbiAgICBcIioqLyoudGVzdC50c1wiLCAvLyBFeGNsdWRlIFR5cGVTY3JpcHQgdGVzdCBmaWxlc1xuICAgIFwiKiovKi50ZXN0LnRzeFwiLCAvLyBFeGNsdWRlIFR5cGVTY3JpcHQgUmVhY3QgdGVzdCBmaWxlc1xuICAgIFwiKiovX190ZXN0c19fLypcIiwgLy8gRXhjbHVkZSBhbnkgZmlsZXMgaW5zaWRlIGEgX190ZXN0c19fIGRpcmVjdG9yeVxuICBdLFxuICBlc2J1aWxkUGx1Z2luczogW3J1bkJlZm9yZUJ1aWxkUGx1Z2luIGFzIGFueV0sXG4gIC4uLm9wdGlvbnMsXG59KSk7XG4iLCAiY29uc3QgX19pbmplY3RlZF9maWxlbmFtZV9fID0gXCIvVXNlcnMvYXJpZWx3ZWluYmVyZ2VyL0RldmVsb3BtZW50L2NvcGlsb3RraXQvQ29waWxvdEtpdC9Db3BpbG90S2l0L3BhY2thZ2VzL3J1bnRpbWUtY2xpZW50LWdxbC9jb2RlZ2VuLnRzXCI7Y29uc3QgX19pbmplY3RlZF9kaXJuYW1lX18gPSBcIi9Vc2Vycy9hcmllbHdlaW5iZXJnZXIvRGV2ZWxvcG1lbnQvY29waWxvdGtpdC9Db3BpbG90S2l0L0NvcGlsb3RLaXQvcGFja2FnZXMvcnVudGltZS1jbGllbnQtZ3FsXCI7Y29uc3QgX19pbmplY3RlZF9pbXBvcnRfbWV0YV91cmxfXyA9IFwiZmlsZTovLy9Vc2Vycy9hcmllbHdlaW5iZXJnZXIvRGV2ZWxvcG1lbnQvY29waWxvdGtpdC9Db3BpbG90S2l0L0NvcGlsb3RLaXQvcGFja2FnZXMvcnVudGltZS1jbGllbnQtZ3FsL2NvZGVnZW4udHNcIjtpbXBvcnQgdHlwZSB7IENvZGVnZW5Db25maWcgfSBmcm9tIFwiQGdyYXBocWwtY29kZWdlbi9jbGlcIjtcbmltcG9ydCBwYXRoIGZyb20gXCJub2RlOnBhdGhcIjtcblxuY29uc3Qgc2NoZW1hID0gcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgXCIuLi9ydW50aW1lL19fc25hcHNob3RzX18vc2NoZW1hL3NjaGVtYS5ncmFwaHFsXCIpO1xuXG5jb25zdCBjb25maWc6IENvZGVnZW5Db25maWcgPSB7XG4gIHNjaGVtYSxcbiAgZG9jdW1lbnRzOiBbXCJzcmMvZ3JhcGhxbC9kZWZpbml0aW9ucy8qKi8qLnt0cyx0c3h9XCJdLFxuICBnZW5lcmF0ZXM6IHtcbiAgICBcIi4vc3JjL2dyYXBocWwvQGdlbmVyYXRlZC9cIjoge1xuICAgICAgcHJlc2V0OiBcImNsaWVudFwiLFxuICAgICAgY29uZmlnOiB7XG4gICAgICAgIHVzZVR5cGVJbXBvcnRzOiB0cnVlLFxuICAgICAgICB3aXRoSG9va3M6IGZhbHNlLFxuICAgICAgfSxcbiAgICAgIHBsdWdpbnM6IFtdLFxuICAgIH0sXG4gIH0sXG4gIGhvb2tzOiB7fSxcbn07XG5cbmV4cG9ydCBkZWZhdWx0IGNvbmZpZztcbiJdLAogICJtYXBwaW5ncyI6ICI7QUFBMmEsU0FBUyxvQkFBNkI7QUFFamQsU0FBUyxnQkFBZ0I7OztBQ0R6QixPQUFPLFVBQVU7QUFEMEgsSUFBTSx1QkFBdUI7QUFHeEssSUFBTSxTQUFTLEtBQUssUUFBUSxzQkFBVyxnREFBZ0Q7QUFFdkYsSUFBTSxTQUF3QjtBQUFBLEVBQzVCO0FBQUEsRUFDQSxXQUFXLENBQUMsdUNBQXVDO0FBQUEsRUFDbkQsV0FBVztBQUFBLElBQ1QsNkJBQTZCO0FBQUEsTUFDM0IsUUFBUTtBQUFBLE1BQ1IsUUFBUTtBQUFBLFFBQ04sZ0JBQWdCO0FBQUEsUUFDaEIsV0FBVztBQUFBLE1BQ2I7QUFBQSxNQUNBLFNBQVMsQ0FBQztBQUFBLElBQ1o7QUFBQSxFQUNGO0FBQUEsRUFDQSxPQUFPLENBQUM7QUFDVjtBQUVBLElBQU8sa0JBQVE7OztBRGhCZixJQUFNLHVCQUErQjtBQUFBLEVBQ25DLE1BQU07QUFBQSxFQUNOLE1BQU0sT0FBTztBQUNYLFVBQU0sU0FBUyxNQUFNLGVBQWU7QUFFcEMsVUFBTSxRQUFRLFlBQVk7QUFDeEIsY0FBUSxJQUFJLElBQUksaUNBQWlDO0FBQ2pELFlBQU0sU0FBUyxlQUFhO0FBQzVCLGNBQVEsSUFBSSxJQUFJLGdEQUFnRDtBQUFBLElBQ2xFLENBQUM7QUFBQSxFQUNIO0FBQ0Y7QUFFQSxJQUFPLHNCQUFRLGFBQWEsQ0FBQyxhQUFzQjtBQUFBLEVBQ2pELE9BQU8sQ0FBQyxtQkFBbUI7QUFBQSxFQUMzQixRQUFRLENBQUMsT0FBTyxLQUFLO0FBQUEsRUFDckIsS0FBSztBQUFBLEVBQ0wsUUFBUTtBQUFBLEVBQ1IsVUFBVSxDQUFDLE9BQU87QUFBQSxFQUNsQixXQUFXO0FBQUEsRUFDWCxTQUFTO0FBQUEsSUFDUDtBQUFBO0FBQUEsSUFDQTtBQUFBO0FBQUEsSUFDQTtBQUFBO0FBQUEsRUFDRjtBQUFBLEVBQ0EsZ0JBQWdCLENBQUMsb0JBQTJCO0FBQUEsRUFDNUMsR0FBRztBQUNMLEVBQUU7IiwKICAibmFtZXMiOiBbXQp9Cg==
